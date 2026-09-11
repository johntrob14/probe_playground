"""Explicit --run serial coordinator; otherwise read-only preflight.

Never launches/restarts a rollout server, changes an old queue, or resumes a
partially completed arm. Both leases survive an orphaned owned child process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import prepare_matched_baselines as prep

ROOT, ARTIFACT_ROOT, REPO = prep.ROOT, prep.ARTIFACT_ROOT, prep.REPO
SHARED_LEASE = REPO / "experiments/rank_replication_20260905/coordinator.lock"
storage_guard = prep.storage_guard


@contextmanager
def leases(root=ROOT):
    """Yield both descriptors; children must receive them via pass_fds."""
    prep.check_paths(root)
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("a")) for path in (SHARED_LEASE, Path(root) / "coordinator.lock")]
        for handle in handles:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield tuple(handle.fileno() for handle in handles)


def _query(arguments):
    result = subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(f"GPU0 idle check failed: {result.stderr.strip()}")
    return result.stdout.strip()


def gpu0_idle(max_wait=30):
    """Read-only query of GPU0; bounded waiting only for allocator cleanup."""
    prep.require(0 <= max_wait <= 30, "GPU idle wait must be bounded by 30 seconds")
    started = time.monotonic()
    while True:
        lines = _query(["--id=0", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"]).splitlines()
        prep.require(len(lines) == 1, "GPU0 query did not return exactly one device")
        fields = [field.strip() for field in lines[0].split(",")]
        prep.require(len(fields) == 2 and fields[0].startswith("GPU-"), "Malformed GPU0 identity/memory result")
        uuid, memory = fields[0], int(fields[1])
        prep.require(memory >= 0, "Invalid GPU0 memory usage")
        active = []
        for line in _query(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]).splitlines():
            fields = [field.strip() for field in line.split(",")]
            prep.require(len(fields) == 2 and fields[0].startswith("GPU-") and fields[1].isdigit(),
                         "Malformed compute-process query; cannot establish GPU0 idleness")
            if fields[0] == uuid:
                active.append(int(fields[1]))
        waited = time.monotonic() - started
        if memory <= 256 and not active:
            return {"gpu": 0, "uuid": uuid, "memory_used_mib": memory, "active_compute_pids": [], "waited_seconds": waited}
        if waited >= max_wait:
            raise RuntimeError(f"GPU0 not idle: {memory} MiB, compute PIDs {active}; no process was stopped")
        time.sleep(min(1., max_wait - waited))


def child(command, logfile, env, lease_fds):
    prep.require(len(lease_fds) == 2 and len(set(lease_fds)) == 2, "Both distinct coordinator leases must be inherited")
    with Path(logfile).open("x") as handle:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   pass_fds=tuple(lease_fds))
        print(json.dumps({"event": "child_started", "pid": process.pid, "command": command,
                          "log": str(logfile), "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Child exited {code}; preserve {logfile}; no automatic restart")


def prelaunch(manifest, root=ROOT):
    """Recheck all frozen sources/configs/software, server, storage, GPU0."""
    prep.require(prep.read_json(Path(root) / "manifest.json") == manifest, "Manifest changed before child launch")
    configs = prep.validate_manifest(manifest, root, check_server=False)
    server = prep.pinned_server()
    prep.require(server["argv"] == manifest["rollout_server"]["argv"], "Server command changed")
    resources = storage_guard(root, headroom=prep.WRITE_HEADROOM)
    idle = gpu0_idle()
    print(json.dumps({"event": "prelaunch_verified", "manifest_sha256": prep.digest(Path(root) / "manifest.json"),
                      "rollout_server": server, "gpu0_idle": idle, **resources}), flush=True)
    return configs


def evaluation_command(adapter, destination, cfg, lora_id):
    return [sys.executable, str(REPO / "scripts/eval_replay_replication.py"), "--adapter", str(adapter),
            "--out", str(destination), "--audit-bank", cfg["audit_bank"], "--probe-file", cfg["probe_file"],
            "--task-ids-file", cfg["fresh_test_task_ids"], "--exclude-task-ids-file", cfg["train_task_ids"],
            "--model-id", cfg["model_id"], "--ports", "8005", "--split", "test", "--n", "4",
            "--max-new-tokens", "2048", "--temperature", "1", "--top-p", "1", "--seed", "1234",
            "--lora-id", str(lora_id), "--layer", "15", "--pool", "mean_cot", "--acts-from", "policy,base",
            "--micro-batch", "1", "--device", "cuda", "--length-penalty", "0.003", "--nonterm-penalty", "0",
            "--score-workers", "8", "--test-timeout", "6", "--server-wait", "60"]


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                raise ValueError(f"Blank JSONL record: {path}")
            yield json.loads(line)


def token_hash(row):
    return hashlib.sha256(json.dumps([row["prompt_token_ids"], row["completion_token_ids"]]).encode()).hexdigest()


def verify_evaluation(directory, cfg, adapter, lora_id):
    directory = Path(directory)
    config, result = prep.read_json(directory / "config.json"), prep.read_json(directory / "eval.json")
    expected = {"adapter": str(adapter), "model_id": cfg["model_id"], "probe_file": cfg["probe_file"],
                "audit_bank": cfg["audit_bank"], "n": 4, "max_tokens": 2048, "temperature": 1., "top_p": 1.,
                "seed": 1234, "lora_id": lora_id, "layer": 15, "pool": "mean_cot", "acts_from": "policy,base",
                "ports": "8005", "split": "test", "micro_batch": 1, "device": "cuda", "length_penalty": .003,
                "nonterm_penalty": 0., "score_workers": 8, "test_timeout": 6., "exact_token_ids_retained": True}
    prep.require(all(config.get(k) == v for k, v in expected.items()), "Completed evaluation configuration differs")
    tids = prep.read_json(cfg["fresh_test_task_ids"])
    excluded = prep.read_json(cfg["train_task_ids"])
    prep.require(len(tids) == len(set(tids)) == 120 and len(excluded) == len(set(excluded)) == 300
                 and config.get("task_ids") == tids
                 and config.get("excluded_task_ids") == sorted(excluded) and config.get("exclusion_checked") is True,
                 "Completed evaluation task/exclusion metadata differs")
    prep.require(not set(tids) & set(excluded), "Fresh evaluation overlaps training")
    prep.require(config.get("probe_sha256") == prep.digest(cfg["probe_file"])
                 and config.get("audit_bank_sha256") == prep.digest(cfg["audit_bank"]), "Evaluation probe/bank hash changed")
    prep.require(result.get("n_rollouts") == 480 and result.get("n_tasks") == 120 and result.get("adapter") == str(adapter),
                 "Evaluation completion counts/checkpoint differ")
    fresh, audit = list(rows(directory / "scored.jsonl")), list(rows(directory / "audit_scored.jsonl"))
    identities = [(r["task_id"], r["sample_idx"]) for r in fresh]
    prep.require(len(identities) == 480 and set(identities) == {(t, i) for t in tids for i in range(4)},
                 "Incomplete/duplicate evaluation task-sample grid")
    bank = {}
    for index, row in enumerate(rows(cfg["audit_bank"]), 1):
        bank[row.get("key", f"audit|{row['task_id']}|{index}")] = row
    prep.require(len(audit) == len(bank) == 146 and {r["key"] for r in audit} == set(bank), "Incomplete fixed-audit grid")
    prep.require(not {r["task_id"] for r in audit} & set(excluded)
                 and set(config.get("audit_task_ids", [])) == {r["task_id"] for r in audit}, "Fixed audit task exclusions differ")
    for row in fresh + audit:
        for key in ("prompt_token_ids", "completion_token_ids"):
            prep.require(isinstance(row[key], list) and all(type(v) is int and v >= 0 for v in row[key]), "Invalid saved evaluation tokens")
        prep.require(row["prompt_token_ids"] and len(row["completion_token_ids"]) == row["n_completion_tokens"]
                     and token_hash(row) == row["eval_token_ids_sha256"], "Evaluation token count/hash mismatch")
        for source in ("policy", "base"):
            p, logit = row.get(f"probe_{source}"), row.get(f"probe_logit_{source}")
            if row["n_completion_tokens"]:
                prep.require(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1
                             and type(logit) in (float, int) and math.isfinite(logit), "Evaluation probe scores missing/nonfinite")
            else:
                prep.require(p is None and logit is None, "Empty completion has unexpected scores")
    for row in fresh:
        prep.require(len(row["prompt_token_ids"]) + 2048 <= 4096 and row["n_completion_tokens"] <= 2048,
                     "Evaluation context/token budget changed")
        prep.require(row["finish_reason"] in ("stop", "length"), "Unexpected evaluation finish reason")
        prep.validate_score({k: row[k] for k in ("parsed", "truncated", "reward", "hack", "honest_pass", "shown_pass", "original_pass", "flags")})
    for task_id in tids:
        prompts = [row["prompt_token_ids"] for row in fresh if row["task_id"] == task_id]
        prep.require(all(prompt == prompts[0] for prompt in prompts), "Evaluation prompt differs within task")
    for row in audit:
        original = bank[row["key"]]
        prep.require(row["label"] == original.get("replay_label", original.get("label"))
                     and all(row[k] == original[k] for k in ("task_id", "prompt_text", "completion", "prompt_token_ids", "completion_token_ids")),
                     "Fixed audit labels/text/tokens changed")
    return {"n_fresh": 480, "n_tasks": 120, "n_audit": 146}


def verify_training(cfg):
    run = Path(cfg["run_dir"])
    prep.require(prep.read_json(run / "config.json") == cfg, "Saved training configuration differs")
    complete = prep.read_json(run / "complete.json")
    prep.require(complete.get("steps") == 20 and Path(complete["adapter"]).resolve() == run / "serving_adapter", "Training completion differs")
    for relative in ("serving_adapter/adapter_model.safetensors", "serving_adapter/adapter_config.json",
                     "checkpoint_step_10/adapter_model.safetensors", "checkpoint_step_10/adapter_config.json", "optimizer_final.pt"):
        path = run / relative
        prep.require(path.is_file() and path.stat().st_size > 0, f"Missing retained training artifact: {relative}")
    prep.require(complete.get("final_adapter_sha256") == prep.digest(run / "serving_adapter/adapter_model.safetensors")
                 and complete.get("first_batch_sha256") == cfg["first_batch_sha256"], "Completed training artifact/batch hashes differ")
    metrics = list(rows(run / "metrics.jsonl"))
    prep.require([r["step"] for r in metrics] == list(range(1, 21)), "Incomplete/out-of-order training metrics")
    for row in metrics:
        prep.require(row.get("n") == 128 and row.get("lambda") == cfg["lambda"]
                     and row.get("replay_weight") == cfg["replay_weight"]
                     and row.get("penalty_transform") == cfg["penalty_transform"], "Training metrics treatment/coverage differ")
    bank = prep.read_json(cfg["first_batch_file"])
    count, prompts = 0, {}
    allowed = set(prep.read_json(cfg["train_task_ids"]))
    for index, row in enumerate(rows(run / "rollouts.jsonl")):
        count += 1
        step, position = index // 128 + 1, index % 128
        prep.require(row["step"] == step and row["sample_idx"] == position % 8 and row["task_id"] in allowed,
                     "Training rollout step/task/sample coverage differs")
        key = (step, position // 8)
        identity = (row["task_id"], row["prompt_token_ids"])
        prep.require(key not in prompts or prompts[key] == identity, "Training prompt group differs")
        prompts[key] = identity
        if step == 1:
            prep.require(all(row[k] == v for k, v in bank["rows"][position].items())
                         and all(row[k] == v for k, v in bank["first_scores"][position].items()), "Training did not preserve frozen first batch/scores")
    prep.require(count == 2560 and len(prompts) == 320, "Training rollout count differs from 20x128")
    for step in range(1, 21):
        prep.require(len({prompts[step, group][0] for group in range(16)}) == 16, "Training step repeats a sampled task")
    return complete["adapter"]


def require_initial_precedes_training(configs, root=ROOT):
    """A missing initial eval is not recoverable after any training artifacts."""
    prep.require(not (Path(root) / "eval_initial.log").exists(), "Partial initial evaluation log exists; never automatically restart")
    for name in prep.RUN_ORDER:
        paths = (Path(configs[name]["run_dir"]), ARTIFACT_ROOT / f"eval_{name}",
                 Path(root) / f"train_{name}.log", Path(root) / f"eval_{name}.log")
        prep.require(not any(path.exists() for path in paths), "Initial evaluation must precede every training artifact")


def selection(configs, names, root=ROOT):
    """Existing complete prefix is valid only for an explicit continuation."""
    complete, gap = [], False
    for name in prep.RUN_ORDER:
        run, evaluation = Path(configs[name]["run_dir"]), ARTIFACT_ROOT / f"eval_{name}"
        exists = run.exists() or evaluation.exists() or (Path(root) / f"train_{name}.log").exists() or (Path(root) / f"eval_{name}.log").exists()
        if exists:
            prep.require(not gap, "Existing later artifacts follow an unfinished earlier arm")
            adapter = verify_training(configs[name])
            verify_evaluation(evaluation, configs[name], adapter, 1_700_000_001 + prep.RUN_ORDER.index(name))
            complete.append(name)
        else:
            gap = True
    if names is None:
        prep.require(not complete, "Use explicit --arms for a verified-prefix continuation; no automatic skipping")
        return list(prep.RUN_ORDER)
    prep.require(names and len(names) == len(set(names)), "Require unique, nonempty declared --arms")
    expected = list(prep.RUN_ORDER[len(complete):len(complete) + len(names)])
    prep.require(list(names) == expected, "Selected arms must be the next contiguous declared block, in order")
    return list(names)


def main(root=ROOT, names=None, execute=False, initial_only=False):
    root = Path(root).resolve(strict=True)
    prep.check_paths(root)
    manifest_path = root / "manifest.json"
    manifest_hash = prep.digest(manifest_path)
    manifest = prep.read_json(manifest_path)
    configs = prep.validate_manifest(manifest, root)
    prep.require(not (initial_only and names is not None), "--initial-only and --arms are mutually exclusive")
    initial = ARTIFACT_ROOT / "eval_initial"
    cfg = configs[prep.RUN_ORDER[0]]
    if names is not None:
        prep.require(initial.exists(), "Explicit arm continuation requires a completed initial evaluation")
    if initial.exists():
        verify_evaluation(initial, cfg, cfg["init_adapter"], 1_700_000_000)
        prep.require(not initial_only, "Initial evaluation already exists; never overwrite")
    else:
        require_initial_precedes_training(configs, root)
    chosen = [] if initial_only else selection(configs, names, root)
    if not execute:
        print(json.dumps({"status": "read_only_preflight", "initial_evaluation_needed": not initial.exists(),
                          "selected_arms": chosen, **storage_guard(root, headroom=prep.WRITE_HEADROOM)}))
        return
    prep.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Require explicit CUDA_VISIBLE_DEVICES=0")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME="/ssd1/john/.cache/huggingface",
               TOKENIZERS_PARALLELISM="false", TESTBED_SYSTEM_SUFFIX="", TESTBED_STORE=str(prep.STORE),
               PYTHONPATH=str(REPO / "src"))
    with leases(root) as lease_fds:
        # Re-evaluate state under both leases, not merely before acquisition.
        prep.require(prep.digest(manifest_path) == manifest_hash, "Manifest changed before launch")
        if not initial_only:
            chosen = selection(configs, names, root)
        if not initial.exists():
            require_initial_precedes_training(configs, root)
            prelaunch(manifest, root)
            child(evaluation_command(cfg["init_adapter"], initial, cfg, 1_700_000_000), root / "eval_initial.log", env, lease_fds)
            verify_evaluation(initial, cfg, cfg["init_adapter"], 1_700_000_000)
        else:
            verify_evaluation(initial, cfg, cfg["init_adapter"], 1_700_000_000)
            prep.require(not initial_only, "Initial evaluation appeared before lease acquisition")
        for name in chosen:
            prep.require(prep.digest(manifest_path) == manifest_hash, "Manifest changed during run")
            configs = prelaunch(manifest, root)
            cfg = configs[name]
            run, evaluation = Path(cfg["run_dir"]), ARTIFACT_ROOT / f"eval_{name}"
            if run.exists() or evaluation.exists():
                raise FileExistsError(f"Existing run/evaluation {name}; inspect, never automatically restart")
            child([sys.executable, str(REPO / "scripts" / prep.TRAINER), "--config", str(root / f"config_{name}.json")],
                  root / f"train_{name}.log", env, lease_fds)
            adapter = verify_training(cfg)
            prelaunch(manifest, root)
            lora_id = manifest["evaluation_lora_ids"][name]
            child(evaluation_command(adapter, evaluation, cfg, lora_id), root / f"eval_{name}.log", env, lease_fds)
            verify_evaluation(evaluation, cfg, adapter, lora_id)
            print(json.dumps({"event": "arm_completed", "arm": name, **storage_guard(root)}), flush=True)
        print(json.dumps({"event": "requested_work_complete", "initial_only": initial_only, "arms": chosen}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", action="store_true", help="Explicitly authorize the declared GPU0 serial workflow")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--arms", nargs="+", help="Only the next contiguous block after verified completed predecessors")
    group.add_argument("--initial-only", action="store_true", help="Create the one initial evaluation, then stop")
    args = parser.parse_args()
    main(args.root, args.arms, args.run, args.initial_only)
