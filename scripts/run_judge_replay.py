"""Explicit serial coordinator for the six-run judge-labelled replay study.

Default is offline read-only preflight. --run starts new stages; --resume must
also be explicit to skip a verified completed prefix. Completed training may be
followed by its missing audit/evaluations, but partial stages are never retried,
overwritten, or resumed as training. Every child inherits both GPU0 leases.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/judge_labelled_replay_20260907"
ARTIFACT_ROOT = Path("/ssd1/john/probe_playground/runs/judge_labelled_replay_20260907")
PROTOCOL = "judge_labelled_replay_20260907"
RUN_ORDER = ("prob_none_s613", "prob_exec_replay_s613", "prob_judge_replay_s613",
             "prob_judge_replay_s719", "prob_exec_replay_s719", "prob_none_s719")
SHARED_LEASE = REPO / "experiments/rank_replication_20260905/coordinator.lock"
sys.path.insert(0, str(REPO / "scripts"))
import run_matched_baselines as COMMON
import eval_judge_replay as JUDGE


def preparer():
    import prepare_judge_replay
    return prepare_judge_replay


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def leases(root):
    paths = (SHARED_LEASE, Path(root) / "coordinator.lock")
    descriptors = []
    try:
        for path in paths:
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            descriptors.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield tuple(descriptors)
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def prelaunch(manifest, root):
    """Only called for an explicitly authorized child, never by offline preflight."""
    prep = preparer()
    require(prep.read_json(Path(root) / "manifest.json") == manifest, "Frozen manifest changed")
    configs = prep.validate_manifest(manifest, root, check_server=False)
    servers = prep.pinned_servers({"rollout": manifest["rollout_server"], "judge": manifest["judge_server"]})
    resources = prep.storage_guard(root, headroom=prep.WRITE_HEADROOM)
    idle = COMMON.gpu0_idle(max_wait=30)
    return configs, {"event": "prelaunch_verified", "gpu0_idle": idle, "servers": servers, **resources}


def child(command, logfile, env, lease_fds, emit):
    require(len(lease_fds) == 2 and len(set(lease_fds)) == 2, "Both coordinator leases must survive in the child")
    with Path(logfile).open("x") as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   pass_fds=tuple(lease_fds))
        emit({"event": "child_started", "pid": process.pid, "command": command, "log": str(logfile)})
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                continue
    require(code == 0, f"Child exited {code}; preserve {logfile}; no retry or process termination")


def verify_training(cfg):
    prep = preparer()
    run = Path(cfg["run_dir"])
    require(prep.read_json(run / "config.json") == cfg, "Saved training configuration changed")
    complete = prep.read_json(run / "complete.json")
    adapter = run / "serving_adapter"
    require(complete.get("steps") == 20 and complete.get("adapter") == str(adapter)
            and complete.get("first_batch_sha256") == cfg["first_batch_sha256"]
            and complete.get("replay_label_source") == cfg["replay_label_source"], "Training completion metadata differs")
    for file in ("serving_adapter/adapter_model.safetensors", "serving_adapter/adapter_config.json",
                 "checkpoint_step_10/adapter_model.safetensors", "checkpoint_step_10/adapter_config.json", "optimizer_final.pt"):
        require((run / file).is_file() and (run / file).stat().st_size > 0, f"Missing retained training artifact: {file}")
    require(complete.get("final_adapter_sha256") == prep.digest(adapter / "adapter_model.safetensors"), "Completed adapter hash differs")
    metrics = list(COMMON.rows(run / "metrics.jsonl"))
    require([r["step"] for r in metrics] == list(range(1, 21)), "Training metrics do not cover exactly20 updates")
    for row in metrics:
        require(row.get("n") == 128 and all(row.get(key) == cfg[key] for key in
                ("lambda", "replay_weight", "penalty_transform", "replay_label_source")), "Training treatment/coverage differs")
    bank = prep.read_json(cfg["first_batch_file"])
    require(bank.get("schema_version") == 2 and len(bank["rows"]) == len(bank["first_scores"]) == 128,
            "Shared first bank must be the public-only schema2")
    require(prep.digest(cfg["first_batch_file"]) == cfg["first_batch_sha256"], "Shared first-bank hash differs")
    allowed, prompts, count = set(prep.read_json(cfg["train_task_ids"])), {}, 0
    public = {"parsed", "truncated", "reward", "shown_pass", "code_len"}
    forbidden = {"hack", "honest_pass", "original_pass", "flags", "label", "replay_label"}
    for index, row in enumerate(COMMON.rows(run / "rollouts.jsonl")):
        count += 1
        step, position = index // 128 + 1, index % 128
        require(row["step"] == step and row["sample_idx"] == position % 8 and row["task_id"] in allowed, "Training task/sample/step order differs")
        require(not forbidden.intersection(row) and public <= row.keys(), "Training raw rows expose forbidden execution labels")
        require(type(row["parsed"]) is bool and type(row["truncated"]) is bool and type(row["code_len"]) is int and row["code_len"] >= 0,
                "Invalid public score schema")
        require(isinstance(row["shown_pass"], list) and len(row["shown_pass"]) == 1 and type(row["shown_pass"][0]) is bool,
                "Public score must include only the single shown test")
        reward = max(0., 1. - .003 * row["code_len"]) if all(row["shown_pass"]) else 0.
        require(type(row["reward"]) in (int, float) and math.isclose(row["reward"], reward, abs_tol=1e-12), "Public-only reward arithmetic differs")
        for field in ("prompt_token_ids", "completion_token_ids"):
            require(isinstance(row[field], list) and all(type(t) is int and t >= 0 for t in row[field]), "Invalid saved training token IDs")
        require(row["prompt_token_ids"] and len(row["prompt_token_ids"]) + 2048 <= 4096
                and len(row["completion_token_ids"]) <= 2048 and row["finish_reason"] in ("stop", "length"), "Training token/context/termination mismatch")
        group = (step, position // 8)
        identity = (row["task_id"], row["prompt_token_ids"])
        require(group not in prompts or prompts[group] == identity, "Training group prompt identity changed")
        prompts[group] = identity
        if step == 1:
            require(set(bank["first_scores"][position]) == public and all(row[key] == value for key, value in bank["rows"][position].items())
                    and all(row[key] == value for key, value in bank["first_scores"][position].items()), "Shared first raw batch/public scores changed")
    require(count == 2560 and len(prompts) == 320, "Training must retain all20x128 outputs")
    require(all(len({prompts[step, group][0] for group in range(16)}) == 16 for step in range(1, 21)), "Repeated actor task within update")
    requests = list(COMMON.rows(run / "requests.jsonl"))
    require([r["step"] for r in requests] == list(range(1, 21)), "Saved sampling requests are incomplete")
    for request in requests:
        step = request["step"]
        task_ids = [prompts[step, i][0] for i in range(16)]
        prompt_ids = [prompts[step, i][1] for i in range(16)]
        expected = bank["request"] if step == 1 else {"prompt_token_ids": prompt_ids, "n": 8, "max_tokens": 2048,
            "temperature": 1., "top_p": 1., "seed": cfg["seed"] * 100000 + step,
            "lora_path": str(adapter), "lora_id": cfg["lora_id_base"] + step}
        require(request["request"] == expected and request["task_ids"] == task_ids and request["generated_now"] is (step > 1),
                "Saved request differs from task/seed/adapter/token budget")
        if step == 1:
            require(request["adapter_sha256"] == cfg["init_adapter_sha256"] and request["first_batch_sha256"] == cfg["first_batch_sha256"],
                    "First request differs from the frozen common initialization")
    candidates, summaries, added = [list(COMMON.rows(run / name)) for name in
        ("replay_candidates.jsonl", "replay_refresh_summary.jsonl", "replay_refresh.jsonl")]
    require([r["step"] for r in summaries] == [4, 8, 12, 16, 20], "Refresh summary must retain all five scheduled steps")
    keys = [(r["step"], r["task_id"], r["sample_idx"]) for r in candidates]
    require(len(keys) == len(set(keys)), "Duplicate refresh candidates")
    for summary in summaries:
        selected = [r for r in candidates if r["step"] == summary["step"]]
        require(summary["replay_label_source"] == cfg["replay_label_source"] and summary["selected"] == len(selected) and 16 <= len(selected) <= 32
                and summary["usable_added"] == sum(r["added"] for r in selected), "Refresh candidate/summary coverage differs")
    require(all(r["step"] in (4, 8, 12, 16, 20) and r["replay_label_source"] == cfg["replay_label_source"] for r in candidates)
            and len(added) == sum(r["added"] for r in candidates), "Retained refresh records differ")
    oracle = run / "trainer_refresh_execution.jsonl"
    require(oracle.is_file() if cfg["replay_label_source"] == "execution" else not oracle.exists(), "Oracle-training sidecar appears in the wrong arm")
    return str(adapter)


def verify_execution_audit(cfg):
    prep, run = preparer(), Path(cfg["run_dir"])
    marker = prep.read_json(run / "audit_complete.json")
    require(marker.get("schema_version") == 1 and marker.get("n_rollouts") == 2560 and marker.get("n_steps") == 20,
            "Posttraining execution audit is incomplete")
    candidates = run / "replay_candidates.jsonl"
    expected_sources = {"rollouts": prep.digest(run / "rollouts.jsonl"), "replay_candidates": prep.digest(candidates) if candidates.exists() else None}
    require(marker.get("source_sha256") == expected_sources, "Execution audit references different training inputs")
    expected_paths = {run / name for name in ("execution_audit.jsonl", "refresh_execution_audit.jsonl", "refresh_quality.json")}
    hashes = marker.get("output_sha256", {})
    resolved = {(Path(key) if Path(key).is_absolute() else run / key): value for key, value in hashes.items()}
    require(set(resolved) == expected_paths and all(prep.digest(path) == value for path, value in resolved.items()), "Execution audit output hashes differ")
    require(sum(1 for _ in COMMON.rows(run / "execution_audit.jsonl")) == 2560, "Execution audit row count differs")
    return marker


def verify_evaluation(name, cfg, manifest):
    directory = ARTIFACT_ROOT / f"eval_{name}"
    adapter = cfg["init_adapter"] if name == "initial" else str(Path(cfg["run_dir"]) / "serving_adapter")
    result = COMMON.verify_evaluation(directory, cfg, adapter, manifest["evaluation_lora_ids"][name])
    if name != "initial":
        reference = {r["key"]: r for r in COMMON.rows(ARTIFACT_ROOT / "eval_initial/audit_scored.jsonl")}
        for row in COMMON.rows(directory / "audit_scored.jsonl"):
            for field, atol in (("probe_base", .0005), ("probe_logit_base", .002)):
                require(math.isclose(row[field], reference[row["key"]][field], rel_tol=.0001, abs_tol=atol), "Fixed-token adapter-disabled base reader changed")
    return result


def stages(configs, root):
    result = []
    for name in ("initial", *RUN_ORDER):
        cfg = configs[RUN_ORDER[0]] if name == "initial" else configs[name]
        run, evaluation = Path(cfg["run_dir"]), ARTIFACT_ROOT / f"eval_{name}"
        if name != "initial":
            result.extend([
                {"name": name, "kind": "train", "marker": run / "complete.json", "targets": [run], "log": root / f"train_{name}.log"},
                {"name": name, "kind": "audit", "marker": run / "audit_complete.json",
                 "targets": [run / f for f in ("audit_complete.json", "execution_audit.jsonl", "refresh_execution_audit.jsonl", "refresh_quality.json")],
                 "log": root / f"audit_{name}.log"}])
        result.extend([
            {"name": name, "kind": "eval", "marker": evaluation / "eval.json", "targets": [evaluation], "log": root / f"eval_{name}.log"},
            {"name": name, "kind": "judge", "marker": evaluation / "monitors/judge_complete.json", "targets": [evaluation / "monitors"],
             "log": root / f"judge_{name}.log"}])
    return result


def verify_stage(stage, configs, manifest, root):
    name, kind = stage["name"], stage["kind"]
    cfg = configs[RUN_ORDER[0]] if name == "initial" else configs[name]
    if kind == "train":
        return verify_training(cfg)
    if kind == "audit":
        return verify_execution_audit(cfg)
    if kind == "eval":
        return verify_evaluation(name, cfg, manifest)
    return JUDGE.verify(ARTIFACT_ROOT / f"eval_{name}", root / "manifest.json")


def inspect_stages(configs, manifest, root, *, resume=False, initial_only=False):
    require(set(configs) == set(RUN_ORDER), "Need exactly the six declared run configurations")
    complete, pending, gap = [], [], False
    all_stages = stages(configs, Path(root))
    for stage in all_stages:
        exists = any(os.path.lexists(path) for path in (*stage["targets"], stage["log"]))
        if stage["marker"].is_file():
            require(not gap, "Existing completed stage follows an unfinished predecessor")
            verify_stage(stage, configs, manifest, Path(root))
            complete.append(stage)
        else:
            require(not exists, f"Partial {stage['kind']} for {stage['name']}; preserve artifacts, never overwrite/retry")
            gap = True
            pending.append(stage)
    require(resume or not complete, "A completed prefix exists; use explicit --resume to continue without retraining")
    if initial_only:
        require(all(stage["name"] == "initial" for stage in complete), "--initial-only cannot follow training")
        pending = [stage for stage in pending if stage["name"] == "initial"]
    return complete, pending


def command(stage, configs, manifest, root):
    name, kind = stage["name"], stage["kind"]
    cfg = configs[RUN_ORDER[0]] if name == "initial" else configs[name]
    if kind == "train":
        return [sys.executable, "-B", str(REPO / "scripts/train_judge_replay.py"), "--config", str(root / f"config_{name}.json")]
    if kind == "audit":
        return [sys.executable, "-B", str(REPO / "scripts/audit_judge_replay.py"), "--run-dir", cfg["run_dir"], "--config", str(root / f"config_{name}.json")]
    if kind == "judge":
        return [sys.executable, "-B", str(REPO / "scripts/eval_judge_replay.py"), "--dir", str(ARTIFACT_ROOT / f"eval_{name}"), "--manifest", str(root / "manifest.json")]
    adapter = cfg["init_adapter"] if name == "initial" else str(Path(cfg["run_dir"]) / "serving_adapter")
    return COMMON.evaluation_command(adapter, ARTIFACT_ROOT / f"eval_{name}", cfg, manifest["evaluation_lora_ids"][name])


def main(root=ROOT, *, execute=False, resume=False, initial_only=False):
    prep, root = preparer(), Path(root).resolve(strict=True)
    prep.check_paths(root)
    require(tuple(prep.RUN_ORDER) == RUN_ORDER and Path(prep.ARTIFACT_ROOT) == ARTIFACT_ROOT and prep.PROTOCOL == PROTOCOL, "Preparer/coordinator protocol differs")
    manifest_path = root / "manifest.json"
    manifest_hash, manifest = prep.digest(manifest_path), prep.read_json(manifest_path)
    configs = prep.validate_manifest(manifest, root, check_server=False)
    done, pending = inspect_stages(configs, manifest, root, resume=resume, initial_only=initial_only)
    if not execute:
        report = {"status": "read_only_preflight", "n_completed_stages": len(done), "pending_stages": [[s["name"], s["kind"]] for s in pending],
                  "server_or_gpu_queried": False, **prep.storage_guard(root, headroom=prep.WRITE_HEADROOM)}
        print(json.dumps(report), flush=True)
        return report
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Launch coordinator explicitly with CUDA_VISIBLE_DEVICES=0")
    require(not os.environ.get("TESTBED_SYSTEM_SUFFIX"), "Ambient prompt suffix is forbidden")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME="/ssd1/john/.cache/huggingface",
               TOKENIZERS_PARALLELISM="false", TESTBED_SYSTEM_SUFFIX="", TESTBED_STORE=str(prep.STORE),
               PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    with leases(root) as lease_fds:
        require(prep.digest(manifest_path) == manifest_hash, "Manifest changed before leases")
        done, pending = inspect_stages(configs, manifest, root, resume=resume, initial_only=initial_only)
        status = root / "coordinator_events.jsonl"
        if status.exists():
            require(resume and all(r.get("manifest_sha256") == manifest_hash for r in COMMON.rows(status)), "Existing coordinator log needs matching explicit resume")
        with status.open("a" if resume else "x") as stream:
            def emit(event):
                value = {"utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest_hash, **event}
                stream.write(json.dumps(value, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps(value, allow_nan=False), flush=True)
            try:
                emit({"event": "coordinator_started", "resume": resume, "completed_prefix": len(done), "pending_stages": len(pending)})
                for stage in pending:
                    current, evidence = prelaunch(manifest, root)
                    require(current == configs and prep.digest(manifest_path) == manifest_hash, "Frozen configurations changed before launch")
                    require(not any(os.path.lexists(p) for p in (*stage["targets"], stage["log"])), "Stage appeared before launch; do not overwrite")
                    emit(evidence)
                    stage_env = dict(env, CUDA_VISIBLE_DEVICES="0" if stage["kind"] in ("train", "eval") else "")
                    child(command(stage, configs, manifest, root), stage["log"], stage_env, lease_fds, emit)
                    verify_stage(stage, configs, manifest, root)
                    emit({"event": "stage_complete", "name": stage["name"], "kind": stage["kind"], **prep.storage_guard(root)})
                emit({"event": "requested_work_complete", "initial_only": initial_only, "completed_new_stages": len(pending)})
            except BaseException as exc:
                emit({"event": "stopped_preserve_all", "error_type": type(exc).__name__, "reason": str(exc), "automatic_retry": False})
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", action="store_true", help="Authorize the declared serial GPU0/existing-server workflow")
    parser.add_argument("--resume", action="store_true", help="Skip only a fully verified completed prefix; never retry partial stages")
    parser.add_argument("--initial-only", action="store_true", help="Only the initialization probe and judge evaluations")
    args = parser.parse_args()
    main(args.root, execute=args.run, resume=args.resume, initial_only=args.initial_only)
