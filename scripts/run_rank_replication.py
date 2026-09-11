"""Serial, new-only training and compact evaluation on the coordinated GPU 0.

Does not launch or stop rollout servers. Does not edit the old shared queue.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from prepare_rank_replication import (ARMS, OLD, REPO, ROOT, STORE, expected_config,
                                      server_evidence, software_versions)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def storage_guard(root, reserve_bytes=0):
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    free = shutil.disk_usage(root).free
    if used + reserve_bytes >= 20 * 1024**3 or free - reserve_bytes < 30 * 1024**3:
        raise RuntimeError(f"Storage guard reached: used={used}, free={free}; all results preserved")
    return {"artifact_bytes": used, "filesystem_free_bytes": free, "reserved_next_bytes": reserve_bytes}


def child(command, logfile, env, lock_fd):
    with logfile.open("x") as handle:
        # An active child retains the lease even if its coordinator is interrupted.
        # A new coordinator cannot overlap an orphaned owned trainer/evaluator.
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   pass_fds=(lock_fd,))
        print(json.dumps({"event": "started", "pid": process.pid, "log": str(logfile),
                          "command": command, "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Child exited {code}; inspect {logfile}; no automatic restart or overwrite")


def verify_manifest(root, manifest):
    """Reject missing hashes, changed configurations, and mislabeled treatments."""
    base = json.loads((OLD / "config_rank_combined.json").read_text())
    expected_paths = {str(root / f"config_{arm[0]}.json") for arm in ARMS}
    if set(manifest.get("config_sha256", {})) != expected_paths:
        raise ValueError("Manifest does not lock every declared arm configuration")
    required_sources = {str((REPO / "scripts" / name).resolve()) for name in (
        "train_rank_replay_pilot.py", "train_replay_pilot.py", "train_monitor_control_pilot.py",
        "eval_replay_checkpoint.py", "eval_replay_replication.py",
        "prepare_rank_replication.py", "run_rank_replication.py")}
    required_sources.update(str(p.resolve()) for p in (REPO / "src/testbed").rglob("*.py"))
    required_sources.update(str(p.resolve()) for p in (
        OLD / "config_rank_combined.json", OLD / "replay_bank.jsonl", OLD / "audit_bank.jsonl",
        OLD / "train_task_ids.json", OLD / "fresh_test_task_ids.json", OLD / "dev_task_ids.json",
        Path(base["probe_file"]), Path(base["init_adapter"]) / "adapter_model.safetensors",
        Path(base["init_adapter"]) / "adapter_config.json", root / "PLAN.md",
        STORE / "data/mbpp-hardcode/data/train-00000-of-00001.parquet",
        STORE / "data/mbpp-hardcode/data/test-00000-of-00001.parquet"))
    if not required_sources <= manifest.get("source_sha256", {}).keys():
        raise ValueError("Manifest omits an implementation/protocol dependency")
    for mapping in (manifest["source_sha256"], manifest["config_sha256"]):
        for path, expected in mapping.items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Declared source/configuration changed: {path}")
    expected_arms = [{"name": name, "seed": seed, "transform": transform, "replay_weight": weight,
                      "trainer": trainer, "config": str(root / f"config_{name}.json")}
                     for name, seed, transform, weight, trainer in ARMS]
    if manifest.get("arms") != expected_arms:
        raise ValueError("Manifest arm labels differ from the declared grid")
    for arm in ARMS:
        actual = json.loads((root / f"config_{arm[0]}.json").read_text())
        if actual != expected_config(base, root, arm):
            raise ValueError(f"Configuration differs from exact declared treatment/base settings: {arm[0]}")
    if manifest.get("software_versions") != software_versions():
        raise ValueError("Declared Python package versions changed")
    if (manifest.get("train_gpu"), manifest.get("rollout_gpu"), manifest.get("rollout_ports"),
            manifest.get("rollout_max_model_len")) != (0, 5, [8005], 4096):
        raise ValueError("Manifest device/server protocol mismatch")


def verify_server(manifest):
    locked = manifest["rollout_server"]
    observed = server_evidence(locked["pid"])
    for field in ("pid", "starttime_ticks", "cmdline_sha256", "cuda_visible_devices", "max_model_len"):
        if observed[field] != locked[field]:
            raise ValueError(f"Coordinated server identity changed: {field}")
    return observed


def main(root, names):
    root = root.resolve(strict=True)
    if root != ROOT.resolve() or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Require the declared root and explicitly coordinated CUDA_VISIBLE_DEVICES=0")
    all_names = {a[0] for a in ARMS}
    if not names or len(names) != len(set(names)) or set(names) - all_names:
        raise ValueError("Require unique declared arm names")
    manifest = json.loads((root / "manifest.json").read_text())
    verify_manifest(root, manifest)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               TESTBED_SYSTEM_SUFFIX="", TESTBED_STORE=str(STORE), PYTHONPATH=str(REPO / "src"))
    started = time.monotonic()
    # Prevent two new coordinators from overlapping on the one reserved trainer device.
    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in names:
            verify_manifest(root, manifest)
            server = verify_server(manifest)
            cfg_path = root / f"config_{name}.json"
            cfg = json.loads(cfg_path.read_text())
            run = Path(cfg["run_dir"]).resolve()
            if run != root / name or cfg["ports"] != [8005] or cfg["steps"] != 20:
                raise ValueError("Unexpected run target/protocol")
            if run.exists() or (root / f"eval_{name}").exists():
                raise FileExistsError(f"Existing run/evaluation {name}; inspect rather than restart")
            trainer = REPO / "scripts" / cfg["trainer"]
            evaluator = REPO / "scripts/eval_replay_replication.py"
            if not trainer.is_file() or not evaluator.is_file():
                raise FileNotFoundError("Trainer/evaluator implementation not ready")
            # Completed historical runs use ~2.6 GiB; reserve 3 GiB before starting
            # the unchanged trainer, whose final optimizer is written in one go.
            print(json.dumps({"event": "preflight", "arm": name, **storage_guard(root, 3 * 1024**3),
                              "rollout_server": server,
                              "trainer_sha256": digest(trainer), "evaluator_sha256": digest(evaluator),
                              "config_sha256": digest(cfg_path)}), flush=True)
            child([sys.executable, str(trainer), "--config", str(cfg_path)], root / f"train_{name}.log", env,
                  lock.fileno())
            complete = json.loads((run / "complete.json").read_text())
            if complete["steps"] != 20 or Path(complete["adapter"]).resolve() != run / "serving_adapter":
                raise RuntimeError("Training returned without expected final completion evidence")
            if not (run / "serving_adapter/adapter_model.safetensors").is_file():
                raise RuntimeError("Completed final adapter missing")
            if json.loads((run / "config.json").read_text()) != cfg:
                raise RuntimeError("Saved training configuration differs from locked configuration")
            steps = [json.loads(line)["step"] for line in (run / "metrics.jsonl").read_text().splitlines()]
            if steps != list(range(1, cfg["steps"] + 1)):
                raise RuntimeError("Completed training metrics do not contain exactly the declared updates")
            storage_guard(root, 100 * 1024**2)
            verify_server(manifest)
            child([sys.executable, str(evaluator), "--adapter", complete["adapter"],
                   "--out", str(root / f"eval_{name}"), "--audit-bank", cfg["audit_bank"],
                   "--probe-file", cfg["probe_file"], "--task-ids-file", cfg["fresh_test_task_ids"],
                   "--exclude-task-ids-file", cfg["train_task_ids"], "--ports", "8005", "--split", "test",
                   "--n", "4", "--max-new-tokens", "2048", "--layer", "15", "--pool", "mean_cot",
                   "--acts-from", "policy,base", "--nonterm-penalty", "0", "--seed", "1234"],
                  root / f"eval_{name}.log", env, lock.fileno())
            summary = json.loads((root / f"eval_{name}/eval.json").read_text())
            if summary["n_rollouts"] != 480 or summary["n_tasks"] != 120:
                raise RuntimeError("Unexpected completed evaluation grid")
            print(json.dumps({"event": "arm_completed", "arm": name,
                              "elapsed_seconds": time.monotonic() - started, **storage_guard(root)}), flush=True)
        print(json.dumps({"event": "selected_arms_complete", "arms": names}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    main(args.root, args.arms)
