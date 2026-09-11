"""Prepare new-only configurations for the declared six-run replication grid."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import urllib.request

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "experiments/replay_pilot_20260904"
ROOT = REPO / "experiments/rank_replication_20260905"
SERVER_PID = 2227988
STORE = Path("/ssd1/john/probe_playground")
ARMS = (
    ("rank_replay_s31", 31, "within_prompt_average_logit_rank", .05, "train_rank_replay_pilot.py"),
    ("rank_only_s31", 31, "within_prompt_average_logit_rank", 0., "train_rank_replay_pilot.py"),
    ("probability_replay_s31", 31, "probability", .05, "train_replay_pilot.py"),
    ("matched_probability_replay_s31", 31, "within_prompt_variance_matched_probability", .05, "train_monitor_control_pilot.py"),
    ("rank_only_s17", 17, "within_prompt_average_logit_rank", 0., "train_rank_replay_pilot.py"),
    ("rank_replay_s47", 47, "within_prompt_average_logit_rank", .05, "train_rank_replay_pilot.py"),
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def expected_config(base, root, arm, ports=(8005,)):
    """The entire declared configuration, not just its treatment label."""
    name, seed, transform, weight, trainer = arm
    return dict(base, experiment_root=str(root), ports=list(ports), storage_budget_bytes=20 * 1024**3,
                replication_protocol="rank_replication_20260905", rollout_max_model_len=4096,
                system_suffix="", evaluation_seed=1234, dataset_store=str(STORE),
                run_dir=str(root / name), seed=seed, replay_weight=weight,
                penalty_transform=transform, trainer=trainer,
                audit_bank=str(OLD / "audit_bank.jsonl"),
                fresh_test_task_ids=str(OLD / "fresh_test_task_ids.json"))


def software_versions():
    result = {}
    for name in ("numpy", "torch", "transformers", "peft", "vllm", "flash-attn", "pandas", "scikit-learn"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def server_evidence(pid=SERVER_PID):
    """Read only the coordinated process identity and local HTTP readiness.

    This is process/health evidence, not an attestation of loaded model weights.
    It neither starts generation nor changes or stops the existing service.
    """
    process = Path(f"/proc/{pid}")
    raw = (process / "cmdline").read_bytes()
    argv = [item.decode() for item in raw.split(b"\0") if item]
    def option(flag):
        if argv.count(flag) != 1 or argv.index(flag) + 1 == len(argv):
            raise ValueError(f"Server lacks unambiguous {flag}")
        return argv[argv.index(flag) + 1]
    environment = (process / "environ").read_bytes().split(b"\0")
    cuda = [item.split(b"=", 1)[1].decode() for item in environment
            if item.startswith(b"CUDA_VISIBLE_DEVICES=")]
    start_ticks = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
    if (option("-m") != "testbed.rollout_server" or option("--model") != "Qwen/Qwen3-8B"
            or option("--port") != "8005" or option("--max-model-len") != "4096"
            or "--enable-lora" not in argv or cuda != ["5"]):
        raise ValueError("Server identity differs from coordinated GPU5/Qwen/port8005/context4096")
    url = "http://127.0.0.1:8005/health"
    with urllib.request.urlopen(url, timeout=5) as response:
        status, body = response.status, response.read(64).decode()
    if status != 200 or body.strip() != "ok":
        raise ValueError("Coordinated rollout service is not ready")
    if (process / "cmdline").read_bytes() != raw or (
            process / "stat").read_text().rsplit(")", 1)[1].split()[19] != start_ticks:
        raise ValueError("Server process changed during readiness check")
    return {"pid": pid, "starttime_ticks": start_ticks, "argv": argv,
            "cmdline_sha256": hashlib.sha256(raw).hexdigest(), "cuda_visible_devices": "5",
            "max_model_len": 4096, "health_url": url, "health_status": status,
            "health_body": body, "checked_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Observed process arguments/environment and HTTP readiness, not loaded-weight attestation"}


def prepare(root=ROOT, ports=(8005,)):
    root = root.resolve()
    if root != ROOT.resolve() or not (root / "PLAN.md").is_file():
        raise ValueError("Require the declared experiment root and its pre-result plan")
    if tuple(ports) != (8005,):
        raise ValueError("Only the coordinated rollout port 8005 is authorized")
    destinations = [root / "manifest.json"] + [root / f"config_{a[0]}.json" for a in ARMS]
    if any(p.exists() for p in destinations) or any((root / a[0]).exists() for a in ARMS):
        raise FileExistsError("Preparation is new-only; prior configuration/results must be preserved")
    if shutil.disk_usage(root).free < 50 * 1024**3:
        raise RuntimeError("Require 50 GiB free before preparing the bounded 20 GiB experiment")
    cfg = json.loads((OLD / "config_rank_combined.json").read_text())
    ids = {name: json.loads((OLD / f"{name}.json").read_text())
           for name in ("train_task_ids", "fresh_test_task_ids", "dev_task_ids")}
    if len(ids["train_task_ids"]) != 300 or len(ids["fresh_test_task_ids"]) != 120:
        raise ValueError("Unexpected original task grid")
    if any(len(v) != len(set(v)) for v in ids.values()):
        raise ValueError("Duplicate task IDs")
    if (set(ids["train_task_ids"]) & (set(ids["fresh_test_task_ids"]) | set(ids["dev_task_ids"]))
            or set(ids["fresh_test_task_ids"]) & set(ids["dev_task_ids"])):
        raise ValueError("Task split overlap")
    server = server_evidence()
    sources = [OLD / "config_rank_combined.json", OLD / "replay_bank.jsonl", OLD / "audit_bank.jsonl",
               OLD / "train_task_ids.json", OLD / "fresh_test_task_ids.json", OLD / "dev_task_ids.json",
               Path(cfg["probe_file"]), Path(cfg["init_adapter"]) / "adapter_model.safetensors",
               Path(cfg["init_adapter"]) / "adapter_config.json",
               STORE / "data/mbpp-hardcode/data/train-00000-of-00001.parquet",
               STORE / "data/mbpp-hardcode/data/test-00000-of-00001.parquet",
               root / "PLAN.md"]
    sources += [REPO / "scripts" / name for name in (
        "train_rank_replay_pilot.py", "train_replay_pilot.py", "train_monitor_control_pilot.py",
        "eval_replay_checkpoint.py", "eval_replay_replication.py",
        "prepare_rank_replication.py", "run_rank_replication.py")]
    # Include indirect imports (probe normalization, pooling, environment, prompt,
    # rollout client/server, and policy) without importing model/GPU libraries.
    sources += sorted((REPO / "src/testbed").rglob("*.py"))
    hashes = {str(p.resolve()): digest(p) for p in sources}
    arms = []
    config_hashes = {}
    for arm in ARMS:
        name, seed, transform, weight, trainer = arm
        value = expected_config(cfg, root, arm, ports)
        write_new(root / f"config_{name}.json", value)
        config_hashes[str(root / f"config_{name}.json")] = digest(root / f"config_{name}.json")
        arms.append({"name": name, "seed": seed, "transform": transform, "replay_weight": weight,
                     "trainer": trainer, "config": str(root / f"config_{name}.json")})
    manifest = {"prepared_at": datetime.now(timezone.utc).isoformat(), "status": "prepared_not_results",
                "arms": arms, "source_sha256": hashes, "config_sha256": config_hashes,
                "software_versions": software_versions(), "rollout_server": server,
                "storage_cap_bytes": 20 * 1024**3,
                "free_space_reserve_bytes": 30 * 1024**3, "train_gpu": 0, "rollout_gpu": 5,
                "rollout_ports": list(ports), "rollout_max_model_len": 4096,
                "reserved_gpus_untouched": [2, 4], "evaluation_n_tasks": 120, "evaluation_n_per_task": 4,
                "primary_seed": 31, "training_steps": 20,
                "notes": ["Existing banks are referenced, not copied; all original artifacts are preserved.",
                          "Seed31 four-way comparison; seed17 rank-only uses a historical rank+replay comparator; seed47 only replicates rank+replay.",
                          "A future near-onset branch requires a separate pre-result protocol and must fit the remaining storage bound."]}
    write_new(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    prepare(args.out)
