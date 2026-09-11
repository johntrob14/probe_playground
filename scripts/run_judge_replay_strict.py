"""Isolated two-seed strict-replay coordinator; frozen study files are unchanged.

Reuse old first batches and the old initialization evaluation. Explicit --run
authorizes eight serial stages; --resume skips only a verified complete prefix.
Audit/judge adapters change process-local paths/guards, never the monitor rubric.
No server restart, GPU2/4 use, partial overwrite, retraining, or implicit retry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/judge_labelled_replay_strict_20260908"
ARTIFACT_ROOT = Path("/ssd1/john/probe_playground/runs/judge_labelled_replay_strict_20260908")
RUN_ORDER = ("prob_judge_replay_strict_s613", "prob_judge_replay_strict_s719")
sys.path.insert(0, str(REPO / "scripts"))
import run_judge_replay as OLD
import prepare_judge_replay as OLD_PREP
import eval_judge_replay as JUDGE
import audit_judge_replay as AUDIT


def prep_module():
    import prepare_judge_replay_strict
    return prepare_judge_replay_strict


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def adapted(prep):
    """An explicit, reversible import adapter in this process only.

    OLD ROOT remains the canonical evaluation parser/diagnostic helper location;
    only the allowed artifact tree, run inventory, and provenance guard change.
    """
    proxy = ModuleType("prepare_judge_replay")
    proxy.__dict__.update(vars(OLD_PREP))
    proxy.ARTIFACT_ROOT, proxy.RUN_ORDER, proxy.PROTOCOL = ARTIFACT_ROOT, RUN_ORDER, prep.PROTOCOL
    def validate(manifest, root=None, check_server=False):
        require(root is None or Path(root).resolve() == ROOT.resolve(), "Wrong strict manifest directory")
        return prep.validate_manifest(manifest, check_server=check_server)
    proxy.validate_manifest = validate
    proxy.storage_guard = lambda *args, headroom=0, **kwargs: prep.storage_guard(headroom=headroom)
    previous, old_audit_prep, old_protocol = sys.modules.get("prepare_judge_replay"), AUDIT.prep, JUDGE.PROTOCOL
    sys.modules["prepare_judge_replay"] = proxy
    AUDIT.prep, JUDGE.PROTOCOL = proxy, prep.PROTOCOL
    try:
        yield
    finally:
        AUDIT.prep, JUDGE.PROTOCOL = old_audit_prep, old_protocol
        if previous is None:
            sys.modules.pop("prepare_judge_replay", None)
        else:
            sys.modules["prepare_judge_replay"] = previous


def stage_list(configs):
    require(set(configs) == set(RUN_ORDER), "Need exactly the two strict seeds")
    result = []
    for name in RUN_ORDER:
        run, evaluation = Path(configs[name]["run_dir"]), ARTIFACT_ROOT / f"eval_{name}"
        require(run == ARTIFACT_ROOT / name, "Run escaped the declared strict artifact tree")
        for kind, marker, targets in (
            ("train", run / "complete.json", [run]),
            ("audit", run / "audit_complete.json", [run / p for p in ("audit_complete.json", "execution_audit.jsonl", "refresh_execution_audit.jsonl", "refresh_quality.json")]),
            ("eval", evaluation / "eval.json", [evaluation]),
            ("judge", evaluation / "monitors/judge_complete.json", [evaluation / "monitors"]),
        ):
            result.append({"name": name, "kind": kind, "marker": marker, "targets": targets, "log": ROOT / f"{kind}_{name}.log"})
    return result


def verify_stage(stage, configs, manifest, prep):
    name, kind = stage["name"], stage["kind"]
    cfg, evaluation = configs[name], ARTIFACT_ROOT / f"eval_{name}"
    if kind == "train":
        return OLD.verify_training(cfg)
    if kind == "audit":
        return OLD.verify_execution_audit(cfg)
    if kind == "judge":
        with adapted(prep):
            return JUDGE.verify(evaluation, ROOT / "manifest.json")
    adapter = str(Path(cfg["run_dir"]) / "serving_adapter")
    result = OLD.COMMON.verify_evaluation(evaluation, cfg, adapter, manifest["evaluation_lora_ids"][name])
    initial = OLD_PREP.ARTIFACT_ROOT / "eval_initial/audit_scored.jsonl"
    reference = {row["key"]: row for row in OLD.COMMON.rows(initial)}
    for row in OLD.COMMON.rows(evaluation / "audit_scored.jsonl"):
        for field, tolerance in (("probe_base", .0005), ("probe_logit_base", .002)):
            require(math.isclose(row[field], reference[row["key"]][field], abs_tol=tolerance, rel_tol=.0001),
                    "Fixed-token adapter-disabled readout changed from the reused initialization")
    return result


def inspect(configs, manifest, prep, *, resume=False):
    complete, remaining, gap = [], [], False
    for stage in stage_list(configs):
        if stage["marker"].is_file():
            require(not gap, "Completed stage follows an unfinished predecessor")
            verify_stage(stage, configs, manifest, prep)
            complete.append(stage)
        else:
            require(not any(os.path.lexists(p) for p in (*stage["targets"], stage["log"])),
                    f"Partial {stage['kind']} for {stage['name']}; preserve all files and do not retry")
            gap = True
            remaining.append(stage)
    require(resume or not complete, "Existing completed prefix requires explicit --resume")
    return complete, remaining


def command(stage, configs, manifest):
    cfg, kind, name = configs[stage["name"]], stage["kind"], stage["name"]
    config_path = ROOT / f"config_{name}.json"
    if kind == "train":
        return [sys.executable, "-B", str(REPO / "scripts/train_judge_replay_strict.py"), "--config", str(config_path)]
    if kind in ("audit", "judge"):
        return [sys.executable, "-B", str(Path(__file__).resolve()), "--stage", kind, "--config", str(config_path)]
    return OLD.COMMON.evaluation_command(str(Path(cfg["run_dir"]) / "serving_adapter"), ARTIFACT_ROOT / f"eval_{name}", cfg,
                                        manifest["evaluation_lora_ids"][name])


def internal_stage(kind, config_path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Audit/judge adapters are CPU-only")
    prep = prep_module()
    configs = prep.validate_manifest(check_server=False)
    cfg = OLD_PREP.read_json(config_path)
    name = Path(cfg["run_dir"]).name
    require(name in configs and cfg == configs[name] and Path(config_path).resolve() == ROOT / f"config_{name}.json", "Undeclared adapter configuration")
    # The coordinator already checks GPU0 and both servers before every child.
    # Do not alter frozen modules on disk or replace the evaluation judge rubric.
    with adapted(prep):
        if kind == "audit":
            AUDIT.audit(cfg["run_dir"], config_path)
        else:
            JUDGE.main(ARTIFACT_ROOT / f"eval_{name}", ROOT / "manifest.json")


def main(*, execute=False, resume=False):
    prep = prep_module()
    require(Path(prep.ROOT) == ROOT and Path(prep.ARTIFACT_ROOT) == ARTIFACT_ROOT, "Strict preparer paths differ")
    manifest_path = ROOT / "manifest.json"
    manifest_hash = OLD_PREP.digest(manifest_path)
    manifest = OLD_PREP.read_json(manifest_path)
    configs = prep.validate_manifest(manifest, check_server=False)
    complete, remaining = inspect(configs, manifest, prep, resume=resume)
    if not execute:
        result = {"status": "read_only_preflight", "completed_stages": len(complete),
                  "remaining_stages": [[s["name"], s["kind"]] for s in remaining], "reuses_old_initial_evaluation": True,
                  "server_or_gpu_queried": False, **prep.storage_guard(headroom=3 * 1024**3)}
        print(json.dumps(result), flush=True)
        return result
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and not os.environ.get("TESTBED_SYSTEM_SUFFIX"),
            "Require explicit GPU0 coordinator and an empty system suffix")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME="/ssd1/john/.cache/huggingface",
               TOKENIZERS_PARALLELISM="false", TESTBED_SYSTEM_SUFFIX="", TESTBED_STORE=str(OLD_PREP.STORE),
               PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    with OLD.leases(ROOT) as lease_fds:
        require(OLD_PREP.digest(manifest_path) == manifest_hash, "Strict manifest changed before lease acquisition")
        complete, remaining = inspect(configs, manifest, prep, resume=resume)
        status = ROOT / "coordinator_events.jsonl"
        if status.exists():
            require(resume and all(r.get("manifest_sha256") == manifest_hash for r in OLD.COMMON.rows(status)), "Existing event log needs matching explicit resume")
        with status.open("a" if resume else "x") as stream:
            def emit(event):
                record = {"utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest_hash, **event}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
                print(json.dumps(record, allow_nan=False), flush=True)
            try:
                for stage in remaining:
                    require(OLD_PREP.digest(manifest_path) == manifest_hash, "Strict manifest changed during workflow")
                    require(prep.validate_manifest(manifest, check_server=False) == configs, "Strict configuration/source changed")
                    servers = prep.pinned_servers({"rollout": manifest["rollout_server"], "judge": manifest["judge_server"]})
                    resources = prep.storage_guard(headroom=3 * 1024**3)
                    idle = OLD.COMMON.gpu0_idle(max_wait=30)
                    require(not any(os.path.lexists(p) for p in (*stage["targets"], stage["log"])), "Stage appeared before launch; preserve it")
                    emit({"event": "prelaunch_verified", "name": stage["name"], "kind": stage["kind"], "servers": servers, "gpu0_idle": idle, **resources})
                    child_env = dict(env, CUDA_VISIBLE_DEVICES="0" if stage["kind"] in ("train", "eval") else "")
                    OLD.child(command(stage, configs, manifest), stage["log"], child_env, lease_fds, emit)
                    verify_stage(stage, configs, manifest, prep)
                    emit({"event": "stage_complete", "name": stage["name"], "kind": stage["kind"], **prep.storage_guard()})
                emit({"event": "requested_work_complete", "completed_new_stages": len(remaining)})
            except BaseException as exc:
                emit({"event": "stopped_preserve_all", "error_type": type(exc).__name__, "reason": str(exc), "automatic_retry": False})
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("audit", "judge"), help="Explicit process-local CPU adapter, used by coordinator")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.stage:
        require(args.config is not None and not args.run and not args.resume, "Internal stage requires only --stage and --config")
        internal_stage(args.stage, args.config)
    else:
        require(args.config is None, "--config is only for an internal adapter")
        main(execute=args.run, resume=args.resume)
