"""Four output-view replay runs using an isolated frozen-coordinator adapter.

No frozen file is edited. Only roots, the declared run inventory, the preparer,
and child command paths differ from the strict-follow-up coordinator. Its
completed-prefix, no-retry, inherited leases, GPU0, storage, source-integrity,
fixed-base, and historical-judge checks remain unchanged. Default is offline.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/judge_labelled_replay_output_20260908"
ARTIFACT_ROOT = Path("/ssd1/john/probe_playground/runs/judge_labelled_replay_output_20260908")
RUN_ORDER = ("prob_judge_output_cot_answer_s613", "prob_judge_output_cot_only_s613",
             "prob_judge_output_cot_only_s719", "prob_judge_output_cot_answer_s719")
sys.path.insert(0, str(REPO / "scripts"))


def prep_module():
    import prepare_judge_replay_output
    return prepare_judge_replay_output


def load_runner():
    """Fresh module namespace: changing these globals cannot change old runners."""
    source = REPO / "scripts/run_judge_replay_strict.py"
    spec = importlib.util.spec_from_file_location("_judge_output_frozen_coordinator", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT, runner.ARTIFACT_ROOT, runner.RUN_ORDER = ROOT, ARTIFACT_ROOT, RUN_ORDER
    runner.prep_module = prep_module
    original_stages, original_verify = runner.stage_list, runner.verify_stage

    def stage_list(configs):
        stages = []
        for stage in original_stages(configs):
            stages.append(stage)
            if stage["kind"] == "judge":
                name = stage["name"]
                output = runner.ARTIFACT_ROOT / f"eval_{name}" / "output_monitors"
                stages.append({"name": name, "kind": "output_judge", "marker": output / "complete.json",
                               "targets": [output], "log": runner.ROOT / f"output_judge_{name}.log"})
        return stages

    def verify_stage(stage, configs, manifest, prep):
        if stage["kind"] == "output_judge":
            import eval_judge_replay_output
            return eval_judge_replay_output.verify(runner.ARTIFACT_ROOT / f"eval_{stage['name']}", runner.ROOT / "manifest.json")
        return original_verify(stage, configs, manifest, prep)

    def command(stage, configs, manifest):
        name, kind = stage["name"], stage["kind"]
        cfg = configs[name]
        config = runner.ROOT / f"config_{name}.json"
        if kind == "train":
            return [sys.executable, "-B", str(REPO / "scripts/train_judge_replay_output.py"), "--config", str(config)]
        if kind in ("audit", "judge"):
            return [sys.executable, "-B", str(Path(__file__).resolve()), "--stage", kind, "--config", str(config)]
        if kind == "output_judge":
            return [sys.executable, "-B", str(REPO / "scripts/eval_judge_replay_output.py"), "--config", str(config)]
        runner.require(kind == "eval", "Undeclared output-study stage")
        return runner.OLD.COMMON.evaluation_command(str(Path(cfg["run_dir"]) / "serving_adapter"),
            runner.ARTIFACT_ROOT / f"eval_{name}", cfg, manifest["evaluation_lora_ids"][name])

    runner.command, runner.stage_list, runner.verify_stage = command, stage_list, verify_stage
    return runner


RUNNER = load_runner()
# Explicit API for independent offline aggregation/verification.
stage_list, verify_stage, inspect = RUNNER.stage_list, RUNNER.verify_stage, RUNNER.inspect
command, internal_stage, main = RUNNER.command, RUNNER.internal_stage, RUNNER.main
adapted, require = RUNNER.adapted, RUNNER.require


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("audit", "judge"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.stage:
        require(args.config is not None and not args.run and not args.resume,
                "Internal stage requires only --stage and --config")
        internal_stage(args.stage, args.config)
    else:
        require(args.config is None, "--config is only for an internal adapter")
        main(execute=args.run, resume=args.resume)
