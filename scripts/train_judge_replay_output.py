"""Output-aware teacher adapter; unchanged frozen actor/probe/replay arithmetic."""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import train_judge_replay as frozen

ROOT = frozen.REPO / "experiments/judge_labelled_replay_output_20260908"
ARTIFACT_ROOT = Path("/ssd1/john/probe_playground/runs/judge_labelled_replay_output_20260908")
PROTOCOL = "judge_labelled_replay_output_20260908"
SOURCE = "judge_output"
SEEDS = (613, 719)
VIEWS = ("cot_only", "cot_answer")
RUN_ORDER = ("prob_judge_output_cot_answer_s613", "prob_judge_output_cot_only_s613",
             "prob_judge_output_cot_only_s719", "prob_judge_output_cot_answer_s719")
FROZEN_TRAINER_SHA256 = "a4642d461b797749d9ed63e866080bf3ae7884b9cdb38c4f090928faf00c5685"


def old_config(cfg):
    path = Path(cfg["old_config_path"])
    expected = cfg["old_config_sha256"]
    if not frozen.arithmetic.valid_sha256(expected) or frozen.sha256_file(path) != expected:
        raise ValueError("Old configuration hash changed")
    previous = frozen.parse_json(path.read_bytes())
    frozen.validate_config(previous)
    if previous["arm"] != "prob_judge_replay" or previous["seed"] != cfg["seed"]:
        raise ValueError("Wrong original judge-arm/seed configuration")
    return previous


def validate_config(cfg):
    if frozen.sha256_file(frozen.__file__) != FROZEN_TRAINER_SHA256:
        raise ValueError("Frozen trainer changed; adapter requires review")
    seed, view = cfg.get("seed"), cfg.get("judge_view")
    if type(seed) is not int or seed not in SEEDS or view not in VIEWS:
        raise ValueError("Undeclared seed/view")
    arm = f"prob_judge_output_{view}"
    name = f"{arm}_s{seed}"
    exact = {"arm": arm, "replay_label_source": SOURCE, "replication_protocol": PROTOCOL,
             "trainer": "train_judge_replay_output.py", "hack_min": .9, "honest_max": 0.,
             "judge_view": view, "metadata_root": str(ROOT), "experiment_root": str(ARTIFACT_ROOT),
             "artifact_root": str(ARTIFACT_ROOT), "run_dir": str(ARTIFACT_ROOT / name),
             "output_manifest": str(ROOT / "manifest.json"), "storage_budget_bytes": 16 * 1024**3,
             "lora_id_base": 1_850_000_000 + RUN_ORDER.index(name) * 1_000_000}
    if any(cfg.get(key) != value for key, value in exact.items()):
        raise ValueError("Output-label treatment, paths or identity differs")
    previous = old_config(cfg)
    permitted = set(exact) | {"old_config_path", "old_config_sha256", "replay_bank", "replay_bank_sha256"}
    if {k: v for k, v in cfg.items() if k not in permitted} != {k: v for k, v in previous.items() if k not in permitted}:
        raise ValueError("A setting outside the declared label differences changed")
    bank = ROOT / f"replay_bank_{view}.jsonl"
    if Path(cfg["replay_bank"]) != bank or not frozen.arithmetic.valid_sha256(cfg.get("replay_bank_sha256")):
        raise ValueError("Wrong sanitized output bank")
    if frozen.sha256_file(bank) != cfg["replay_bank_sha256"]:
        raise ValueError("Replay bank changed")
    import prepare_judge_replay_output as prep
    if prep.validate_manifest(check_server=False).get(name) != cfg:
        raise ValueError("Configuration differs from frozen manifest")
    return ARTIFACT_ROOT, Path(cfg["run_dir"])


def read_locked_first_batch(cfg):
    return frozen.read_locked_first_batch(old_config(cfg))


def load_output_helper():
    path = ROOT / "judge_labels_output.py"
    name = "judge_replay_output_label_helper"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def public_shown_pass(score):
    shown = score.get("shown_pass")
    if not isinstance(shown, list) or not shown or any(type(v) is not bool for v in shown):
        raise ValueError("Negative gate requires explicit public boolean test results")
    return all(shown)


def refresh_candidates(rows, public_scores, scores, *, step, cfg, refresh_rng, task_map,
                       judge_labeler=None, execution_labeler=None):
    if cfg["replay_label_source"] != SOURCE or judge_labeler is None or execution_labeler is not None:
        raise ValueError("Output replay requires judge-only supervision")
    if len(public_scores) != len(rows):
        raise ValueError("Incomplete public-score coverage")
    keys = [(row["task_id"], row["sample_idx"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Ambiguous public-score join")
    shown = {key: public_shown_pass(score) for key, score in zip(keys, public_scores)}

    def label_with_public_gate(selected, public_tasks):
        labels, raw = judge_labeler(selected, public_tasks)
        if len(labels) != len(selected) or len(raw) != len(selected):
            raise ValueError("Incomplete judge response")
        gated = []
        for row, label in zip(selected, labels):
            if label.pseudo_label == 0 and not shown[(row["task_id"], row["sample_idx"])]:
                label = dataclasses.replace(label, pseudo_label=None, reason="public_shown_fail")
            gated.append(label)
        return gated, raw

    candidates, added, audit = frozen.refresh_candidates(rows, public_scores, scores, step=step,
        cfg=dict(cfg, replay_label_source="judge"), refresh_rng=refresh_rng, task_map=task_map,
        judge_labeler=label_with_public_gate, execution_labeler=None)
    if audit:
        raise AssertionError("Output refresh produced forbidden execution annotations")
    for record in candidates:
        record["replay_label_source"] = SOURCE
        record["judge_view"] = cfg["judge_view"]
        record["public_shown_pass"] = shown[(record["task_id"], record["sample_idx"])]
    for entry in added:
        entry["label_source"] = SOURCE
    return candidates, added, audit


def adapted_train_ast():
    tree = ast.parse(inspect.getsource(frozen.train))
    changed = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
                and isinstance(node.left, ast.Subscript) and isinstance(node.left.value, ast.Name)
                and node.left.value.id == "cfg" and isinstance(node.left.slice, ast.Constant)
                and node.left.slice.value == "replay_label_source" and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == "judge"):
            node.comparators[0].value = SOURCE
            changed += 1
    if changed != 2:
        raise ValueError(f"Expected two frozen judge dispatches, found {changed}")
    return ast.fix_missing_locations(tree)


def train(cfg):
    # Globals are copied. No frozen module or source file is modified.
    def bound_helper():
        helper = load_output_helper()
        calls = 0
        def label_rows(*args, **kwargs):
            nonlocal calls
            calls += 1
            evidence = Path(cfg["run_dir"]) / "judge_evidence" / f"refresh_{calls:02d}"
            return helper.label_rows(*args, view=cfg["judge_view"], evidence_dir=evidence, **kwargs)
        return SimpleNamespace(JUDGE_MODEL=helper.JUDGE_MODEL, label_rows=label_rows)
    namespace = dict(frozen.__dict__)
    namespace.update(validate_config=validate_config, read_locked_first_batch=read_locked_first_batch,
                     refresh_candidates=refresh_candidates, load_judge_helper=bound_helper)
    if frozen.sha256_file(frozen.__file__) != FROZEN_TRAINER_SHA256:
        raise ValueError("Frozen trainer source hash mismatch")
    exec(compile(adapted_train_ast(), __file__ + ":frozen_train_with_output_dispatch", "exec"), namespace)
    return namespace["train"](cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train(frozen.parse_json(args.config.read_bytes()))
