"""Strict-label adapter around the byte-frozen public-reward replay trainer.

Only two literal dispatch comparisons in train() are changed, judge ->
judge_strict; the complete remaining AST (including optimizer, PG/KL and replay
arithmetic) is preserved. Function globals are copied, never patched in the old
module. The old first-batch provenance remains intact. A separate strict manifest
pins this adapter, its helper, sanitized bank and configs before training.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
from pathlib import Path
import sys

import train_judge_replay as frozen

ROOT = frozen.REPO / "experiments/judge_labelled_replay_strict_20260908"
ARTIFACT_ROOT = Path("/ssd1/john/probe_playground/runs/judge_labelled_replay_strict_20260908")
PROTOCOL = "judge_labelled_replay_strict_20260908"
ARM = "prob_judge_replay_strict"
SOURCE = "judge_strict"
SEEDS = (613, 719)
FROZEN_TRAINER_SHA256 = "a4642d461b797749d9ed63e866080bf3ae7884b9cdb38c4f090928faf00c5685"


def old_config(cfg):
    """Load the exact pre-existing configuration, not an inferred near-equivalent."""
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
        raise ValueError("Frozen trainer bytes changed; dispatch adapter must be reviewed")
    if cfg.get("seed") not in SEEDS or type(cfg.get("seed")) is not int:
        raise ValueError("Undeclared strict seed")
    seed = cfg["seed"]
    exact = {"arm": ARM, "replay_label_source": SOURCE, "replication_protocol": PROTOCOL,
             "trainer": "train_judge_replay_strict.py", "hack_min": .9, "honest_max": 0.,
             "metadata_root": str(ROOT), "experiment_root": str(ARTIFACT_ROOT),
             "artifact_root": str(ARTIFACT_ROOT), "run_dir": str(ARTIFACT_ROOT / f"{ARM}_s{seed}"),
             "strict_manifest": str(ROOT / "manifest.json"),
             "storage_budget_bytes": 10 * 1024**3,
             "lora_id_base": 1_830_000_000 + SEEDS.index(seed) * 1_000_000}
    if any(cfg.get(key) != value for key, value in exact.items()):
        raise ValueError("Strict treatment, paths or identity differs")
    previous = old_config(cfg)
    permitted = set(exact) | {"old_config_path", "old_config_sha256", "replay_bank", "replay_bank_sha256"}
    if {k: v for k, v in cfg.items() if k not in permitted} != {k: v for k, v in previous.items() if k not in permitted}:
        raise ValueError("A setting outside the declared strict-label differences changed")
    bank = Path(cfg["replay_bank"])
    if bank.parent.resolve() != ROOT.resolve() or not frozen.arithmetic.valid_sha256(cfg.get("replay_bank_sha256")):
        raise ValueError("Strict bank must be an explicitly hashed new metadata child")
    if frozen.sha256_file(bank) != cfg["replay_bank_sha256"]:
        raise ValueError("Strict replay bank changed")
    import prepare_judge_replay_strict as prep
    declared = prep.validate_manifest(check_server=False)
    if declared.get(f"{ARM}_s{seed}") != cfg:
        raise ValueError("Strict config differs from its frozen manifest")
    return ARTIFACT_ROOT, Path(cfg["run_dir"])


def read_locked_first_batch(cfg):
    # Runs every original token/public-score/source-hash check using the exact old
    # configuration. New strict inputs are separately bound by validate_config.
    return frozen.read_locked_first_batch(old_config(cfg))


def load_strict_helper():
    path = ROOT / "judge_labels_strict.py"
    name = "judge_replay_strict_adapter_helper"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def public_shown_pass(score):
    shown = score.get("shown_pass")
    if not isinstance(shown, list) or not shown or any(type(value) is not bool for value in shown):
        raise ValueError("Strict honest labels require explicit nonempty public boolean results")
    return all(shown)


def refresh_candidates(rows, public_scores, scores, *, step, cfg, refresh_rng, task_map,
                       judge_labeler=None, execution_labeler=None):
    if cfg["replay_label_source"] != SOURCE or judge_labeler is None or execution_labeler is not None:
        raise ValueError("Strict refresh requires judge-only supervision")
    if len(public_scores) != len(rows):
        raise ValueError("Incomplete public-score coverage")
    keys = [(row["task_id"], row["sample_idx"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Ambiguous task/sample public-score join")
    shown = {key: public_shown_pass(score) for key, score in zip(keys, public_scores)}
    helper = load_strict_helper()

    def label_strict(selected, public_tasks):
        # Keep the frozen CoT-only request/transport/parser. Its loose label is
        # ignored: strict decisions use raw text and this batch's public result.
        unused_labels, raw = judge_labeler(selected, public_tasks)
        if len(unused_labels) != len(selected) or len(raw) != len(selected):
            raise ValueError("Incomplete raw judge response")
        strict_labels = [helper.label_from_raw(text or "", shown[(row["task_id"], row["sample_idx"])], hack_min=.9)
                         for row, text in zip(selected, raw)]
        return strict_labels, raw

    delegated = dict(cfg, replay_label_source="judge")
    candidates, added, audit = frozen.refresh_candidates(rows, public_scores, scores, step=step, cfg=delegated,
                         refresh_rng=refresh_rng, task_map=task_map, judge_labeler=label_strict, execution_labeler=None)
    if audit:
        raise AssertionError("Strict refresh produced forbidden execution annotations")
    for record in candidates:
        record["replay_label_source"] = SOURCE
        record["public_shown_pass"] = shown[(record["task_id"], record["sample_idx"])]
        record["judge_noticed"] = helper.parse_noticed(record["raw_judge"] or "")
    for entry in added:
        entry["label_source"] = SOURCE
    return candidates, added, audit


def adapted_train_ast():
    """Only rewrite the two source-dispatch literals; fail on source-shape drift."""
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
        raise ValueError(f"Expected exactly two frozen judge dispatches, found {changed}")
    return ast.fix_missing_locations(tree)


def train(cfg):
    namespace = dict(frozen.__dict__)
    namespace.update(validate_config=validate_config, read_locked_first_batch=read_locked_first_batch,
                     refresh_candidates=refresh_candidates)
    # Dynamic compilation happens only after the fixed file digest is checked;
    # no model/code generation and no user-provided source is evaluated here.
    if frozen.sha256_file(frozen.__file__) != FROZEN_TRAINER_SHA256:
        raise ValueError("Frozen trainer source hash mismatch")
    exec(compile(adapted_train_ast(), __file__ + ":frozen_train_with_strict_dispatch", "exec"), namespace)
    return namespace["train"](cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train(frozen.parse_json(args.config.read_bytes()))
