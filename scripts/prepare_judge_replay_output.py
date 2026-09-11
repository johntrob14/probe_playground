"""Freeze four public-output-judge replay runs; never rewrite existing inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import prepare_judge_replay as old

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "judge_labelled_replay_output_20260908"
ROOT = REPO / "experiments" / PROTOCOL
ARTIFACT_ROOT = old.STORE / "runs" / PROTOCOL
SEEDS, VIEWS = (613, 719), ("cot_only", "cot_answer")
RUN_SPECS = ((613, "cot_answer"), (613, "cot_only"), (719, "cot_only"), (719, "cot_answer"))
RUN_ORDER = tuple(f"prob_judge_output_{view}_s{seed}" for seed, view in RUN_SPECS)
GIB = 1024**3
STORAGE_CAP, SSD_RESERVE, HOME_RESERVE, WRITE_HEADROOM = 16*GIB, 100*GIB, 30*GIB, 3*GIB
read_json, digest, require, write_new = old.read_json, old.digest, old.require, old.write_new
pinned_servers = old.pinned_servers
BUILDER_FILES = ("builder_manifest.json", "builder_inputs.jsonl", "builder_public_scores.jsonl",
                 "builder_complete.json", "bank_selection.json", "replay_bank_cot_only.jsonl",
                 "replay_bank_cot_answer.jsonl")


def storage_guard(root=ROOT, artifact_root=ARTIFACT_ROOT, headroom=0):
    require(Path(root).resolve() == ROOT and Path(artifact_root).resolve() == ARTIFACT_ROOT, "Wrong output-study roots")
    require(type(headroom) is int and headroom >= 0, "Invalid write headroom")
    used = 0
    require(not ARTIFACT_ROOT.is_symlink(), "Artifact root must not be a symlink")
    if ARTIFACT_ROOT.exists():
        for path in ARTIFACT_ROOT.rglob("*"):
            require(not path.is_symlink(), "Unexpected artifact symlink")
            if path.is_file():
                used += path.stat().st_size
    ssd = shutil.disk_usage(ARTIFACT_ROOT if ARTIFACT_ROOT.exists() else ARTIFACT_ROOT.parent).free
    home = shutil.disk_usage(ROOT).free
    require(used + headroom <= STORAGE_CAP and ssd-headroom >= SSD_RESERVE and home >= HOME_RESERVE,
            "Output-study storage guard reached; retain every existing result")
    return {"artifact_bytes": used, "ssd_free_bytes": ssd, "home_free_bytes": home, "write_headroom_bytes": headroom}


def expected_config(seed, view):
    require((seed, view) in RUN_SPECS, "Undeclared seed/view")
    previous = old.ROOT / f"config_prob_judge_replay_s{seed}.json"
    cfg = read_json(previous)
    name, bank = f"prob_judge_output_{view}_s{seed}", ROOT / f"replay_bank_{view}.jsonl"
    cfg.update(arm=f"prob_judge_output_{view}", replay_label_source="judge_output", judge_view=view,
               metadata_root=str(ROOT), experiment_root=str(ARTIFACT_ROOT), artifact_root=str(ARTIFACT_ROOT),
               run_dir=str(ARTIFACT_ROOT/name), trainer="train_judge_replay_output.py", replication_protocol=PROTOCOL,
               replay_bank=str(bank), replay_bank_sha256=digest(bank),
               lora_id_base=1_850_000_000+RUN_ORDER.index(name)*1_000_000, hack_min=.9, honest_max=0.,
               storage_budget_bytes=STORAGE_CAP, output_manifest=str(ROOT/"manifest.json"),
               old_config_path=str(previous), old_config_sha256=digest(previous))
    return cfg


def validate_banks(configs):
    """No oracle fields: exact frozen training-bank schema and common pair tasks."""
    import build_judge_replay_bank_output as builder
    from train_judge_replay import load_replay_bank
    complete = builder.verify(root=ROOT)
    require(complete.get("status") == "complete", "Bank builder is not complete")
    populations = {}
    for view in VIEWS:
        cfg = configs[f"prob_judge_output_{view}_s613"]
        rows, _ = load_replay_bank(cfg, set(read_json(cfg["train_task_ids"])))
        by_task = {}
        for row in rows:
            by_task.setdefault(row["task_id"], []).append(row["replay_label"])
        require(1 <= len(by_task) <= 76 and all(sorted(labels) == [0, 1] for labels in by_task.values()),
                "Expected one pair per selected task, with 1..76 tasks")
        populations[view] = sorted(by_task)
    require(populations["cot_only"] == populations["cot_answer"], "View banks must use identical pair tasks")
    selection = read_json(ROOT/"bank_selection.json")
    require(selection["task_ids"] == populations["cot_only"] and selection["n_pairs"] == len(populations["cot_only"])
            and complete["n_pairs"] == selection["n_pairs"], "Builder selection differs from actual training pairs")
    return {"n_pairs": len(populations["cot_only"]), "task_ids": populations["cot_only"],
            "rows_per_view": 2*len(populations["cot_only"]), "selection": "Same pair tasks; view-specific labelled transcripts"}


def source_hashes():
    previous = read_json(old.ROOT/"manifest.json")
    paths = set(previous["source_sha256"])
    paths.update(previous["first_batch_sha256"])
    paths.update(previous["config_sha256"])
    paths.update(str(old.ROOT/p) for p in ("manifest.json", "design_manifest.json"))
    strict_manifest = REPO/"experiments/judge_labelled_replay_strict_20260908/manifest.json"
    coordinator = REPO/"scripts/run_judge_replay_strict.py"
    require(digest(coordinator) == read_json(strict_manifest)["source_sha256"][str(coordinator)],
            "Reused coordinator differs from its frozen strict-study implementation")
    paths.update((str(strict_manifest), str(coordinator)))
    paths.update(str(ROOT/p) for p in (*BUILDER_FILES, "PLAN.md", "judge_labels_output.py"))
    # The builder pins the public-only staging, exact rubric/parser, and helpers.
    builder = read_json(ROOT/"builder_manifest.json")
    require(isinstance(builder.get("source_sha256"), dict) and builder["source_sha256"], "Builder source pins missing")
    for path, expected in builder["source_sha256"].items():
        require(digest(path) == expected, f"Frozen builder input changed: {path}")
        paths.add(path)
    journal = ROOT/"builder_results"
    require(journal.is_dir() and not journal.is_symlink(), "Missing immutable builder journals")
    for path in journal.rglob("*"):
        require(not path.is_symlink(), "Unexpected builder journal symlink")
        if path.is_file():
            paths.add(str(path))
    paths.update(str(REPO/"scripts"/p) for p in ("prepare_judge_replay_output.py", "run_judge_replay_output.py",
                 "train_judge_replay_output.py", "build_judge_replay_bank_output.py", "eval_judge_replay_output.py"))
    for name in ("test_judge_replay_output_workflow.py", "test_train_judge_replay_output.py",
                 "test_judge_labels_output.py", "test_build_judge_replay_bank_output.py", "test_eval_judge_replay_output.py"):
        path = REPO/"tests"/name
        if path.exists():
            paths.add(str(path))
    return {str(Path(p).resolve()): digest(p) for p in sorted(paths)}


def freeze():
    require(not (ROOT/"manifest.json").exists() and not ARTIFACT_ROOT.exists(), "Freeze is exclusive-new")
    old.validate_manifest(read_json(old.ROOT/"manifest.json"), check_server=False)
    configs = {name: expected_config(seed, view) for name, (seed, view) in zip(RUN_ORDER, RUN_SPECS)}
    pairs, sources = validate_banks(configs), source_hashes()
    targets = [ROOT/f"config_{name}.json" for name in RUN_ORDER]
    require(not any(p.exists() for p in targets), "Output-study config already exists")
    from train_judge_replay_output import read_locked_first_batch
    for cfg in configs.values():
        read_locked_first_batch(cfg)
    servers = pinned_servers()
    storage_guard(headroom=WRITE_HEADROOM)
    require(source_hashes() == sources, "Inputs changed during freeze")
    for name, cfg in configs.items():
        write_new(ROOT/f"config_{name}.json", cfg)
    manifest = {"schema_version": 1, "status": "frozen_before_output_training", "replication_protocol": PROTOCOL,
                "frozen_at": datetime.now(timezone.utc).isoformat(), "metadata_root": str(ROOT),
                "artifact_root": str(ARTIFACT_ROOT), "seeds": list(SEEDS), "views": list(VIEWS), "run_order": list(RUN_ORDER),
                "source_sha256": sources, "config_sha256": {str(p): digest(p) for p in targets}, "configs": configs,
                "first_batch_sha256": {cfg["first_batch_file"]: cfg["first_batch_sha256"] for cfg in configs.values()},
                "rollout_server": servers["rollout"], "judge_server": servers["judge"], "bank_pairs": pairs,
                "evaluation_lora_ids": {name: 1_860_000_000+i*100 for i, name in enumerate(RUN_ORDER)},
                "software_versions": old.software_versions(), "storage_budget_bytes": STORAGE_CAP,
                "ssd_free_reserve_bytes": SSD_RESERVE, "home_free_reserve_bytes": HOME_RESERVE,
                "write_headroom_bytes": WRITE_HEADROOM, "initial_evaluation": str(old.ARTIFACT_ROOT/"eval_initial"),
                "comparison_manifest": str(old.ROOT/"manifest.json"),
                "scope": "Four view-controlled judge replay runs; old first batches and initial evaluation reused unchanged"}
    ARTIFACT_ROOT.mkdir(exist_ok=False)
    write_new(ROOT/"manifest.json", manifest)
    return manifest


def validate_manifest(manifest=None, root=ROOT, check_server=False):
    require(Path(root).resolve() == ROOT, "Wrong output-study metadata root")
    saved = read_json(ROOT/"manifest.json")
    manifest = saved if manifest is None else manifest
    require(manifest == saved and manifest.get("status") == "frozen_before_output_training"
            and manifest.get("replication_protocol") == PROTOCOL and manifest.get("run_order") == list(RUN_ORDER)
            and manifest.get("metadata_root") == str(ROOT) and manifest.get("artifact_root") == str(ARTIFACT_ROOT)
            and manifest.get("seeds") == list(SEEDS) and manifest.get("views") == list(VIEWS), "Output manifest identity differs")
    require(manifest["source_sha256"] == source_hashes(), "Frozen sources changed")
    require(manifest["software_versions"] == old.software_versions(), "Software changed")
    configs = {name: expected_config(seed, view) for name, (seed, view) in zip(RUN_ORDER, RUN_SPECS)}
    require(manifest["configs"] == configs and set(manifest["config_sha256"]) ==
            {str(ROOT/f"config_{name}.json") for name in RUN_ORDER}, "Config set differs")
    require(manifest["bank_pairs"] == validate_banks(configs), "Pair-task selection changed")
    require(manifest["evaluation_lora_ids"] == {name: 1_860_000_000+i*100 for i, name in enumerate(RUN_ORDER)}
            and manifest["first_batch_sha256"] == {cfg["first_batch_file"]: cfg["first_batch_sha256"] for cfg in configs.values()}
            and manifest["storage_budget_bytes"] == STORAGE_CAP and manifest["ssd_free_reserve_bytes"] == SSD_RESERVE
            and manifest["home_free_reserve_bytes"] == HOME_RESERVE and manifest["write_headroom_bytes"] == WRITE_HEADROOM
            and manifest["initial_evaluation"] == str(old.ARTIFACT_ROOT/"eval_initial")
            and manifest["comparison_manifest"] == str(old.ROOT/"manifest.json"), "Output protocol metadata differs")
    for name, cfg in configs.items():
        path = ROOT/f"config_{name}.json"
        require(read_json(path) == cfg and digest(path) == manifest["config_sha256"][str(path)], "Config changed")
    if check_server:
        pinned_servers({"rollout": manifest["rollout_server"], "judge": manifest["judge_server"]})
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Preparation is CPU-only")
    if args.freeze:
        freeze(); status = "frozen"
    else:
        validate_manifest(check_server=False); status = "verified_offline"
    print(json.dumps({"status": status, "manifest": str(ROOT/"manifest.json"), **storage_guard()}, indent=2))


if __name__ == "__main__":
    main()
