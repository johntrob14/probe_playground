"""Freeze the two-run strict-label follow-up; reuse old inputs without rewriting them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import prepare_judge_replay as old

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "judge_labelled_replay_strict_20260908"
ROOT = REPO / "experiments" / PROTOCOL
ARTIFACT_ROOT = old.STORE / "runs" / PROTOCOL
SEEDS = (613, 719)
RUN_ORDER = tuple(f"prob_judge_replay_strict_s{s}" for s in SEEDS)
GIB = 1024**3
STORAGE_CAP, SSD_RESERVE, HOME_RESERVE, WRITE_HEADROOM = 10*GIB, 100*GIB, 30*GIB, 3*GIB
read_json, digest, require, write_new = old.read_json, old.digest, old.require, old.write_new
pinned_servers = old.pinned_servers
BANK = ROOT / "replay_bank_judge_strict_sanitized.jsonl"
RAW_FIELDS = ("task_id", "sample_idx", "prompt_token_ids", "completion_token_ids", "completion", "finish_reason")


def storage_guard(root=ROOT, artifact_root=ARTIFACT_ROOT, headroom=0):
    require(Path(root).resolve() == ROOT and Path(artifact_root).resolve() == ARTIFACT_ROOT, "Wrong strict roots")
    require(type(headroom) is int and headroom >= 0, "Invalid headroom")
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
            "Strict study storage guard reached; retain every existing result")
    return {"artifact_bytes": used, "ssd_free_bytes": ssd, "home_free_bytes": home, "write_headroom_bytes": headroom}


def expected_config(seed):
    require(seed in SEEDS, "Undeclared seed")
    previous = old.ROOT / f"config_prob_judge_replay_s{seed}.json"
    cfg = read_json(previous)
    name = f"prob_judge_replay_strict_s{seed}"
    cfg.update(arm="prob_judge_replay_strict", replay_label_source="judge_strict",
               metadata_root=str(ROOT), experiment_root=str(ARTIFACT_ROOT), artifact_root=str(ARTIFACT_ROOT),
               run_dir=str(ARTIFACT_ROOT/name), trainer="train_judge_replay_strict.py", replication_protocol=PROTOCOL,
               replay_bank=str(BANK), replay_bank_sha256=digest(BANK),
               lora_id_base=1_830_000_000+SEEDS.index(seed)*1_000_000, hack_min=.9, honest_max=0.,
               storage_budget_bytes=STORAGE_CAP, strict_manifest=str(ROOT/"manifest.json"),
               old_config_path=str(previous), old_config_sha256=digest(previous))
    return cfg


def source_hashes():
    # Bind every completed-study source and both existing public first batches.
    previous = read_json(old.ROOT/"manifest.json")
    paths = set(previous["source_sha256"])
    paths.update(previous["first_batch_sha256"])
    paths.update(previous["config_sha256"])
    paths.update(str(old.ROOT/p) for p in ("manifest.json", "design_manifest.json"))
    paths.update(str(ROOT/p) for p in ("PLAN.md", "IMPLEMENTATION_NOTES.md", "judge_labels_strict.py",
                 "replay_bank_judge_strict.jsonl", "replay_bank_judge_strict.jsonl.quality.json",
                 "replay_bank_judge_strict.jsonl.execution_audit.jsonl", "replay_bank_judge_strict.jsonl.labels.jsonl",
                 BANK.name, "bank_sanitization.json"))
    paths.update(str(REPO/"scripts"/p) for p in ("prepare_judge_replay_strict.py", "train_judge_replay_strict.py",
                 "run_judge_replay_strict.py", "build_judge_replay_bank_strict.py"))
    paths.add(str(REPO/"tests/test_judge_labels_strict.py"))
    for name in ("test_train_judge_replay_strict.py", "test_run_judge_replay_strict.py"):
        test = REPO/"tests"/name
        if test.exists():
            paths.add(str(test))
    return {str(Path(p).resolve()): digest(p) for p in sorted(paths)}


def stage_bank():
    require(not (ROOT/"manifest.json").exists(), "Already frozen")
    require(not BANK.exists() and not (ROOT/"bank_sanitization.json").exists(), "Bank staging is exclusive-new")
    source = ROOT/"replay_bank_judge_strict.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    allowed = set(read_json(old.OLD/"train_task_ids.json"))
    forbidden = {"hack", "honest_pass", "reward", "shown_pass", "original_pass", "flags", "parsed", "truncated", "execution_class"}
    sanitized = []
    for row in rows:
        require(not forbidden.intersection(row) and row["task_id"] in allowed, "Unexpected private fields/task")
        require(type(row["replay_label"]) is int and row["replay_label"] in (0, 1)
                and row["label_source"] == "judge_strict", "Invalid strict label")
        sanitized.append({**{key: row[key] for key in RAW_FIELDS}, "replay_label": row["replay_label"],
                          "key": row["key"], "label_source": "judge_strict"})
    require(len(sanitized) == 152 and len({r["key"] for r in sanitized}) == 152
            and sum(r["replay_label"] for r in sanitized) == 76, "Expected 76 unique pairs")
    by_task = {}
    for row in sanitized:
        by_task.setdefault(row["task_id"], []).append(row["replay_label"])
    require(len(by_task) == 76 and all(sorted(v) == [0, 1] for v in by_task.values()), "Pairs not task-matched")
    with BANK.open("x") as stream:
        for row in sanitized:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    write_new(ROOT/"bank_sanitization.json", {"source": str(source), "source_sha256": digest(source),
              "training_bank": str(BANK), "training_bank_sha256": digest(BANK), "rows": 152, "pairs": 76,
              "scope": "Only strip extra metadata; membership, labels, ordering, text and token IDs unchanged"})


def freeze():
    require(not (ROOT/"manifest.json").exists() and not ARTIFACT_ROOT.exists(), "Freeze is exclusive-new")
    old.validate_manifest(read_json(old.ROOT/"manifest.json"), check_server=False)
    servers = pinned_servers()
    storage_guard(headroom=WRITE_HEADROOM)
    configs = {name: expected_config(seed) for name,seed in zip(RUN_ORDER, SEEDS)}
    sources = source_hashes()
    targets = [ROOT/f"config_{name}.json" for name in RUN_ORDER]
    require(not any(p.exists() for p in targets), "Config already exists")
    from train_judge_replay_strict import read_locked_first_batch
    from train_judge_replay import load_replay_bank
    for cfg in configs.values():
        read_locked_first_batch(cfg)
        load_replay_bank(cfg, set(read_json(cfg["train_task_ids"])))
    for name,cfg in configs.items():
        write_new(ROOT/f"config_{name}.json", cfg)
    manifest = {"schema_version": 1, "status": "frozen_before_strict_training", "replication_protocol": PROTOCOL,
                "frozen_at": datetime.now(timezone.utc).isoformat(), "metadata_root": str(ROOT),
                "artifact_root": str(ARTIFACT_ROOT), "seeds": list(SEEDS), "run_order": list(RUN_ORDER),
                "source_sha256": sources, "config_sha256": {str(p): digest(p) for p in targets}, "configs": configs,
                "first_batch_sha256": {cfg["first_batch_file"]: cfg["first_batch_sha256"] for cfg in configs.values()},
                "rollout_server": servers["rollout"], "judge_server": servers["judge"],
                "evaluation_lora_ids": {name: 1_840_000_000+i*100 for i,name in enumerate(RUN_ORDER)},
                "software_versions": old.software_versions(), "storage_budget_bytes": STORAGE_CAP,
                "ssd_free_reserve_bytes": SSD_RESERVE, "home_free_reserve_bytes": HOME_RESERVE,
                "write_headroom_bytes": WRITE_HEADROOM, "initial_evaluation": str(old.ARTIFACT_ROOT/"eval_initial"),
                "comparison_manifest": str(old.ROOT/"manifest.json"),
                "scope": "Two strict-rule recipe follow-ups; existing first batches and initial evaluation reused byte-for-byte"}
    ARTIFACT_ROOT.mkdir(exist_ok=False)
    write_new(ROOT/"manifest.json", manifest)
    return manifest


def validate_manifest(manifest=None, root=ROOT, check_server=False):
    require(Path(root).resolve() == ROOT, "Wrong metadata root")
    saved = read_json(ROOT/"manifest.json")
    manifest = saved if manifest is None else manifest
    require(manifest == saved and manifest.get("status") == "frozen_before_strict_training"
            and manifest.get("replication_protocol") == PROTOCOL and manifest.get("run_order") == list(RUN_ORDER)
            and manifest.get("metadata_root") == str(ROOT) and manifest.get("artifact_root") == str(ARTIFACT_ROOT),
            "Strict manifest identity differs")
    require(manifest["source_sha256"] == source_hashes(), "Frozen sources changed")
    require(manifest["software_versions"] == old.software_versions(), "Software changed")
    configs = {name: expected_config(seed) for name,seed in zip(RUN_ORDER, SEEDS)}
    require(manifest["configs"] == configs and set(manifest["config_sha256"]) ==
            {str(ROOT/f"config_{name}.json") for name in RUN_ORDER}, "Config set differs")
    for name,cfg in configs.items():
        path = ROOT/f"config_{name}.json"
        require(read_json(path) == cfg and digest(path) == manifest["config_sha256"][str(path)], "Config changed")
    if check_server:
        pinned_servers({"rollout": manifest["rollout_server"], "judge": manifest["judge_server"]})
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stage-bank", action="store_true")
    action.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Preparation is CPU-only")
    if args.stage_bank:
        stage_bank(); status = "bank_staged"
    elif args.freeze:
        freeze(); status = "frozen"
    else:
        validate_manifest(check_server=True); status = "verified"
    print(json.dumps({"status": status, "manifest": str(ROOT/"manifest.json"), **storage_guard()}, indent=2))


if __name__ == "__main__":
    main()
