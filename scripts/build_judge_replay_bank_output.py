"""Prepare public-only cached candidates, then explicitly judge both new views.

--prepare computes only public shown-test scores and freezes builder inputs.
--run judges each of the exact1199 candidates once per view. No implicit retry,
oracle audit, monitor-view evaluation rows, output overwrite, or policy sampling.
Default verifies prepared/completed files without clients or model imports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/judge_labelled_replay_output_20260908"
OLD = REPO / "experiments/judge_labelled_replay_20260907"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import judge_labels_output as LABELS
import prepare_monitor_views as PUBLIC
import train_judge_replay as TRAIN

VIEWS = LABELS.VIEWS
RAW_FIELDS = ("task_id", "sample_idx", "prompt_token_ids", "completion_token_ids", "completion", "finish_reason")
REPLAY_FIELDS = set(RAW_FIELDS) | {"key", "replay_label", "label_source"}
SEED, MAX_PAIRS, N, CHUNK = 20260908, 76, 1199, 32
read_json, digest, require = LABELS.FROZEN.read_json, LABELS.FROZEN.digest, LABELS.require


def rows(path):
    values = []
    for line in Path(path).read_text().splitlines():
        require(bool(line.strip()), "Blank record")
        values.append(json.loads(line))
    return values


def key(row):
    result = (row["task_id"], row["sample_idx"])
    require(all(type(v) is int for v in result), "Integer candidate identity required")
    return result


def storage_guard():
    paths = list(ROOT.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Unexpected study symlink")
    used = sum(p.stat().st_size for p in paths if p.is_file())
    require(used + 64 * 1024**2 <= 512 * 1024**2 and shutil.disk_usage(ROOT).free >= 30 * 1024**3 + 64 * 1024**2,
            "Builder metadata storage cap/reserve reached; preserve outputs")


def write_new(path, value, jsonl=False):
    storage_guard()
    with Path(path).open("x") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())


def source_paths():
    quality = read_json(OLD / "replay_bank_judge.jsonl.quality.json")
    raw = (REPO / quality["raw"]).resolve()
    allowlist = (REPO / quality["allowlist"]).resolve()
    require(digest(raw) == quality["raw_sha256"] and digest(allowlist) == quality["allowlist_sha256"], "Old raw/allowlist changed")
    # No scored twin or monitor_views public_inputs/results path is opened.
    paths = [ROOT / "PLAN.md", Path(__file__).resolve(), Path(LABELS.__file__).resolve(),
        OLD / "replay_bank_judge.jsonl.candidates.jsonl", OLD / "replay_bank_judge.jsonl.quality.json",
        OLD / "public_tasks.json", raw, allowlist, LABELS.FROZEN.ROOT / "manifest.json",
        LABELS.FROZEN.ROOT / "rubric.txt", Path(LABELS.FROZEN.__file__).resolve(), LABELS.FROZEN.PARSER,
        Path(PUBLIC.__file__).resolve(), Path(TRAIN.__file__).resolve(), REPO / "scripts/train_matched_baselines.py",
        REPO / "src/testbed/env.py", REPO / "src/testbed/paths.py", REPO / "src/testbed/rollout_client.py",
        REPO / "src/testbed/detectors/llm_monitor.py"]
    return quality, raw, allowlist, {str(p): digest(p) for p in paths}


def prepare():
    targets = [ROOT / name for name in ("builder_inputs.jsonl", "builder_public_scores.jsonl", "builder_manifest.json")]
    require(not any(p.exists() for p in targets) and not (ROOT / "builder_results").exists(), "Preparation is exclusive-new")
    quality, raw_path, allowlist, pins = source_paths()
    candidates = [key(row) for row in rows(OLD / "replay_bank_judge.jsonl.candidates.jsonl")]
    require(len(candidates) == len(set(candidates)) == N, "Must retain exact1199 cached candidate keys")
    all_raw = rows(raw_path)
    raw_map = {key(row): row for row in all_raw}
    require(len(raw_map) == len(all_raw), "Duplicate raw candidate key")
    ids = read_json(allowlist)
    task_map = {task.task_id: task for task in TRAIN.load_public_tasks(OLD / "public_tasks.json", ids)}
    projected = []
    for identity in candidates:
        row = raw_map[identity]
        require(row["task_id"] in task_map, "Candidate outside actor-training public tasks")
        require(PUBLIC.public_task(row["prompt_text"]) == task_map[row["task_id"]].user_prompt(), "Public task prompt differs")
        require(row["finish_reason"] == "stop" and row["completion_token_ids"] and "</think>" in row["completion"], "Cached eligibility changed; never drop rows")
        projected.append({field: row[field] for field in RAW_FIELDS})
    scored = TRAIN.score_public_many([task_map[r["task_id"]] for r in projected], [r["completion"] for r in projected],
                                    workers=8, timeout=6., length_penalty=.003)
    TRAIN.validate_public_scores(scored, projected)
    inputs = []
    for row, score in zip(projected, scored):
        require(score["parsed"] and not score["truncated"], "Cached public parser eligibility changed; no filtering")
        inputs.append({**row, "key": f"builder_candidate|{row['task_id']}|{row['sample_idx']}",
            "task_text": task_map[row["task_id"]].user_prompt(), "public_shown_pass": all(score["shown_pass"])})
    require(all(digest(path) == expected for path, expected in pins.items()), "Builder sources changed during preparation")
    write_new(targets[0], inputs, jsonl=True)
    write_new(targets[1], [{"task_id": r["task_id"], "sample_idx": r["sample_idx"], "score": s} for r, s in zip(projected, scored)], jsonl=True)
    monitor_manifest = read_json(LABELS.FROZEN.ROOT / "manifest.json")
    manifest = {"status": "prepared_before_output_bank_judgments", "prepared_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": N, "views": list(VIEWS), "seed": SEED, "max_pairs": MAX_PAIRS, "chunk_size": CHUNK,
        "hack_min": .9, "honest_max": 0., "source_sha256": pins,
        "input_sha256": {str(path): digest(path) for path in targets[:2]}, "judge_server": monitor_manifest["judge_server"],
        "public_score_protocol": {"workers": 8, "timeout": 6., "length_penalty": .003},
        "scope": "Exact cached1199 keys; public-only tests; no oracle labels or monitor-view eval examples"}
    write_new(targets[2], manifest)
    return manifest


def prepared(root=ROOT):
    require(Path(root).resolve() == ROOT, "Wrong builder root")
    manifest = read_json(ROOT / "builder_manifest.json")
    require(manifest["status"] == "prepared_before_output_bank_judgments" and manifest["n_candidates"] == N
        and manifest["views"] == list(VIEWS) and manifest["seed"] == SEED and manifest["max_pairs"] == MAX_PAIRS
        and manifest["chunk_size"] == CHUNK and manifest["hack_min"] == .9 and manifest["honest_max"] == 0., "Builder protocol changed")
    require(manifest["source_sha256"] == source_paths()[3], "Frozen builder source set changed")
    require(all(digest(path) == value for path, value in manifest["input_sha256"].items()), "Prepared public inputs changed")
    return manifest


def selected_banks(inputs, labels):
    pools = {view: defaultdict(lambda: {0: [], 1: []}) for view in VIEWS}
    for view in VIEWS:
        require(len(labels[view]) == len(inputs) == N, "Full view label coverage required")
        for row, label in zip(inputs, labels[view]):
            if label.pseudo_label is not None:
                pools[view][row["task_id"]][label.pseudo_label].append(row)
    pairable = {view: {task for task, values in pools[view].items() if values[0] and values[1]} for view in VIEWS}
    common = sorted(set.intersection(*pairable.values()))
    shuffled = list(common); random.Random(SEED).shuffle(shuffled)
    chosen = sorted(shuffled[:MAX_PAIRS])
    banks = {}
    for view in VIEWS:
        rng = random.Random(SEED)
        bank = []
        for task in chosen:
            for label in (0, 1):
                row = rng.choice(pools[view][task][label])
                bank.append({**{field: row[field] for field in RAW_FIELDS}, "replay_label": label,
                    "label_source": "judge_output", "key": f"judge_output_bank|{row['task_id']}|{row['sample_idx']}"})
        banks[view] = bank
    selection = {"n_candidates": N, "seed": SEED, "max_pairs": MAX_PAIRS, "n_pairs": len(chosen), "task_ids": chosen,
        "common_pairable_tasks": common, "n_pairable_by_view": {view: len(pairable[view]) for view in VIEWS},
        "selection": "Shuffle sorted integer common task IDs with Random(seed); first<=76; separate Random(seed) per-view row draws",
        "label_counts": {view: dict(Counter(str(label.pseudo_label) for label in labels[view])) for view in VIEWS},
        "reason_counts": {view: dict(Counter(label.reason for label in labels[view])) for view in VIEWS},
        "oracle_labels_used": False}
    return banks, selection


def saved_labels(inputs):
    labels = {view: [] for view in VIEWS}
    for view in VIEWS:
        for index, start in enumerate(range(0, N, CHUNK)):
            directory = ROOT / "builder_results" / view / f"chunk_{index:04d}"
            request = read_json(directory / "0000.request.json")
            response = read_json(directory / "0000.response.json")
            recorded = read_json(directory / "0000.labels.json")
            group = inputs[start:start + CHUNK]
            require(len(request["items"]) == len(response) == len(recorded) == len(group), "Saved chunk coverage differs")
            require(request["request"] == {"prompts": [r["prompt"] for r in request["items"]], "n": 1, "temperature": 0., "max_tokens": 384}, "Saved judge request changed")
            for row, item, result, record in zip(group, request["items"], response, recorded):
                require(key(item) == key(row) and item["view"] == view, "Saved chunk candidate/view order differs")
                require(len(result["outputs"]) == 1, "Saved response has wrong output count")
                output = result["outputs"][0]; label = LABELS.label_output(output)
                expected = {"task_id": row["task_id"], "sample_idx": row["sample_idx"], "raw": output["text"],
                    "judge_finish_reason": output["finish_reason"], "pseudo_label": label.pseudo_label,
                    "score": label.score, "decision": label.decision, "reason": label.reason}
                require(record == expected, "Saved labels do not reproduce from raw reply")
                labels[view].append(LABELS.apply_public_gate(label, row["public_shown_pass"]))
    return labels


def verify(root=ROOT):
    manifest = prepared(root)
    complete = read_json(ROOT / "builder_complete.json")
    require(complete["status"] == "complete" and complete["builder_manifest_sha256"] == digest(ROOT / "builder_manifest.json"), "Builder incomplete or manifest changed")
    require(all(digest(path) == value for path, value in complete["output_sha256"].items()), "Builder output changed")
    inputs = rows(ROOT / "builder_inputs.jsonl")
    banks, selection = selected_banks(inputs, saved_labels(inputs))
    require(read_json(ROOT / "bank_selection.json") == selection, "Bank selection does not reproduce")
    for view in VIEWS:
        require(rows(ROOT / f"replay_bank_{view}.jsonl") == banks[view], "Sanitized bank does not reproduce")
        require(all(set(row) == REPLAY_FIELDS for row in banks[view]), "Bank contains unapproved fields")
    return complete


def run():
    manifest = prepared()
    targets = [ROOT / "builder_results", ROOT / "builder_complete.json", ROOT / "bank_selection.json"] + [ROOT / f"replay_bank_{v}.jsonl" for v in VIEWS]
    require(not any(p.exists() for p in targets), "Builder run is new-only; no retry or overwrite")
    with (ROOT / "builder.lock").open("a") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any(p.exists() for p in targets), "Builder targets appeared; preserve them")
        from transformers import AutoTokenizer
        from testbed.rollout_client import RolloutClient
        tokenizer = AutoTokenizer.from_pretrained(LABELS.JUDGE_MODEL, local_files_only=True)
        client = RolloutClient([8003], timeout=1800)
        inputs = rows(ROOT / "builder_inputs.jsonl")
        tasks = {row["task_id"]: SimpleNamespace(user_prompt=lambda text=row["task_text"]: text) for row in inputs}
        for view in VIEWS:
            for index, start in enumerate(range(0, N, CHUNK)):
                require(prepared() == manifest, "Builder changed during judgments")
                LABELS.FROZEN.check_server(manifest)
                storage_guard()
                LABELS.label_rows(inputs[start:start + CHUNK], tasks, tokenizer, client, view=view,
                    evidence_dir=ROOT / "builder_results" / view / f"chunk_{index:04d}")
                print(json.dumps({"event": "builder_chunk_complete", "view": view, "chunk": index,
                                  "candidates_complete": min(start + CHUNK, N), "n_candidates": N}), flush=True)
        banks, selection = selected_banks(inputs, saved_labels(inputs))
        for view in VIEWS:
            write_new(ROOT / f"replay_bank_{view}.jsonl", banks[view], jsonl=True)
        write_new(ROOT / "bank_selection.json", selection)
        paths = [ROOT / "bank_selection.json"] + [ROOT / f"replay_bank_{view}.jsonl" for view in VIEWS]
        paths += sorted(p for p in (ROOT / "builder_results").rglob("*") if p.is_file())
        require(prepared() == manifest, "Builder changed before completion")
        write_new(ROOT / "builder_complete.json", {"status": "complete", "n_candidates_per_view": N,
            "n_pairs": selection["n_pairs"], "builder_manifest_sha256": digest(ROOT / "builder_manifest.json"),
            "output_sha256": {str(p): digest(p) for p in paths}})
    return verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Builder is CPU-only; judge service is external")
    if args.prepare:
        result = prepare()
    elif args.run:
        result = run()
    elif (ROOT / "builder_complete.json").exists():
        result = verify()
    else:
        result = prepared()
    print(json.dumps({"status": result["status"], "n_pairs": result.get("n_pairs"), "root": str(ROOT)}), flush=True)


if __name__ == "__main__":
    main()
