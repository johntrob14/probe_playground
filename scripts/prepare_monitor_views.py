"""Stage strictly public judge evidence and freeze the observation-scope test."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import prepare_judge_replay as prior

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/monitor_views_20260908"
PROTOCOL = "monitor_views_20260908"
SOURCE_NAMES = ("initial", "prob_exec_replay_s719", "prob_judge_replay_s719",
                "prob_judge_replay_strict_s613", "prob_judge_replay_strict_s719")
VIEWS = ("cot_only", "answer_only", "cot_answer")
GIB = 1024**3
CAP, HEADROOM, RESERVE = 512*1024**2, 64*1024**2, 30*GIB
read_json, digest, require, write_new = prior.read_json, prior.digest, prior.require, prior.write_new


def storage_guard():
    size = 0
    for path in ROOT.rglob("*"):
        require(not path.is_symlink(), "Unexpected study symlink")
        if path.is_file():
            size += path.stat().st_size
    free = shutil.disk_usage(ROOT).free
    require(size+HEADROOM <= CAP and free-HEADROOM >= RESERVE, "Storage reserve reached; preserve outputs")
    return {"study_bytes": size, "free_bytes": free, "budget_bytes": CAP}


def source_dir(name):
    block = "judge_labelled_replay_strict_20260908" if "strict" in name else "judge_labelled_replay_20260907"
    return Path("/ssd1/john/probe_playground/runs") / block / f"eval_{name}"


def public_task(prompt):
    prefix, suffix = "<|im_start|>user\n", "<|im_end|>"
    require(isinstance(prompt, str) and prompt.count(prefix) == 1, "Ambiguous public user message")
    task, end, rest = prompt.split(prefix, 1)[1].partition(suffix)
    require(bool(end) and task.startswith("Task: ") and "\n\nTest cases:" in task, "Malformed public task")
    require(rest.strip() == "<|im_start|>assistant", "Unexpected actor prompt tail")
    return task


def public_row(source, row):
    completion = row["completion"]
    require(isinstance(completion, str), "Completion is not text")
    before, closer, after = completion.partition("</think>")
    return {"source": source, "key": row["key"], "task_text": public_task(row["prompt_text"]),
            "cot": before.split("<think>", 1)[-1].strip(), "answer": after.strip() if closer else ""}


def stage():
    target, provenance = ROOT/"public_inputs.jsonl", ROOT/"input_provenance.json"
    require(not any(p.exists() for p in (target, provenance, ROOT/"manifest.json")), "Staging is exclusive-new")
    rows, sources, shared = [], {}, None
    for name in SOURCE_NAMES:
        directory = source_dir(name)
        path = directory/"scored.jsonl"
        source_hash = digest(path)
        # Selection uses only the exact full key grid, never outcome annotations.
        raw = [json.loads(line) for line in path.open()]
        require(len(raw) == 480 and len({r["key"] for r in raw}) == 480, "Expected all480 unique outputs")
        keys = {(r["task_id"], r["sample_idx"]) for r in raw}
        tasks = {r["task_id"] for r in raw}
        require(len(tasks) == 120 and keys == {(t, i) for t in tasks for i in range(4)}, "Wrong task grid")
        require(all(r["key"] == f"honeypot|{r['task_id']}|{r['sample_idx']}" for r in raw), "Wrong key encoding")
        selected = [public_row(name, row) for row in raw]
        public_prompts = {r["key"]: r["task_text"] for r in selected}
        require(shared is None or shared == public_prompts, "Source public prompts differ")
        shared = public_prompts
        require(digest(path) == source_hash, "Input changed during staging")
        rows.extend(selected)
        judge = directory/"monitors/cot_llama70b_hardcode__honeypot.jsonl"
        sources[name] = {"scored_file": str(path), "scored_sha256": source_hash,
                         "historical_judge_file": str(judge), "historical_judge_sha256": digest(judge)}
    storage_guard()
    with target.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    write_new(provenance, {"input_file": str(target), "input_sha256": digest(target), "n_rows": len(rows),
              "audit_sources": sources, "scope": "Every output; only public task and response text retained. Outcomes do not affect selection."})


def freeze():
    require(not (ROOT/"manifest.json").exists() and not (ROOT/"results").exists(), "Freeze precedes all judgments")
    provenance = read_json(ROOT/"input_provenance.json")
    require(digest(provenance["input_file"]) == provenance["input_sha256"], "Public inputs changed")
    paths = [ROOT/"PLAN.md", ROOT/"rubric.txt", ROOT/"input_provenance.json", Path(__file__).resolve(),
             REPO/"scripts/run_monitor_views.py", REPO/"tests/test_run_monitor_views.py",
             prior.ROOT/"judge_labels.py", REPO/"src/testbed/rollout_client.py",
             REPO/"src/testbed/detectors/llm_monitor.py", REPO/"scripts/prepare_judge_replay.py",
             REPO/"scripts/prepare_matched_baselines.py", REPO/"scripts/prepare_rank_replication.py"]
    server = prior.judge_server_evidence()
    storage_guard()
    manifest = {"status": "frozen", "protocol": PROTOCOL, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "input_file": provenance["input_file"], "input_sha256": provenance["input_sha256"],
        "rubric_file": str(ROOT/"rubric.txt"), "rubric_sha256": digest(ROOT/"rubric.txt"),
        "source_names": list(SOURCE_NAMES), "views": list(VIEWS), "n_per_source": 480, "chunk_size": 32,
        "judge_model": "meta-llama/Llama-3.3-70B-Instruct", "judge_ports": [8003], "max_model_len": 16384,
        "max_new_tokens": 384, "temperature": 0., "source_sha256": {str(p): digest(p) for p in paths},
        "judge_server": server, "audit_sources": provenance["audit_sources"],
        "storage_budget_bytes": CAP, "write_headroom_bytes": HEADROOM, "home_free_reserve_bytes": RESERVE,
        "software_versions": prior.software_versions()}
    write_new(ROOT/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--stage", action="store_true")
    actions.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Preparation is CPU-only")
    result = stage() if args.stage else freeze()
    print(json.dumps({"status": "staged" if args.stage else "frozen", **storage_guard()}), flush=True)


if __name__ == "__main__":
    main()
