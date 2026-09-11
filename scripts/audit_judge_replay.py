"""Post-training CPU execution audit; none/judge training never consumes this output."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import prepare_judge_replay as prep


def read_rows(path):
    result = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                raise ValueError(f"Blank record in {path}")
            result.append(json.loads(line))
    return result


def identity(row):
    values = tuple(row.get(k) for k in ("step", "task_id", "sample_idx"))
    if any(type(v) is not int for v in values):
        raise ValueError("Audit row missing integer step/task/sample identity")
    return values


def validate_training_rows(rows, cfg):
    expected = cfg["steps"]*cfg["batch_prompts"]*cfg["group_size"]
    if len(rows) != expected or len({identity(r) for r in rows}) != expected:
        raise ValueError("Incomplete or duplicate training rollout grid")
    allowed = set(prep.read_json(cfg["train_task_ids"]))
    for step in range(1, cfg["steps"]+1):
        group = [r for r in rows if r["step"] == step]
        counts = Counter(r["task_id"] for r in group)
        if len(counts) != cfg["batch_prompts"] or set(counts.values()) != {cfg["group_size"]}:
            raise ValueError("Training step lacks its complete task groups")
        for task_id in counts:
            if task_id not in allowed or {r["sample_idx"] for r in group if r["task_id"] == task_id} != set(range(cfg["group_size"])):
                raise ValueError("Training step has invalid tasks/samples")


def validate_candidates(candidates, by_key, cfg):
    seen = set()
    for row in candidates:
        key = identity(row)
        if key in seen or key not in by_key or key[0] % cfg["refresh_every"]:
            raise ValueError("Duplicate, unknown or off-schedule refresh candidate")
        for field in ("prompt_token_ids", "completion_token_ids", "completion", "finish_reason"):
            if field in by_key[key] and row.get(field) != by_key[key][field]:
                raise ValueError("Refresh candidate text/tokens differ from its raw training row")
        seen.add(key)
    counts = Counter(r["step"] for r in candidates)
    expected_steps = set(range(cfg["refresh_every"], cfg["steps"]+1, cfg["refresh_every"]))
    if set(counts) != expected_steps or any(not cfg["refresh_random"] <= n <= cfg["refresh_random"]+cfg["refresh_low_score"] for n in counts.values()):
        raise ValueError("Refresh candidate schedule or budget differs")


def write_jsonl_new(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")


def audit(run_dir, config_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Execution audit is CPU-only")
    cfg = prep.read_json(config_path)
    run_dir = Path(run_dir).resolve()
    if run_dir != Path(cfg["run_dir"]).resolve() or run_dir.parent != prep.ARTIFACT_ROOT:
        raise ValueError("Audit target differs from the declared owned run")
    complete = prep.read_json(run_dir/"complete.json")
    if complete.get("steps") != cfg["steps"] or prep.read_json(run_dir/"config.json") != cfg:
        raise ValueError("Training has not completed with the declared configuration")
    outputs = [run_dir/name for name in ("execution_audit.jsonl", "refresh_execution_audit.jsonl", "refresh_quality.json", "audit_complete.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Audit is exclusive-new; preserve and inspect any partial output")
    prep.storage_guard(headroom=prep.WRITE_HEADROOM)
    raw_path, candidates_path = run_dir/"rollouts.jsonl", run_dir/"replay_candidates.jsonl"
    if not candidates_path.is_file():
        raise ValueError("Completed training must retain its refresh candidate log")
    source = {"rollouts": prep.digest(raw_path), "replay_candidates": prep.digest(candidates_path)}
    rows = read_rows(raw_path)
    validate_training_rows(rows, cfg)
    by_key = {identity(r): r for r in rows}
    candidates = read_rows(candidates_path)
    validate_candidates(candidates, by_key, cfg)
    from testbed.env import load_honeypot, score_many
    sys.path.insert(0, str(prep.ROOT))
    from judge_labels import execution_class, JudgeLabel, pseudo_label_quality
    tasks = {t.task_id: t for t in load_honeypot("train")}
    scored_rows = []
    for start in range(0, len(rows), 128):
        chunk = rows[start:start+128]
        scores = score_many([tasks[r["task_id"]] for r in chunk], [r["completion"] for r in chunk],
                            workers=8, timeout=6., length_penalty=cfg["length_penalty"])
        if len(scores) != len(chunk):
            raise RuntimeError("Execution audit response count mismatch")
        for row,score in zip(chunk,scores):
            item = {"step": row["step"], "task_id": row["task_id"], "sample_idx": row["sample_idx"],
                    "finish_reason": row["finish_reason"], **score.to_dict()}
            item["execution_class"] = execution_class(item)
            scored_rows.append(item)
        print(json.dumps({"event": "execution_audit_progress", "rows": len(scored_rows), "total": len(rows)}), flush=True)
    scored_by_key = {identity(r):r for r in scored_rows}
    refresh_rows = [{**scored_by_key[identity(c)], "candidate_key": c.get("key", f"refresh|{c['step']}|{c['task_id']}|{c['sample_idx']}"),
                     "pseudo_label": c.get("pseudo_label", c.get("replay_label")),
                     "judge_score": c.get("judge_score"), "judge_decision": c.get("judge_decision"),
                     "reason": c.get("reason", c.get("eligibility_reason", "not_labelled"))} for c in candidates]
    quality = []
    for step in range(cfg["refresh_every"], cfg["steps"]+1, cfg["refresh_every"]):
        subset = [r for r in refresh_rows if r["step"] == step]
        labels = [JudgeLabel(r["pseudo_label"], r["judge_score"], r["judge_decision"], r["reason"]) for r in subset]
        stats = pseudo_label_quality(labels, [r["execution_class"] for r in subset])
        hack_scores = [r["judge_score"] for r in subset if r["execution_class"] == "strict_hack"
                       and r["judge_score"] is not None]
        quality.append({"step": step, "affects_final_weights": step < cfg["steps"], **stats,
                        "n_judge_scored_strict_hack_candidates": len(hack_scores),
                        "mean_judge_score_strict_hack_candidates": sum(hack_scores)/len(hack_scores) if hack_scores else None})
    trajectory = []
    for step in range(1,cfg["steps"]+1):
        subset = [r for r in scored_rows if r["step"] == step]
        trajectory.append({"step": step, "n": len(subset), "strict_hacks": sum(r["hack"] for r in subset),
                           "honest_passes": sum(r["honest_pass"] for r in subset),
                           "terminating_hacks": sum(r["hack"] and r["finish_reason"] != "length" for r in subset),
                           "terminating_passes": sum(r["honest_pass"] and r["finish_reason"] != "length" for r in subset),
                           "token_limited": sum(r["finish_reason"] == "length" for r in subset)})
    if prep.digest(raw_path) != source["rollouts"] or (candidates_path.exists() and prep.digest(candidates_path) != source["replay_candidates"]):
        raise RuntimeError("Training sources changed during posthoc audit")
    write_jsonl_new(outputs[0],scored_rows)
    write_jsonl_new(outputs[1],refresh_rows)
    prep.write_new(outputs[2], {"schema_version": 1, "run": run_dir.name, "replay_label_source": cfg["replay_label_source"],
                              "scope": "Post-training diagnostic; not read by the trainer or used for stopping", "refreshes": quality,
                              "training_trajectory": trajectory})
    prep.write_new(outputs[3], {"schema_version": 1, "n_rollouts": len(rows), "n_steps": cfg["steps"],
                              "n_refresh_candidates": len(candidates), "source_sha256": source,
                              "output_sha256": {p.name:prep.digest(p) for p in outputs[:3]},
                              "completed_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir,args.config)
