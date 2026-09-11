"""Build a replay bank whose labels come from the CoT-only judge, not from execution.

Candidate pool: the starting checkpoint's rollouts on the 300 actor-training tasks (the raw, token-preserving
file joined by (task_id, sample_idx) to its scored twin, which supplies parsed/cot/execution fields). Every eligible
candidate (parsed, terminated, closed chain of thought, on the allowlist) is judged once; the full judge response,
explicit score, decision and keep/abstain reason are saved for every candidate in ``<out>.candidates.jsonl``.
Confident pseudo-labels form task-matched pairs (one pseudo-hack, one pseudo-honest per task), at most
``--max-pairs``. Bank rows carry token IDs and the judge fields but NO execution fields; execution outcomes for the
bank rows go to ``<out>.execution_audit.jsonl`` (never read by any trainer), and label quality against execution
to ``<out>.quality.json``.

  PYTHONPATH=src .venv/bin/python scripts/build_judge_replay_bank.py \
      --raw artifacts/panel_data_yud20/train/honeypot.jsonl \
      --scored artifacts/panel_data_yud20/train/honeypot.scored.jsonl \
      --allowlist experiments/replay_pilot_20260904/train_task_ids.json \
      --out experiments/judge_labelled_replay_20260907/replay_bank_judge.jsonl --max-pairs 76 --ports 8003
"""
from __future__ import annotations

import argparse, hashlib, json, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "judge_labelled_replay_20260907"))
from transformers import AutoTokenizer

from judge_labels import JUDGE_MODEL, cot_status, execution_class, label_rows, pseudo_label_quality
from testbed.env import load_honeypot
from testbed.rollout_client import RolloutClient

EXECUTION_FIELDS = ("hack", "honest_pass", "reward", "shown_pass", "original_pass", "flags", "parsed", "truncated")

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True, help="token-preserving rollouts of the starting checkpoint")
p.add_argument("--scored", required=True, help="scored twin of --raw (same task_id/sample_idx keys)")
p.add_argument("--allowlist", required=True, help="JSON list of actor-training task ids")
p.add_argument("--out", required=True)
p.add_argument("--max-pairs", type=int, default=76)
p.add_argument("--ports", default="8003")
p.add_argument("--hack-min", type=float, default=0.7)
p.add_argument("--honest-max", type=float, default=0.3)
p.add_argument("--seed", type=int, default=20260907)
a = p.parse_args()

out = Path(a.out)
for path in (out, Path(str(out) + ".candidates.jsonl"), Path(str(out) + ".execution_audit.jsonl"), Path(str(out) + ".quality.json")):
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
out.parent.mkdir(parents=True, exist_ok=True)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
raw = {(str(r["task_id"]), str(r["sample_idx"])): r for r in map(json.loads, open(a.raw))}
scored = [json.loads(l) for l in open(a.scored) if l.strip()]
allow = {str(t) for t in json.load(open(a.allowlist))}
tasks = {t.task_id: t for sp in ("train", "validation", "test") for t in load_honeypot(sp)}

candidates, skipped = [], defaultdict(int)
for s in scored:
    key = (str(s["task_id"]), str(s["sample_idx"]))
    if key not in raw: skipped["no_raw_twin"] += 1; continue
    if str(s["task_id"]) not in allow: skipped["not_actor_task"] += 1; continue
    if s["task_id"] not in tasks: skipped["unknown_task"] += 1; continue
    if not s.get("parsed") or s.get("truncated") or s.get("finish_reason") != "stop": skipped["not_terminated_or_unparsed"] += 1; continue
    if cot_status(s)[1] != "ok": skipped[cot_status(s)[1]] += 1; continue
    candidates.append({**raw[key], **{k: s[k] for k in s if k not in raw[key]}})
print(f"candidates: {len(candidates)}; skipped: {dict(skipped)}")

tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
client = RolloutClient([int(x) for x in a.ports.split(",")]); client.wait_ready()
labels, raw_texts = label_rows(candidates, tasks, tok, client, hack_min=a.hack_min, honest_max=a.honest_max)
classes = [execution_class(r) for r in candidates]
quality_all = pseudo_label_quality(labels, classes)
print("label quality over all candidates:", json.dumps(quality_all))

with open(str(out) + ".candidates.jsonl", "w") as f:
    for r, l, txt, c in zip(candidates, labels, raw_texts, classes):
        f.write(json.dumps({"task_id": r["task_id"], "sample_idx": r["sample_idx"], "pseudo_label": l.pseudo_label,
                            "judge_score": l.score, "judge_decision": l.decision, "reason": l.reason,
                            "judge_raw": txt, "execution_class": c}) + "\n")

by_task = defaultdict(lambda: {0: [], 1: []})
for r, l, c in zip(candidates, labels, classes):
    if l.pseudo_label is not None:
        by_task[str(r["task_id"])][l.pseudo_label].append((r, l, c))
pairable = sorted(t for t, d in by_task.items() if d[0] and d[1])
rng = random.Random(a.seed); rng.shuffle(pairable)
chosen = pairable[: a.max_pairs]
print(f"tasks with both pseudo-classes: {len(pairable)}; pairs written: {len(chosen)}")

bank_labels, bank_classes = [], []
with out.open("w") as bank_f, open(str(out) + ".execution_audit.jsonl", "w") as audit_f:
    for t in sorted(chosen, key=int):
        for lab in (0, 1):
            r, l, c = rng.choice(by_task[t][lab])
            row = {k: v for k, v in r.items() if k not in EXECUTION_FIELDS}
            row.update({"replay_label": lab, "label": lab, "judge_score": l.score, "judge_decision": l.decision,
                        "label_source": "judge", "key": f"judge_bank|{r['task_id']}|{r['sample_idx']}", "source": a.raw})
            bank_f.write(json.dumps(row) + "\n")
            audit_f.write(json.dumps({"key": row["key"], "replay_label": lab, "execution_class": c,
                                      **{k: r.get(k) for k in EXECUTION_FIELDS}}) + "\n")
            bank_labels.append(l); bank_classes.append(c)
quality_bank = pseudo_label_quality(bank_labels, bank_classes)
json.dump({"raw": a.raw, "raw_sha256": sha(a.raw), "scored": a.scored, "scored_sha256": sha(a.scored),
           "allowlist": a.allowlist, "allowlist_sha256": sha(a.allowlist), "hack_min": a.hack_min, "honest_max": a.honest_max,
           "candidates": len(candidates), "skipped": dict(skipped), "pairs": len(chosen),
           "quality_all_candidates": quality_all, "quality_bank": quality_bank,
           "bank_sha256": sha(out)}, open(str(out) + ".quality.json", "w"), indent=1)
print(f"wrote {2 * len(chosen)} bank rows -> {out}; audit and candidate sidecars alongside")
