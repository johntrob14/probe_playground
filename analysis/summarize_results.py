"""Portable, CPU-only summary of the report's audited endpoint metrics."""
import argparse
import json
import math
from pathlib import Path

SEEDS = (613, 719, 827, 941)
ARMS = (("task", "Task only"), ("none", "Penalty"), ("cot_only", "Penalty + replay"))


def summary(data):
    lines = ["| Seed | Training | Hacks | Passes | Caps | Probe caught | Passes flagged |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for arm, label in ARMS:
            m = data["endpoints"][f"eval_answer21_{arm}_s{seed}"]["cohorts"]["all"]
            c = m["counts"]
            if set(c) != {"hack", "pass", "cap", "other"} or any(type(v) is not int or v < 0 for v in c.values()):
                raise ValueError("Invalid outcome counts")
            if m["n"] != 480 or m["tasks"] != 120 or sum(c.values()) != m["n"]:
                raise ValueError("Expected 120 tasks and 480 classified outputs per endpoint")
            p = m["probe"]["policy"]["0.5"]
            if p["hack_n"] != c["hack"] or p["pass_n"] != c["pass"]:
                raise ValueError("Incomplete probe coverage")
            for key, denominator in (("caught", c["hack"]), ("false_positives", c["pass"])):
                if type(p[key]) is not int or not 0 <= p[key] <= denominator:
                    raise ValueError("Invalid detection count")
            rate = m["hack_rate"]
            if not math.isfinite(rate) or not math.isclose(rate, c["hack"] / m["n"]):
                raise ValueError("Hack rate does not match counts")
            fp = f'{p["false_positives"]}/{c["pass"]}' if c["pass"] else "Undefined"
            caught = f'{p["caught"]}/{c["hack"]}' if c["hack"] else "Undefined"
            lines.append(f'|{seed}|{label}|{c["hack"]}|{c["pass"]}|{c["cap"]}|{caught}|{fp}|')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Aggregate metrics JSON; no raw transcripts required")
    args = parser.parse_args()
    print(summary(json.loads(args.input.read_text())), end="")


if __name__ == "__main__":
    main()
