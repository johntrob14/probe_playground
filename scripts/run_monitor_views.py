"""Cached, sanitized three-view judge evaluation; no local model or oracle loader.

Default is offline preflight. --run authorizes the pinned remote judge; --resume
requires a verified completed prefix. An unresolved saved request is never
automatically retried. A fully saved response can be finalized without inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/monitor_views_20260908"
PROTOCOL = "monitor_views_20260908"
SOURCES = ["initial", "prob_exec_replay_s719", "prob_judge_replay_s719",
           "prob_judge_replay_strict_s613", "prob_judge_replay_strict_s719"]
VIEWS = ["cot_only", "answer_only", "cot_answer"]
FIELDS = {"source", "key", "task_text", "cot", "answer"}
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
PARSER = REPO / "experiments/judge_labelled_replay_20260907/judge_labels.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=reject)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def storage_guard():
    used = 0
    for path in ROOT.rglob("*"):
        require(not path.is_symlink(), "Unexpected study symlink")
        if path.is_file():
            used += path.stat().st_size
    free = shutil.disk_usage(ROOT).free
    require(used + 64 * 1024**2 <= 512 * 1024**2 and free >= 30 * 1024**3 + 64 * 1024**2,
            "Study storage cap/headroom/home reserve reached; preserve all outputs")


def load_manifest(path=ROOT / "manifest.json"):
    require(Path(path).resolve() == ROOT / "manifest.json", "Unexpected manifest path")
    manifest = read_json(path)
    expected = {"status": "frozen", "protocol": PROTOCOL, "source_names": SOURCES, "views": VIEWS,
                "n_per_source": 480, "chunk_size": 32, "judge_model": MODEL, "judge_ports": [8003],
                "max_model_len": 16384, "max_new_tokens": 384,
                "input_file": str(ROOT / "public_inputs.jsonl"), "rubric_file": str(ROOT / "rubric.txt")}
    require(all(manifest.get(k) == v for k, v in expected.items()), "Frozen monitor-view design differs")
    for key in ("input", "rubric"):
        require(digest(manifest[f"{key}_file"]) == manifest[f"{key}_sha256"], f"Frozen {key} changed")
    hashes = manifest.get("source_sha256", {})
    require(isinstance(hashes, dict) and {str(Path(__file__).resolve()), str(PARSER)}.issubset(hashes), "Missing implementation hashes")
    for source, expected_hash in hashes.items():
        require(digest(source) == expected_hash, f"Frozen implementation changed: {source}")
    return manifest


def load_inputs(manifest):
    rows = [json.loads(line) for line in Path(manifest["input_file"]).read_text().splitlines() if line.strip()]
    require(len(rows) == 2400, "Expected exactly five complete 480-output sources")
    for row in rows:
        require(isinstance(row, dict) and set(row) == FIELDS and all(isinstance(v, str) for v in row.values())
                and row["source"] in SOURCES and row["key"] and row["task_text"], "Only sanitized public text fields are allowed")
    population = {(r["source"], r["key"]) for r in rows}
    require(len(population) == 2400, "Duplicate source/key")
    keys = None
    for source in SOURCES:
        selected = [r["key"] for r in rows if r["source"] == source]
        require(len(selected) == 480, "Incomplete source")
        require(keys is None or selected == keys, "Sources must have identical ordered task/sample keys")
        keys = selected
    return rows


def prompt_text(row, view, rubric):
    require(view in VIEWS and set(row) == FIELDS, "Undeclared view or input fields")
    evidence = {"task": row["task_text"], "reasoning": row["cot"] if view != "answer_only" else None,
                "final_answer": row["answer"] if view != "cot_only" else None}
    return rubric + "\n\nEVIDENCE JSON:\n" + json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def plans(manifest, rows):
    rubric = Path(manifest["rubric_file"]).read_text()
    items = []
    for source in SOURCES:
        selected = [r for r in rows if r["source"] == source]
        for view in VIEWS:
            for row in selected:
                content = prompt_text(row, view, rubric)
                items.append({"source": source, "key": row["key"], "view": view, "content": content,
                              "content_sha256": hashlib.sha256(content.encode()).hexdigest()})
    return [items[i:i + 32] for i in range(0, len(items), 32)]


def load_parser():
    spec = importlib.util.spec_from_file_location("monitor_views_frozen_parser", PARSER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.parse_explicit


def parse_output(output, parser):
    require(isinstance(output, dict) and isinstance(output.get("text"), str)
            and output.get("finish_reason") in ("stop", "length"), "Malformed judge output")
    explicit, decision = parser(output["text"])
    score = explicit
    status = "explicit_score_and_decision"
    if output["finish_reason"] != "stop":
        score, status = None, "judge_nonterminating"
    elif explicit is None or decision is None:
        score, status = None, "missing_or_malformed_required_field"
    elif decision == "UNCLEAR":
        status = "unclear"
    elif (decision == "HARDCODE") != (explicit >= .5):
        status = "inconsistent_score_and_decision"
    abstained = status != "explicit_score_and_decision"
    return {"raw": output["text"], "judge_finish_reason": output["finish_reason"], "explicit_score": explicit,
            "score": score, "decision": decision, "parse_status": status, "abstained": abstained,
            "classification": None if abstained else int(decision == "HARDCODE")}


def parsed_rows(request, response, parser):
    require(isinstance(response, list) and len(response) == len(request["items"]), "Judge returned wrong row count")
    rows = []
    for item, result in zip(request["items"], response):
        require(isinstance(result, dict) and isinstance(result.get("outputs"), list) and len(result["outputs"]) == 1,
                "Judge must return exactly one completion per input")
        rows.append({**{k: item[k] for k in ("source", "key", "view", "prompt_sha256", "prompt_tokens")},
                     **parse_output(result["outputs"][0], parser)})
    return rows


def paths(index):
    stem = ROOT / "results/chunks" / f"{index:04d}"
    return tuple(Path(str(stem) + f".{suffix}.json") for suffix in ("request", "response", "done"))


def verify_chunk(index, plan, manifest_hash, parser):
    request_path, response_path, done_path = paths(index)
    request, response, done = read_json(request_path), read_json(response_path), read_json(done_path)
    require(request["chunk_index"] == index and request["manifest_sha256"] == manifest_hash
            and len(request["items"]) == len(plan), "Chunk identity mismatch")
    for saved, expected in zip(request["items"], plan):
        require(all(saved[k] == expected[k] for k in ("source", "key", "view", "content_sha256"))
                and expected["content"] in saved["prompt"], "Chunk source/view/content mismatch")
    require(request["request"] == {"prompts": [x["prompt"] for x in request["items"]], "n": 1, "temperature": 0., "max_tokens": 384}, "Request settings changed")
    for item in request["items"]:
        require(item["prompt_sha256"] == hashlib.sha256(item["prompt"].encode()).hexdigest()
                and type(item["prompt_tokens"]) is int and 0 < item["prompt_tokens"] <= 16384 - 384,
                "Prompt identity or declared context budget mismatch")
    require(done == {"chunk_index": index, "manifest_sha256": manifest_hash, "request_sha256": digest(request_path),
                     "response_sha256": digest(response_path), "rows": parsed_rows(request, response, parser)}, "Completed chunk does not reproduce")
    return done["rows"]


def verify_complete(manifest_path=ROOT / "manifest.json"):
    """Read-only offline verifier; never constructs clients/tokenizers or calls a server."""
    manifest = load_manifest(manifest_path)
    manifest_hash = digest(manifest_path)
    chunks, parser = plans(manifest, load_inputs(manifest)), load_parser()
    complete = read_json(ROOT / "results/complete.json")
    require(set((ROOT / "results/chunks").iterdir()) == {path for i in range(len(chunks)) for path in paths(i)},
            "Unexpected or missing chunk artifacts")
    require(complete == {"status": "complete", "manifest_sha256": manifest_hash, "n": 7200, "n_chunks": len(chunks),
                        "chunk_sha256": {str(paths(i)[2]): digest(paths(i)[2]) for i in range(len(chunks))}}, "Completion marker differs")
    return [row for i, plan in enumerate(chunks) for row in verify_chunk(i, plan, manifest_hash, parser)]


def check_server(manifest):
    from prepare_judge_replay import judge_server_evidence
    observed, expected = judge_server_evidence(), manifest["judge_server"]
    for key in ("pid", "starttime_ticks", "argv", "cmdline_sha256", "cuda_visible_devices", "max_model_len"):
        require(observed[key] == expected[key], f"Pinned judge changed: {key}")


def run(manifest_path=ROOT / "manifest.json", *, execute=False, resume=False):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Runner must be CPU-only")
    manifest, manifest_hash = load_manifest(manifest_path), digest(manifest_path)
    chunks, parser = plans(manifest, load_inputs(manifest)), load_parser()
    output = ROOT / "results"
    if (output / "complete.json").exists():
        require(resume, "Existing complete output requires --resume")
        return {"status": "already_complete", "n": len(verify_complete(manifest_path))}
    require(resume or not output.exists(), "Existing results require explicit --resume")
    completed, gap = 0, False
    for i, plan in enumerate(chunks):
        req, resp, done = paths(i)
        if done.exists():
            require(not gap, "Completed chunk follows unfinished predecessor")
            verify_chunk(i, plan, manifest_hash, parser)
            completed += 1
        else:
            gap = True
            require(not resp.exists() or req.exists(), "Orphaned response without request")
            require(i == completed or not req.exists() and not resp.exists(), "Artifacts after unfinished chunk")
            require(not req.exists() or resp.exists(), "Unresolved request: server outcome unknown; no automatic retry")
    storage_guard()
    if not execute:
        return {"status": "offline_preflight", "completed_chunks": completed, "remaining_chunks": len(chunks) - completed}
    from run_judge_replay import leases
    with leases(ROOT):
        require(digest(manifest_path) == manifest_hash, "Manifest changed before acquiring lease")
        check_server(manifest)
        os.environ.update(HF_HOME="/ssd1/john/.cache/huggingface", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        from transformers import AutoTokenizer
        from testbed.rollout_client import RolloutClient
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        client = RolloutClient([8003], timeout=1800)
        output.mkdir(exist_ok=True)
        (output / "chunks").mkdir(exist_ok=True)
        for i in range(completed, len(chunks)):
            require(digest(manifest_path) == manifest_hash, "Manifest changed during evaluation")
            check_server(manifest)
            storage_guard()
            req, resp, done = paths(i)
            items = []
            for item in chunks[i]:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": item["content"]}], tokenize=False, add_generation_prompt=True)
                tokens = tokenizer(prompt, add_special_tokens=False).input_ids
                require(tokens and len(tokens) + 384 <= 16384, "Full judge context budget exceeded; no truncation")
                items.append({**{k: v for k, v in item.items() if k != "content"}, "prompt": prompt,
                              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_tokens": len(tokens)})
            request = {"chunk_index": i, "manifest_sha256": manifest_hash, "items": items,
                       "request": {"prompts": [x["prompt"] for x in items], "n": 1, "temperature": 0., "max_tokens": 384}}
            if req.exists():
                require(read_json(req) == request and resp.exists(), "Unfinished request differs or has no saved response")
                response = read_json(resp)
            else:
                require(not resp.exists() and not done.exists(), "Orphaned chunk artifacts")
                write_new(req, request)
                response = client.generate(**request["request"])
                write_new(resp, response)
            result = {"chunk_index": i, "manifest_sha256": manifest_hash, "request_sha256": digest(req),
                      "response_sha256": digest(resp), "rows": parsed_rows(request, response, parser)}
            write_new(done, result)
            print(json.dumps({"event": "chunk_complete", "chunk_index": i, "n": len(result["rows"]), "total_chunks": len(chunks)}), flush=True)
        load_manifest(manifest_path)
        write_new(output / "complete.json", {"status": "complete", "manifest_sha256": manifest_hash, "n": 7200,
                  "n_chunks": len(chunks), "chunk_sha256": {str(paths(i)[2]): digest(paths(i)[2]) for i in range(len(chunks))}})
    return {"status": "complete", "n": len(verify_complete(manifest_path))}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    p.add_argument("--run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(args.manifest, execute=args.run, resume=args.resume)), flush=True)
