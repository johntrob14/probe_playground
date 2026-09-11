"""Generate and score five frozen initial batches, once, before any training.

Uses the coordinated GPU5 HTTP service; never loads a policy or probe locally.
Default is read-only preflight. --generate is explicit, exclusive-new and not
retryable against an existing batch directory. Existing results are preserved.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/matched_baselines_20260906"
SEEDS = (101, 211, 307, 401, 503)
SCORE_PROTOCOL = {"timeout": 6.0, "workers": 8, "length_penalty": .003, "nonterm_task_reward": 0}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def selected_tasks(tasks, allowed_ids):
    if len(allowed_ids) != 300 or len(set(allowed_ids)) != 300:
        raise ValueError("Expected the frozen 300 actor-training task IDs")
    chosen = [task for task in tasks if task.task_id in set(allowed_ids)]
    if len(chosen) != 300 or {task.task_id for task in chosen} != set(allowed_ids):
        raise ValueError("Dataset does not supply each selected training task exactly once")
    # Preserve parquet/loader order, matching rng.sample in the unified trainer.
    return chosen


def batch_tasks_for(seed, tasks):
    if seed not in SEEDS or len(tasks) != 300:
        raise ValueError("Expected a declared seed and the complete actor-training pool")
    return random.Random(seed).sample(tasks, 16)


def validate_prompt_tokens(prompt_ids):
    if not isinstance(prompt_ids, list) or len(prompt_ids) != 16:
        raise ValueError("Expected sixteen initial prompts")
    for tokens in prompt_ids:
        if (not isinstance(tokens, list) or not tokens or len(tokens) + 2048 > 4096
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Initial prompt/token context budget is invalid")


def flatten_outputs(batch_tasks, request, results):
    """Preserve every generated output, including empty/capped/failed answers."""
    validate_prompt_tokens(request["prompt_token_ids"])
    if len(batch_tasks) != 16 or not isinstance(results, list) or len(results) != 16:
        raise ValueError("Generation did not return every initial prompt")
    rows = []
    for task, prompt, result in zip(batch_tasks, request["prompt_token_ids"], results):
        if result.get("prompt_token_ids") != prompt or len(result.get("outputs", [])) != 8:
            raise ValueError("Initial generation task/prompt/sample coverage mismatch")
        for index, output in enumerate(result["outputs"]):
            tokens = output.get("token_ids")
            if (not isinstance(tokens, list) or len(tokens) > 2048
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or not isinstance(output.get("text"), str)
                    or output.get("finish_reason") not in ("stop", "length")):
                raise ValueError("Malformed initial completion or finish reason")
            rows.append({"task_id": task.task_id, "sample_idx": index, "prompt_token_ids": prompt,
                         "completion_token_ids": tokens, "completion": output["text"],
                         "finish_reason": output["finish_reason"]})
    return rows


def build_payload(seed, request, batch_tasks, rows, scores, design_hash, init_hash, server, sources):
    if seed not in SEEDS or len(rows) != 128 or len(scores) != 128:
        raise ValueError("Expected a complete declared initial batch and execution scores")
    if not server or not sources:
        raise ValueError("Missing server/source evidence")
    return {"schema_version": 1, "replication_protocol": "matched_baselines_20260906",
            "created_at": datetime.now(timezone.utc).isoformat(), "design_manifest_sha256": design_hash,
            "seed": seed, "init_adapter_sha256": init_hash, "request": request,
            "task_ids": [task.task_id for task in batch_tasks], "rows": rows,
            "first_scores": scores, "score_protocol": dict(SCORE_PROTOCOL),
            "server_evidence": server, "source_sha256": sources,
            "selection": "All sixteen sampled prompts and all eight outputs each; no outcome selection or retry",
            "scope": "Identical first-update actions and execution scores across four arms; not identical later rollouts"}


def storage_preflight(root, artifact_root):
    home_free = shutil.disk_usage(root).free
    ssd_free = shutil.disk_usage(artifact_root if artifact_root.exists() else artifact_root.parent).free
    if home_free < 30 * 1024**3 + 200 * 1024**2 or ssd_free < 100 * 1024**3:
        raise RuntimeError("Initial-batch storage reserve unavailable; nothing is deleted")
    return {"home_free_bytes": home_free, "ssd_free_bytes": ssd_free, "reserved_batch_bytes": 200 * 1024**2}


def main(generate=False):
    import prepare_matched_baselines as prep
    from prepare_rank_replication import server_evidence
    from run_matched_baselines import gpu0_idle, leases

    root = ROOT.resolve(strict=True)
    if prep.ROOT.resolve() != root or tuple(prep.SEEDS) != SEEDS:
        raise ValueError("Sampler/preparer protocol mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Sampler requires CUDA_VISIBLE_DEVICES='' (HTTP generation only)")
    os.environ.update(HF_HOME="/ssd1/john/.cache/huggingface", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      TESTBED_STORE="/ssd1/john/probe_playground", TESTBED_SYSTEM_SUFFIX="")
    design_path = root / "design_manifest.json"
    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    design_hash = hashlib.sha256(design_bytes).hexdigest()
    prep.validate_design(design, root=root, check_server=True)
    base = prep.base_config()
    init_hash = file_hash(Path(base["init_adapter"]) / "adapter_model.safetensors")
    from transformers import AutoTokenizer
    from testbed.env import load_honeypot, score_many
    from testbed.rollout_client import RolloutClient
    from testbed.sampling import build_prompt_text

    tokenizer = AutoTokenizer.from_pretrained(base["model_id"], local_files_only=True)
    allowed = json.loads(Path(base["train_task_ids"]).read_text())
    pool = selected_tasks(load_honeypot("train"), allowed)
    # Check all train/test prompts, not just the first sampled sixteen.
    expected_test = set(json.loads((prep.OLD / "fresh_test_task_ids.json").read_text()))
    test_pool = [task for task in load_honeypot("test") if task.task_id in expected_test]
    if len(test_pool) != 120 or {task.task_id for task in test_pool} != expected_test:
        raise ValueError("Fresh evaluation task coverage differs from the frozen grid")
    prompt_tokens = {}
    for task in pool + test_pool:
        text = build_prompt_text(tokenizer, task)
        tokens = tokenizer(text, add_special_tokens=False).input_ids
        if not tokens or len(tokens) + 2048 > 4096:
            raise ValueError(f"Full requested completion budget unavailable for task {task.task_id}")
        prompt_tokens[task.task_id] = tokens
    batches = []
    for seed in SEEDS:
        tasks = batch_tasks_for(seed, pool)
        request = prep.expected_request(seed, [prompt_tokens[t.task_id] for t in tasks])
        validate_prompt_tokens(request["prompt_token_ids"])
        batches.append((seed, tasks, request))
    batch_dir = root / "first_batches"
    if batch_dir.exists():
        raise FileExistsError("Initial batches are new-only; inspect existing data, do not repeat sampling")
    print(json.dumps({"event": "initial_batch_preflight", "generate": generate,
                      "n_seeds": len(SEEDS), "outputs_per_seed": 128,
                      "max_prompt_tokens_all_train_and_test": max(map(len, prompt_tokens.values())),
                      **storage_preflight(root, prep.ARTIFACT_ROOT)}), flush=True)
    if not generate:
        return
    with leases(root):
        gpu0_idle()
        prep.validate_design(design, root=root, check_server=True)
        if file_hash(design_path) != design_hash:
            raise RuntimeError("Design changed during preflight")
        batch_dir.mkdir(exist_ok=False)
        client = RolloutClient([8005], timeout=1800)
        client.wait_ready(max_wait=60)
        for seed, tasks, request in batches:
            storage_preflight(root, prep.ARTIFACT_ROOT)
            prep.validate_design(design, root=root, check_server=True)
            if file_hash(design_path) != design_hash:
                raise RuntimeError("Frozen design changed during initial sampling")
            server = server_evidence()
            request_path = batch_dir / f"seed_{seed}.request.json"
            write_new(request_path, {"request": request, "task_ids": [t.task_id for t in tasks],
                                     "design_manifest_sha256": design_hash, "server_evidence": server,
                                     "init_adapter_sha256": init_hash,
                                     "requested_at": datetime.now(timezone.utc).isoformat()})
            print(json.dumps({"event": "initial_batch_requested", "seed": seed,
                              "request_file": str(request_path)}), flush=True)
            rows = flatten_outputs(tasks, request, client.generate(**request))
            task_map = {task.task_id: task for task in tasks}
            scores = score_many([task_map[row["task_id"]] for row in rows],
                                [row["completion"] for row in rows], workers=8, timeout=6.0, length_penalty=.003)
            payload = build_payload(seed, request, tasks, rows, [dataclasses.asdict(s) for s in scores],
                                    design_hash, init_hash, server, prep.source_hashes(root=root))
            prep.validate_first_batch(payload, seed, design, design_hash)
            prep.validate_design(design, root=root, check_server=True)
            destination = batch_dir / f"seed_{seed}.json"
            write_new(destination, payload)
            print(json.dumps({"event": "initial_batch_complete", "seed": seed, "n_outputs": len(rows),
                              "file": str(destination), "sha256": file_hash(destination)}), flush=True)
        print(json.dumps({"event": "all_initial_batches_complete", "seeds": list(SEEDS)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    main(parser.parse_args().generate)
