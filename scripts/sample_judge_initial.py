"""Generate exactly one public-scored first batch per frozen seed via GPU5 HTTP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

import prepare_judge_replay as prep
from sample_matched_initial import flatten_outputs, validate_prompt_tokens


def public_task_objects(records):
    from testbed.env import Task
    required = {"task_id", "text", "setup", "shown_tests"}
    if len(records) != 300 or len({r["task_id"] for r in records}) != 300:
        raise ValueError("Expected the complete unique actor population")
    result = []
    for row in records:
        if set(row) != required or len(row["shown_tests"]) != 1:
            raise ValueError("Public task schema must not contain hidden-test/reference-code fields")
        result.append(Task(task_id=row["task_id"], family="honeypot", split="train", text=row["text"],
                           signature="", entry_point="", setup=row["setup"], shown_tests=row["shown_tests"],
                           original_tests=[], mutated_index=None, mutated_original=None, mutated_replaced=None))
    return result


def main(generate=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Sampler is CPU-only; generation uses the existing GPU5 service")
    if os.environ.get("TESTBED_SYSTEM_SUFFIX", ""):
        raise ValueError("Unexpected ambient system suffix")
    os.environ.update(HF_HOME="/ssd1/john/.cache/huggingface", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TESTBED_STORE=str(prep.STORE), TOKENIZERS_PARALLELISM="false")
    design_path = prep.ROOT/"design_manifest.json"
    design = prep.validate_design(prep.read_json(design_path))
    design_hash = prep.digest(design_path)
    base = prep.base_config()
    from transformers import AutoTokenizer
    from testbed.sampling import build_prompt_text
    from testbed.rollout_client import RolloutClient
    from train_judge_replay import score_public_many, validate_first_batch
    import run_judge_replay as runner

    tokenizer = AutoTokenizer.from_pretrained(base["model_id"], local_files_only=True)
    tasks = public_task_objects(prep.read_json(prep.ROOT/"public_tasks.json"))
    prompt_ids = {t.task_id: tokenizer(build_prompt_text(tokenizer, t), add_special_tokens=False).input_ids for t in tasks}
    for task_id,tokens in prompt_ids.items():
        if not tokens or len(tokens)+2048 > 4096:
            raise ValueError(f"Insufficient full completion budget for actor task {task_id}")
    plans = []
    for seed in prep.SEEDS:
        selected = random.Random(seed).sample(tasks, 16)
        request = prep.expected_request(seed, [prompt_ids[t.task_id] for t in selected])
        validate_prompt_tokens(request["prompt_token_ids"])
        plans.append((seed, selected, request))
    directory = prep.ROOT/"first_batches"
    if directory.exists():
        raise FileExistsError("First batches are exclusive-new; preserve existing requests/outputs")
    prep.storage_guard(headroom=prep.WRITE_HEADROOM)
    print(json.dumps({"event": "first_batch_preflight", "generate": generate, "seeds": list(prep.SEEDS),
                      "outputs_per_seed": 128, "max_actor_prompt_tokens": max(map(len,prompt_ids.values()))}), flush=True)
    if not generate:
        return
    with runner.leases(prep.ROOT):
        runner.COMMON.gpu0_idle()
        prep.validate_design(design)
        if prep.digest(design_path) != design_hash:
            raise RuntimeError("Design changed during preflight")
        directory.mkdir(exist_ok=False)
        client = RolloutClient([8005], timeout=1800)
        client.wait_ready(max_wait=60)
        for seed,selected,request in plans:
            prep.storage_guard(headroom=prep.WRITE_HEADROOM)
            servers = prep.pinned_servers({"rollout": design["rollout_server"], "judge": design["judge_server"]})
            prep.write_new(directory/f"seed_{seed}.request.json", {"seed": seed, "request": request,
                           "task_ids": [t.task_id for t in selected], "design_manifest_sha256": design_hash,
                           "server_evidence": servers["rollout"], "requested_at": datetime.now(timezone.utc).isoformat()})
            print(json.dumps({"event": "first_batch_requested", "seed": seed}), flush=True)
            rows = flatten_outputs(selected, request, client.generate(**request))
            # Preserve raw generations before any execution; never silently resample an interrupted request.
            prep.write_new(directory/f"seed_{seed}.raw.json", rows)
            lookup = {t.task_id: t for t in selected}
            scores = score_public_many([lookup[r["task_id"]] for r in rows], [r["completion"] for r in rows],
                                       workers=8, timeout=6., length_penalty=.003)
            payload = {"schema_version": 2, "replication_protocol": prep.PROTOCOL,
                       "seed": seed, "created_at": datetime.now(timezone.utc).isoformat(),
                       "design_manifest_sha256": design_hash,
                       "init_adapter_sha256": prep.digest(Path(base["init_adapter"])/"adapter_model.safetensors"),
                       "request": request, "task_ids": [t.task_id for t in selected], "rows": rows,
                       "first_scores": scores, "public_score_protocol": prep.PUBLIC_SCORE_PROTOCOL,
                       "server_evidence": servers["rollout"], "source_sha256": design["source_sha256"],
                       "scope": "All 16x8 raw actions and public-only scores shared within seed; no ground-truth labels"}
            cfg = prep.expected_config(prep.ARMS[0], seed, "0"*64, design)
            validate_first_batch(payload, cfg, payload["task_ids"], request["prompt_token_ids"])
            prep.validate_design(design)
            if prep.digest(design_path) != design_hash:
                raise RuntimeError("Design changed during sampling")
            destination = directory/f"seed_{seed}.json"
            prep.write_new(destination, payload)
            print(json.dumps({"event": "first_batch_complete", "seed": seed, "n_outputs": len(rows),
                              "sha256": prep.digest(destination)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    main(parser.parse_args().generate)
