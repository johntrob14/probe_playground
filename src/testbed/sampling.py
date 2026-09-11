"""vLLM rollout sampling for the base model or a LoRA checkpoint.

Data-parallel over GPUs: the driver shards tasks and launches one worker process per GPU.
Each rollout record carries the prompt token ids, completion token ids and text, so activation
capture and scoring can be done later without re-tokenising or re-generating.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from testbed.env import Task

MODEL_ID = "Qwen/Qwen3-8B"


def build_prompt_text(tokenizer, task: Task, enable_thinking: bool = True) -> str:
    return tokenizer.apply_chat_template(task.messages(), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=enable_thinking)


def worker(task_file: str, out_file: str, n: int, max_tokens: int, temperature: float, top_p: float,
           seed: int, lora_path: str | None, model_id: str, gpu_mem: float) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tasks = [Task.from_dict(json.loads(l)) for l in open(task_file)]
    tok = AutoTokenizer.from_pretrained(model_id)
    prompts = [build_prompt_text(tok, t) for t in tasks]
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=gpu_mem, max_model_len=max_tokens + 1024,
              enable_lora=lora_path is not None, max_lora_rank=64, seed=seed, enable_prefix_caching=True)
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    lora = LoRARequest("policy", 1, lora_path) if lora_path else None
    outs = llm.generate(prompts, sp, lora_request=lora)
    with open(out_file, "w") as f:
        for t, p, o in zip(tasks, prompts, outs):
            for k, c in enumerate(o.outputs):
                rec = {
                    "task_id": t.task_id, "family": t.family, "split": t.split, "sample_idx": k,
                    "prompt_text": p, "prompt_token_ids": list(o.prompt_token_ids),
                    "completion": c.text, "completion_token_ids": list(c.token_ids),
                    "finish_reason": c.finish_reason, "n_completion_tokens": len(c.token_ids),
                    "model": model_id, "lora": lora_path,
                }
                f.write(json.dumps(rec) + "\n")


def sample(tasks: list[Task], out_dir: Path, gpus: list[int], n: int = 8, max_tokens: int = 8192,
           temperature: float = 1.0, top_p: float = 1.0, seed: int = 0, lora_path: str | None = None,
           model_id: str = MODEL_ID, gpu_mem: float = 0.85, tag: str = "rollouts") -> Path:
    """Shard ``tasks`` across ``gpus`` and write ``out_dir/{tag}.jsonl``."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    shards = [tasks[i::len(gpus)] for i in range(len(gpus))]
    procs = []
    for g, shard in zip(gpus, shards):
        if not shard:
            continue
        tf = out_dir / f"_tasks_{tag}_{g}.jsonl"
        of = out_dir / f"_{tag}_shard{g}.jsonl"
        with open(tf, "w") as f:
            for t in shard:
                f.write(json.dumps(t.to_dict()) + "\n")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                   VLLM_LOGGING_LEVEL="WARNING", TOKENIZERS_PARALLELISM="false")
        cmd = [sys.executable, "-m", "testbed.sampling", str(tf), str(of), str(n), str(max_tokens), str(temperature),
               str(top_p), str(seed + g), lora_path or "", model_id, str(gpu_mem)]
        log = open(out_dir / f"_{tag}_shard{g}.log", "w")
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
    fail = [g for g, p in procs if p.wait() != 0]
    if fail:
        raise RuntimeError(f"sampling shards failed on gpus {fail}; see logs in {out_dir}")
    merged = out_dir / f"{tag}.jsonl"
    with open(merged, "w") as fo:
        for g, _ in procs:
            with open(out_dir / f"_{tag}_shard{g}.jsonl") as fi:
                fo.write(fi.read())
    return merged


if __name__ == "__main__":
    a = sys.argv[1:]
    worker(a[0], a[1], int(a[2]), int(a[3]), float(a[4]), float(a[5]), int(a[6]), a[7] or None, a[8], float(a[9]))
