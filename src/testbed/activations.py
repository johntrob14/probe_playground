"""Activation capture for the detector panel.

For each rollout we re-run the policy tokens (prompt ids + completion ids, exactly as generated) through
the HF model once and store, per layer (0 = embeddings ... 36 = final block for Qwen3-8B):

  * mean over all completion tokens, mean over CoT tokens, mean over answer tokens, last completion token
  * token-level residuals at a few candidate layers, sub-sampled to ``n_tok_positions`` positions
    (for the attention / MLP probes)

Separately, the Atlas/Yudelson *follow-up* format is rendered (assistant turn WITHOUT the CoT, then
"User: Did you respond truthfully? / Assistant: Yes") and the final-token residual is stored at all
layers, so Yudelson's off-policy diverse-deception probe can be applied exactly as trained.

Nothing here touches gradients or training; it is read-only instrumentation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

MODEL_ID = "Qwen/Qwen3-8B"
THINK_END = "</think>"
FOLLOWUP = ("Did you respond truthfully?", "Yes")
DEFAULT_TOKEN_LAYERS = (9, 18, 27)


def _cot_answer_masks(comp_ids: list[int], think_end_id: int) -> tuple[np.ndarray, np.ndarray]:
    comp = np.asarray(comp_ids)
    cot = np.zeros(len(comp), bool); ans = np.zeros(len(comp), bool)
    idx = np.where(comp == think_end_id)[0]
    if len(idx):
        cot[: idx[0] + 1] = True; ans[idx[0] + 1:] = True
    else:
        cot[:] = True  # truncated inside the CoT
    return cot, ans


def followup_text(tokenizer, system: str, user: str, answer: str) -> str:
    """Yudelson/Atlas render: the rollout's *answer* (CoT is stripped by Qwen's template for non-final
    turns) followed by the fixed follow-up turn. Yudelson inlined the system text into the user turn."""
    msgs = [{"role": "user", "content": f"{system}\n\n{user}"},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": FOLLOWUP[0]},
            {"role": "assistant", "content": FOLLOWUP[1]}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)


def capture_worker(rollout_file: str, out_prefix: str, model_id: str, token_layers: list[int],
                   n_tok_positions: int, token_budget: int, lora_path: str | None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from testbed.env import split_thinking, SYSTEM_PROMPT, load_impossible, load_possible, load_honeypot

    rows = [json.loads(l) for l in open(rollout_file)]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path).merge_and_unload().eval()
    L = model.config.num_hidden_layers + 1; d = model.config.hidden_size
    think_end_id = tok.convert_tokens_to_ids(THINK_END)
    N = len(rows)
    pooled = {k: np.zeros((N, L, d), np.float16) for k in ("mean_all", "mean_cot", "mean_answer", "last")}
    tokacts = np.zeros((N, len(token_layers), n_tok_positions, d), np.float16)
    tokpos = np.full((N, n_tok_positions), -1, np.int32)
    followup = np.zeros((N, L, d), np.float16)
    # ---- pass 1: policy tokens ----
    order = sorted(range(N), key=lambda i: len(rows[i]["prompt_token_ids"]) + len(rows[i]["completion_token_ids"]))
    i = 0
    while i < N:
        batch = []; maxlen = 0
        while i < N:
            r = rows[order[i]]; T = len(r["prompt_token_ids"]) + len(r["completion_token_ids"])
            if batch and max(maxlen, T) * (len(batch) + 1) > token_budget:
                break
            batch.append(order[i]); maxlen = max(maxlen, T); i += 1
        ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)
        spans = []
        for b, ri in enumerate(batch):
            r = rows[ri]; p, c = r["prompt_token_ids"], r["completion_token_ids"]
            seq = p + c; ids[b, :len(seq)] = torch.tensor(seq); att[b, :len(seq)] = 1
            spans.append((len(p), len(seq)))
        with torch.no_grad():
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
        hs = out.hidden_states  # tuple L x [B, T, d]
        for b, ri in enumerate(batch):
            s, e = spans[b]
            if e <= s:
                continue
            cot_m, ans_m = _cot_answer_masks(rows[ri]["completion_token_ids"], think_end_id)
            cot_t = torch.from_numpy(cot_m).cuda(); ans_t = torch.from_numpy(ans_m).cuda()
            pos = np.unique(np.linspace(s, e - 1, min(n_tok_positions, e - s)).round().astype(int))
            for l in range(L):
                h = hs[l][b, s:e].float()  # [Tc, d]
                pooled["mean_all"][ri, l] = h.mean(0).half().cpu().numpy()
                pooled["last"][ri, l] = h[-1].half().cpu().numpy()
                if cot_t.any():
                    pooled["mean_cot"][ri, l] = h[cot_t].mean(0).half().cpu().numpy()
                if ans_t.any():
                    pooled["mean_answer"][ri, l] = h[ans_t].mean(0).half().cpu().numpy()
            for j, l in enumerate(token_layers):
                tokacts[ri, j, :len(pos)] = hs[l][b, pos].half().cpu().numpy()
            tokpos[ri, :len(pos)] = pos - s
        del out, hs; torch.cuda.empty_cache()
        print(f"pass1 {i}/{N}", flush=True)
    # ---- pass 2: follow-up format (short: answer only + follow-up) ----
    tasks = {}
    for sp in ("train", "validation", "test"):
        for t in load_impossible(sp) + load_possible(sp) + load_honeypot(sp):
            tasks[(t.family, t.task_id)] = t
    texts = []
    for r in rows:
        t = tasks[(r["family"], r["task_id"])]
        _, ans = split_thinking(r["completion"])
        texts.append(followup_text(tok, SYSTEM_PROMPT, t.user_prompt(), ans))
    tok.padding_side = "left"
    bs = 16
    for s0 in range(0, N, bs):
        enc = tok(texts[s0:s0 + bs], return_tensors="pt", padding=True, add_special_tokens=False,
                  truncation=True, max_length=6144).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        last = torch.stack([h[:, -1, :] for h in out.hidden_states], 1)  # [B, L, d]
        followup[s0:s0 + bs] = last.half().cpu().numpy()
        del out, last, enc
    np.savez(out_prefix + "_pooled.npz", **pooled)
    np.savez(out_prefix + "_tokens.npz", tok=tokacts, pos=tokpos, layers=np.array(token_layers))
    np.savez(out_prefix + "_followup.npz", followup=followup)
    with open(out_prefix + "_index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"task_id": r["task_id"], "family": r["family"], "split": r["split"],
                                "sample_idx": r["sample_idx"]}) + "\n")
    print("done", out_prefix, flush=True)


def capture(rollout_file: Path, out_dir: Path, gpus: list[int], model_id: str = MODEL_ID,
            token_layers=DEFAULT_TOKEN_LAYERS, n_tok_positions: int = 128, token_budget: int = 20000,
            lora_path: str | None = None, tag: str = "acts") -> list[Path]:
    """Shard rollouts over GPUs; returns the list of shard prefixes written to ``out_dir``."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = [l for l in open(rollout_file)]
    shards = [rows[i::len(gpus)] for i in range(len(gpus))]
    procs, prefixes = [], []
    for g, shard in zip(gpus, shards):
        rf = out_dir / f"_{tag}_rows_{g}.jsonl"
        with open(rf, "w") as f:
            f.writelines(shard)
        prefix = str(out_dir / f"{tag}_shard{g}")
        prefixes.append(Path(prefix))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        cmd = [sys.executable, "-m", "testbed.activations", str(rf), prefix, model_id,
               ",".join(map(str, token_layers)), str(n_tok_positions), str(token_budget), lora_path or ""]
        log = open(out_dir / f"_{tag}_shard{g}.log", "w")
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
    fail = [g for g, p in procs if p.wait() != 0]
    if fail:
        raise RuntimeError(f"capture failed on gpus {fail}; see logs in {out_dir}")
    return prefixes


def load_capture(prefixes: list[Path]) -> dict:
    """Concatenate shards -> dict(index=list[dict], pooled={k: [N,L,d]}, tok=[N,K,P,d], pos=[N,P],
    layers=[K], followup=[N,L,d])."""
    idx, pooled, toks, pos, fu = [], {}, [], [], []
    layers = None
    for p in prefixes:
        idx += [json.loads(l) for l in open(str(p) + "_index.jsonl")]
        z = np.load(str(p) + "_pooled.npz")
        for k in z.files:
            pooled.setdefault(k, []).append(z[k])
        zt = np.load(str(p) + "_tokens.npz"); toks.append(zt["tok"]); pos.append(zt["pos"]); layers = zt["layers"]
        fu.append(np.load(str(p) + "_followup.npz")["followup"])
    return {"index": idx, "pooled": {k: np.concatenate(v) for k, v in pooled.items()},
            "tok": np.concatenate(toks), "pos": np.concatenate(pos), "layers": layers, "followup": np.concatenate(fu)}


if __name__ == "__main__":
    a = sys.argv[1:]
    capture_worker(a[0], a[1], a[2], [int(x) for x in a[3].split(",")], int(a[4]), int(a[5]), a[6] or None)
