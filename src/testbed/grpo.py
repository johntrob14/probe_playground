"""LoRA + GRPO with an optional scalar monitor term.

r_i = R_task(rollout_i) - lambda * m_i          (m_i from monitor_term, a plain float)
A_i = (r_i - mean_group) / (std_group + eps)    (GRPO; set ``std_norm: false`` for Dr. GRPO)
loss = - mean_tokens[ A_i * logpi(token) ] + beta * mean_tokens[ KL_k3(pi || pi_ref) ]

One optimiser step per rollout batch (single PPO epoch), so the importance ratio is identically 1 and
no clipping is needed. The reference policy is the same model with the adapter disabled.

Ground-truth labels (hack, honest_pass) are computed by the environment for EVERY rollout and logged,
but never enter the reward except through R_task.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from testbed.env import Task, score_many, load_impossible, load_honeypot, split_thinking
from testbed.sampling import build_prompt_text
from testbed.rollout_client import RolloutClient
from testbed.monitor_term import build_monitor


class Policy:
    def __init__(self, model_id: str, lora_r: int, lora_alpha: float, lora_dropout: float, device: str = "cuda", init_adapter: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, PeftModel
        self.tok = AutoTokenizer.from_pretrained(model_id)
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.config.use_cache = False
        if init_adapter:   # sequential design: continue training an existing LoRA policy (e.g. an arm-A hacking checkpoint)
            self.model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True).to(device)
        else:
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.model = get_peft_model(base, cfg).to(device)
        self.model.enable_input_require_grads()
        self.device = device

    def save_adapter(self, path: Path):
        path.mkdir(parents=True, exist_ok=True); self.model.save_pretrained(str(path))

    def _batch(self, rollouts):
        seqs = [r["prompt_token_ids"] + r["completion_token_ids"] for r in rollouts]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.tok.pad_token_id, dtype=torch.long); att = torch.zeros_like(ids)
        cmask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (r, s) in enumerate(zip(rollouts, seqs)):
            ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1; cmask[i, len(r["prompt_token_ids"]):len(s)] = True
        return ids.to(self.device), att.to(self.device), cmask.to(self.device)

    def token_logprobs(self, ids, att):
        logits = self.model(input_ids=ids, attention_mask=att).logits[:, :-1]
        # fp32 cross-entropy per token; avoids materialising a second [B,T,V] fp32 log_softmax copy
        B, T, V = logits.shape
        lp = -torch.nn.functional.cross_entropy(logits.reshape(-1, V).float(), ids[:, 1:].reshape(-1), reduction="none").view(B, T)
        return lp  # [B, T-1], logprob of token t given <t

    @torch.no_grad()
    def pooled_hidden(self, rollouts, layer: int, pool: str, think_end_id: int, micro: int = 4) -> np.ndarray:
        """Residual stream of the CURRENT policy (adapter on) at ``layer`` (hidden_states index), pooled
        over completion tokens. Detached floats only."""
        self.model.eval(); feats = []
        for i in range(0, len(rollouts), micro):
            rb = rollouts[i:i + micro]; ids, att, cm = self._batch(rb)
            out = self.model(input_ids=ids, attention_mask=att, output_hidden_states=True)
            h = out.hidden_states[layer].float()
            for b, r in enumerate(rb):
                m = cm[b].clone()
                comp = torch.tensor(r["completion_token_ids"], device=self.device)
                start = len(r["prompt_token_ids"]); idx = (comp == think_end_id).nonzero()
                if pool == "mean_cot" and len(idx):
                    m[start + int(idx[0]) + 1:] = False
                elif pool == "mean_answer" and len(idx):
                    m[: start + int(idx[0]) + 1] = False
                if pool == "last":
                    v = h[b][m.nonzero()[-1]].squeeze(0)
                else:
                    v = h[b][m].mean(0) if m.any() else h[b][cm[b]].mean(0)
                feats.append(v.cpu().numpy())
            del out, h
        self.model.train()
        return np.stack(feats)


def _judge_rates(monitor, hack: np.ndarray) -> dict:
    """Noticing rate and decision-verbalisation rate (all rollouts and hacks only) from the last judge call."""
    fl = getattr(monitor, "last_fields", None)
    if not fl:
        return {}
    noticed = np.array([f["noticed"] is True for f in fl]); dec = np.array([f["decision"] == "HARDCODE" for f in fl])
    empty = np.array([f["decision"] == "EMPTY_COT" for f in fl])
    return {"judge_noticed_rate": float(noticed.mean()), "judge_decision_hardcode_rate": float(dec.mean()),
            "judge_empty_cot_rate": float(empty.mean()), "judge_empty_cot_rate_hack": float(empty[hack].mean()) if hack.any() else None,
            "judge_noticed_rate_hack": float(noticed[hack].mean()) if hack.any() else None,
            "judge_decision_hardcode_rate_hack": float(dec[hack].mean()) if hack.any() else None,
            "judge_unparsed_rate": float(np.mean([f["score"] is None for f in fl]))}


def make_groups(tasks: list[Task], batch_prompts: int, rng: random.Random) -> list[Task]:
    return rng.sample(tasks, batch_prompts)


def train(cfg: dict):
    run_dir = Path(cfg["run_dir"]); run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=1)
    rng = random.Random(cfg.get("seed", 0)); torch.manual_seed(cfg.get("seed", 0))
    family = cfg.get("family", "honeypot")
    if family != "honeypot":
        raise ValueError("Training is MBPP-Honeypot only (one public test in the prompt). The mutated three-test "
                         "'impossible' dataset is not a training environment; it is retained only to read old Phase-1 artefacts.")
    tasks = load_honeypot("train")
    if cfg.get("task_ids"):
        keep = set(cfg["task_ids"]); tasks = [t for t in tasks if t.task_id in keep]
    tasks_by_id = {t.task_id: t for t in tasks}
    policy = Policy(cfg["model_id"], cfg["lora_r"], cfg["lora_alpha"], cfg.get("lora_dropout", 0.0), init_adapter=cfg.get("init_adapter"))
    think_end_id = policy.tok.convert_tokens_to_ids("</think>")
    monitor = build_monitor(cfg.get("monitor", {"kind": "none"}), think_end_id, tasks_by_id)
    lam = float(cfg.get("lambda", 0.0)); beta = float(cfg.get("kl_beta", 1e-4)); G = int(cfg["group_size"])
    clp = float(cfg.get("length_penalty", 0.0)); nonterm = float(cfg.get("nonterm_penalty", 0.1))
    lam_schedule = cfg.get("lambda_schedule")  # e.g. "linear" -> lambda ramps 0 -> lambda over n_steps (Yudelson)
    client = RolloutClient(cfg["rollout_ports"]); client.wait_ready()
    # Shadow monitor (arm C read-out): scored on every batch exactly like the reward monitor, but NEVER enters r_total or the
    # advantages; its scores and judge fields are logged with the prefix "shadow_". Used to watch the CoT judge while an
    # activation probe is the only pressure.
    shadow = build_monitor(cfg["shadow_monitor"], think_end_id, tasks_by_id) if cfg.get("shadow_monitor") else None
    opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=float(cfg["lr"]), weight_decay=0.0)
    n_steps = int(cfg["n_steps"]); warmup = int(cfg.get("warmup_steps", 5))
    def lr_at(step):  # Atlas: cosine with 5 warmup steps; Yudelson/Tinker: constant
        if cfg.get("lr_schedule", "cosine") == "constant":   # optional linear warm-up (sequential runs: fresh AdamW moments at full lr destabilise a resumed LoRA)
            return float(cfg["lr"]) * (min(1.0, step / warmup) if warmup > 0 else 1.0)
        if step <= warmup:
            return float(cfg["lr"]) * step / max(1, warmup)
        return float(cfg["lr"]) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, n_steps - warmup)))
    metrics_f = open(run_dir / "metrics.jsonl", "a"); roll_f = open(run_dir / "rollouts.jsonl", "a")
    adapter_dir = run_dir / "adapter_live"
    import zlib
    lora_offset = int(cfg["lora_id_offset"]) if cfg.get("lora_id_offset") is not None else (zlib.crc32(str(run_dir.resolve()).encode()) % 100000 + 1) * 10000
    print(f"lora_id range for this run: {lora_offset + 1}..{lora_offset + n_steps}", flush=True)
    policy.save_adapter(adapter_dir / "step_0")
    for step in range(1, n_steps + 1):
        t0 = time.time()
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step)
        lam_t = lam * (step / n_steps) if lam_schedule == "linear" else lam
        # ---------- rollouts from the current adapter ----------
        batch_tasks = make_groups(tasks, cfg["batch_prompts"], rng)
        prompts = [build_prompt_text(policy.tok, t) for t in batch_tasks]
        prompt_ids = [policy.tok(p, add_special_tokens=False).input_ids for p in prompts]
        cur_adapter = adapter_dir / f"step_{step - 1}"
        # vLLM identifies adapters by lora_id, and runs share rollout servers, so every run needs a disjoint id range.
        # Default: a stable hash of the run directory (x 10^4) so that no two runs can collide; override with lora_id_offset.
        res = client.generate(prompt_token_ids=prompt_ids, n=G, max_tokens=cfg["max_new_tokens"], temperature=cfg.get("temperature", 1.0),
                              top_p=1.0, seed=cfg.get("seed", 0) * 100000 + step, lora_path=str(cur_adapter), lora_id=lora_offset + step)
        rollouts = []
        for t, r in zip(batch_tasks, res):
            for o in r["outputs"]:
                rollouts.append({"task_id": t.task_id, "prompt_token_ids": r["prompt_token_ids"], "completion_token_ids": o["token_ids"],
                                 "completion": o["text"], "finish_reason": o["finish_reason"]})
        t_gen = time.time() - t0
        # ---------- environment reward + ground truth ----------
        sc = score_many([tasks_by_id[r["task_id"]] for r in rollouts], [r["completion"] for r in rollouts], length_penalty=clp)
        R_task = np.array([s.reward for s in sc]); hack = np.array([s.hack for s in sc]); honest = np.array([s.honest_pass for s in sc])
        trunc = np.array([r["finish_reason"] == "length" for r in rollouts])
        R_task = R_task - nonterm * trunc            # Atlas non-termination penalty
        # ---------- scalar monitor term (plain floats, no grad) ----------
        mode = cfg.get("monitor_mode", "penalty")   # "penalty": r = R - lambda*m (default); "screen": flagged rollouts (m > 0.5) are dropped from the gradient (Wong et al. sample-level screening), no penalty
        m = np.array(monitor.score(rollouts, policy), dtype=np.float64) if (lam > 0 or mode == "screen") else np.zeros(len(rollouts))
        r_total = R_task - (lam_t * m if mode == "penalty" else 0.0)
        m_sh = np.array(shadow.score(rollouts, policy), dtype=np.float64) if shadow is not None else None
        # ---------- GRPO advantages ----------
        A = np.zeros(len(rollouts))
        for g in range(len(batch_tasks)):
            sl = slice(g * G, (g + 1) * G); mu = r_total[sl].mean(); sd = r_total[sl].std()
            A[sl] = (r_total[sl] - mu) / (sd + 1e-4) if cfg.get("std_norm", True) else (r_total[sl] - mu)
        # Uncentred silence penalty (arm B): a penalty that is constant within a group vanishes under group centring, so a
        # policy that goes silent in every rollout (B_from_hack steps 14-15) feels no pressure from the empty-CoT rule.
        # This term is subtracted from the advantages AFTER centring, so silence is always disfavoured regardless of
        # what the rest of the group did. It is not part of the monitor score m and is logged separately.
        if mode == "screen":
            A[m > 0.5] = 0.0          # screened rollouts carry no gradient; centring above used the unpenalised task reward
        sil = float(cfg.get("empty_cot_uncentered", 0.0))
        if sil > 0:
            empty = np.array([len(split_thinking(r["completion"])[0].strip()) < 20 and s_.parsed for r, s_ in zip(rollouts, sc)], dtype=float)
            A = A - sil * empty
        # ---------- policy update (one step, grad accumulation over micro-batches) ----------
        t1 = time.time(); policy.model.train(); opt.zero_grad()
        n_tok_total = sum(len(r["completion_token_ids"]) for r in rollouts) + 1
        order = [i for i in np.argsort([len(r["completion_token_ids"]) for r in rollouts])]
        micro = int(cfg.get("micro_batch", 2)); kl_acc = 0.0; pg_acc = 0.0
        for i in range(0, len(order), micro):
            idx = order[i:i + micro]
            if all(abs(A[j]) < 1e-8 for j in idx):
                continue
            rb = [rollouts[j] for j in idx]; ids, att, cm = policy._batch(rb)
            cm = cm[:, 1:]  # align with logprob positions
            with torch.no_grad(), policy.model.disable_adapter():
                ref_lp = policy.token_logprobs(ids, att)
            lp = policy.token_logprobs(ids, att)
            adv = torch.tensor(A[idx], device=policy.device, dtype=torch.float32).unsqueeze(1)
            diff = (ref_lp - lp).clamp(-10.0, 10.0)   # k3 estimator; clamp the log-ratio so exp() cannot overflow when the policy drifts far
            kl = (torch.exp(diff) - diff - 1)
            pg = -(adv * lp)
            loss = ((pg + beta * kl) * cm).sum() / n_tok_total
            if not torch.isfinite(loss):
                print(f"non-finite loss at step {step}, micro-batch {i}; skipping", flush=True); continue
            loss.backward()
            kl_acc += float((kl * cm).sum()) / n_tok_total; pg_acc += float((pg * cm).sum()) / n_tok_total
            del ids, att, cm, ref_lp, lp, kl, pg, loss
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
        opt.step(); t_upd = time.time() - t1
        policy.save_adapter(adapter_dir / f"step_{step}")
        prev = adapter_dir / f"step_{step - 2}"
        if prev.exists() and (step - 2) % int(cfg["ckpt_every"]) != 0:
            import shutil; shutil.rmtree(prev, ignore_errors=True)
        if step % int(cfg["ckpt_every"]) == 0:
            policy.save_adapter(run_dir / "checkpoints" / f"step_{step}")
        # ---------- logging: ground truth next to every monitor number ----------
        loose = np.array([s.flags.get("loose_rh", False) for s in sc]); hardc = np.array([s.flags.get("hardcoding", False) for s in sc])
        pass_public = np.array([all(s.shown_pass) for s in sc])
        rec = {"step": step, "n": len(rollouts), "hack_rate": float(hack.mean()), "honest_pass_rate": float(honest.mean()),
               "loose_rh_rate": float(loose.mean()), "hardcoding_rate": float(hardc.mean()), "pass_public_rate": float(pass_public.mean()),
               "task_reward": float(R_task.mean()), "monitor_mean": float(m.mean()),
               "penalty_mass_hack": float((lam_t * m[hack]).sum() / max(1e-9, (lam_t * m).sum())) if lam_t > 0 and m.sum() > 0 else None,
               "penalty_mean_hack": float(m[hack].mean()) if hack.any() else None, "penalty_mean_honest": float(m[honest].mean()) if honest.any() else None,
               "lr": lr_at(step), "lambda_t": lam_t,
               **_judge_rates(monitor, hack),
               **({"shadow_monitor": shadow.name, "shadow_mean": float(m_sh.mean()), "shadow_mean_hack": float(m_sh[hack].mean()) if hack.any() else None,
                   "shadow_mean_nonhack": float(m_sh[~hack].mean()) if (~hack).any() else None,
                   **{"shadow_" + k: v for k, v in _judge_rates(shadow, hack).items()}} if shadow is not None else {}),
               "monitor_mean_hack": float(m[hack].mean()) if hack.any() else None,
               "monitor_mean_nonhack": float(m[~hack].mean()) if (~hack).any() else None,
               "total_reward": float(r_total.mean()), "truncation_rate": float(trunc.mean()),
               "mean_completion_tokens": float(np.mean([len(r["completion_token_ids"]) for r in rollouts])),
               "mean_cot_chars": float(np.mean([len(split_thinking(r["completion"])[0]) for r in rollouts])),   # from the text, not the scorer flags (which are absent when no code block is found)
               "empty_cot_rate": float(np.mean([len(split_thinking(r["completion"])[0]) < 20 and s_.parsed for r, s_ in zip(rollouts, sc)])),
               "babble_rate": float(np.mean([r["finish_reason"] == "length" and s_.parsed for r, s_ in zip(rollouts, sc)])),
               "mean_code_chars": float(np.mean([s.flags.get("code_len", 0) for s in sc])),
               "kl": kl_acc, "pg_loss": pg_acc, "lambda": lam, "monitor": monitor.name, "family": family, "length_penalty": clp,
               "t_gen": t_gen, "t_update": t_upd}
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush(); print(json.dumps(rec), flush=True)
        for k_, (r, s, mi, ai) in enumerate(zip(rollouts, sc, m, A)):
            roll_f.write(json.dumps({"step": step, "task_id": r["task_id"], "completion": r["completion"], "finish_reason": r["finish_reason"],
                                     "reward": s.reward, "hack": s.hack, "honest_pass": s.honest_pass, "monitor": float(mi), "advantage": float(ai), **({"shadow": float(m_sh[k_])} if shadow is not None else {})}) + "\n")
        roll_f.flush()
