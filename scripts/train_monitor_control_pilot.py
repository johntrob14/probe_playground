"""Variance-matched probability control. Prior pilot trainers remain unchanged.

The serving adapter is a temporary working copy overwritten after each update.
The midpoint/final adapters, final optimizer, all rollouts and metrics are retained.
Run directories must be new. No full activation arrays are written to disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import zlib

import numpy as np
import torch

from testbed.env import load_honeypot, score_many, split_thinking
from testbed.grpo import Policy
from testbed.replay import FrozenLinearProbe, balanced_replay_loss, replay_logits
from testbed.rank_reward import group_rank_penalties
from testbed.monitor_reward_controls import variance_matched_probability_penalties
from testbed.rollout_client import RolloutClient
from testbed.sampling import build_prompt_text


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def probe_scores(policy, rows, probe, layer, end_id, micro, with_logits=False):
    was_training = policy.model.training
    policy.model.eval()
    values, logit_values = [], []
    with torch.no_grad():
        for start in range(0, len(rows), micro):
            logits = replay_logits(policy, rows[start:start + micro], probe, layer, "mean_cot", end_id)
            values.extend(logits.sigmoid().cpu().tolist())
            logit_values.extend(logits.cpu().tolist())
    policy.model.train(was_training)
    probabilities = np.asarray(values, dtype=np.float64)
    return (probabilities, np.asarray(logit_values, dtype=np.float64)) if with_logits else probabilities


def serializable_rng_state(rng):
    return {"python": rng.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def train(cfg):
    if cfg.get("penalty_transform") != "within_prompt_variance_matched_probability":
        raise ValueError("This control requires its explicitly declared variance-matched configuration")
    run_dir = Path(cfg["run_dir"]).resolve()
    root = Path(cfg["experiment_root"]).resolve()
    if root not in run_dir.parents:
        raise ValueError("Run must be inside the explicitly scoped experiment root")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    rng = random.Random(cfg["seed"])
    replay_rng = random.Random(cfg["seed"] + 10000)
    torch.manual_seed(cfg["seed"])
    selected_ids = set(json.loads(Path(cfg["train_task_ids"]).read_text()))
    tasks = [task for task in load_honeypot("train") if task.task_id in selected_ids]
    assert {task.task_id for task in tasks} == selected_ids
    task_map = {task.task_id: task for task in tasks}
    bank = read_jsonl(cfg["replay_bank"])
    assert all(int(row["task_id"]) in selected_ids for row in bank)
    by_label = {label: [row for row in bank if row["replay_label"] == label] for label in (0, 1)}
    assert all(by_label.values())
    assert all(len(values) <= cfg["max_replay_per_class"] for values in by_label.values())
    policy = Policy(cfg["model_id"], 64, 128, 0.0, init_adapter=cfg["init_adapter"])
    end_id = policy.tok.convert_tokens_to_ids("</think>")
    probe = FrozenLinearProbe.from_file(cfg["probe_file"], device=policy.device)
    params = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    client = RolloutClient(cfg["ports"], timeout=1800)
    client.wait_ready(max_wait=60)
    offset = (zlib.crc32(str(run_dir).encode()) % 100000 + 1) * 10000
    lam, mu, beta = cfg["lambda"], cfg["replay_weight"], cfg["kl_beta"]
    group_size, micro = cfg["group_size"], cfg["micro_batch"]
    live = run_dir / "serving_adapter"
    current_adapter = cfg["init_adapter"]
    start_wall = time.time()
    print(json.dumps({"event": "started", "run": str(run_dir), "pid": os.getpid(),
                      "train_tasks": len(tasks), "replay_per_class": {k: len(v) for k, v in by_label.items()}}), flush=True)

    with (run_dir / "metrics.jsonl").open("x") as metric_file, (run_dir / "rollouts.jsonl").open("x") as rollout_file:
        for step in range(1, cfg["steps"] + 1):
            # Bound our own artifacts, and leave ample free space for concurrent users.
            bytes_used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            if bytes_used > cfg["storage_budget_bytes"] or shutil.disk_usage(root).free < 30 * 1024**3:
                raise RuntimeError("Experiment storage/free-space guard reached; prior results retained")
            step_start = time.time()
            lr = cfg["lr"] * min(1.0, step / max(1, cfg["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch_tasks = rng.sample(tasks, cfg["batch_prompts"])
            prompt_ids = [policy.tok(build_prompt_text(policy.tok, task), add_special_tokens=False).input_ids
                          for task in batch_tasks]
            generated = client.generate(prompt_token_ids=prompt_ids, n=group_size,
                                        max_tokens=cfg["max_new_tokens"], temperature=1.0, top_p=1.0,
                                        seed=cfg["seed"] * 100000 + step,
                                        lora_path=str(current_adapter), lora_id=offset + step)
            rows = []
            for task, result in zip(batch_tasks, generated):
                for sample_idx, output in enumerate(result["outputs"]):
                    rows.append({"task_id": task.task_id, "sample_idx": sample_idx,
                                 "prompt_token_ids": result["prompt_token_ids"],
                                 "completion_token_ids": output["token_ids"], "completion": output["text"],
                                 "finish_reason": output["finish_reason"]})
            generation_seconds = time.time() - step_start
            scored = score_many([task_map[row["task_id"]] for row in rows], [row["completion"] for row in rows],
                                workers=cfg["execution_workers"], length_penalty=cfg["length_penalty"])
            hacks = np.asarray([score.hack for score in scored], dtype=bool)
            honest = np.asarray([score.honest_pass for score in scored], dtype=bool)
            capped = np.asarray([row["finish_reason"] == "length" for row in rows], dtype=bool)
            task_reward = np.asarray([score.reward for score in scored], dtype=np.float64)
            # Consistent across all new arms: code repetition to the token cap gets no task reward.
            task_reward[capped] = 0.0
            scores, monitor_logits = probe_scores(policy, rows, probe, cfg["layer"], end_id, micro, with_logits=True)
            monitor_penalties = variance_matched_probability_penalties(monitor_logits, group_size)
            target_rank_penalties = group_rank_penalties(monitor_logits, group_size)
            actual_variance = monitor_penalties.reshape(-1, group_size).var(axis=1)
            target_variance = target_rank_penalties.reshape(-1, group_size).var(axis=1)
            if not np.allclose(actual_variance, target_variance, rtol=1e-10, atol=1e-14):
                raise FloatingPointError("Monitor penalty variance does not match the declared rank control")
            rewards = task_reward - lam * monitor_penalties
            advantages = (rewards.reshape(-1, group_size) - rewards.reshape(-1, group_size).mean(axis=1, keepdims=True)).ravel()
            update_start = time.time()
            policy.model.train()
            optimizer.zero_grad(set_to_none=True)
            total_tokens = max(1, sum(len(row["completion_token_ids"]) for row in rows))
            order = np.argsort([len(row["completion_token_ids"]) for row in rows]).tolist()
            kl_acc, pg_acc, seq_kl_acc = 0.0, 0.0, 0.0
            for start in range(0, len(order), micro):
                indices = order[start:start + micro]
                chunk = [rows[index] for index in indices]
                ids, attention, completion_mask = policy._batch(chunk)
                completion_mask = completion_mask[:, 1:]
                with torch.no_grad(), policy.model.disable_adapter():
                    reference_logp = policy.token_logprobs(ids, attention)
                logp = policy.token_logprobs(ids, attention)
                # Unclipped Monte Carlo sequence log-ratio is logged separately from the k3 surrogate.
                log_ratio = logp - reference_logp
                seq_kl_acc += float((log_ratio.detach() * completion_mask).sum()) / len(rows)
                diff = (-log_ratio).clamp(-10, 10)
                kl = diff.exp() - diff - 1
                adv = torch.tensor(advantages[indices], device=policy.device, dtype=torch.float32).unsqueeze(1)
                policy_loss = -adv * logp
                loss = ((policy_loss + beta * kl) * completion_mask).sum() / total_tokens
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite policy loss; no optimizer update applied")
                loss.backward()
                kl_acc += float((kl.detach() * completion_mask).sum()) / total_tokens
                pg_acc += float((policy_loss.detach() * completion_mask).sum()) / total_tokens
                del ids, attention, completion_mask, reference_logp, logp, log_ratio, diff, kl, adv, policy_loss, loss
            replay_loss_value = 0.0
            actor_grad_norm = math.sqrt(sum(float(parameter.grad.detach().float().square().sum())
                                            for parameter in params if parameter.grad is not None))
            if mu > 0:
                # Every microbatch is paired, so the sum of class means retains its normalization.
                pairs = [(replay_rng.choice(by_label[0]), replay_rng.choice(by_label[1]))
                         for _ in range(cfg["replay_per_class_per_step"])]
                for good, bad in pairs:
                    chunk = [good, bad]
                    logits = replay_logits(policy, chunk, probe, cfg["layer"], "mean_cot", end_id)
                    labels = torch.tensor([0, 1], device=logits.device)
                    raw_loss = balanced_replay_loss(logits, labels)
                    loss = mu * raw_loss / len(pairs)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite replay loss; no optimizer update applied")
                    loss.backward()
                    replay_loss_value += float(raw_loss.detach()) / len(pairs)
                    del logits, raw_loss, loss
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]))
            if not math.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite gradients; no optimizer update applied")
            optimizer.step()
            policy.save_adapter(live)
            current_adapter = str(live)
            if step == cfg["steps"] // 2:
                policy.save_adapter(run_dir / f"checkpoint_step_{step}")
            mixed = [bool(hacks[start:start + group_size].any() and honest[start:start + group_size].any())
                     for start in range(0, len(rows), group_size)]
            record = {"step": step, "n": len(rows), "lambda": lam, "replay_weight": mu,
                      "penalty_transform": cfg["penalty_transform"],
                      "mean_monitor_penalty": float(monitor_penalties.mean()),
                      "within_group_rank_variance": float(target_variance.mean()),
                      "within_group_monitor_penalty_variance": float(actual_variance.mean()),
                      "variance_match_max_abs_error": float(np.max(np.abs(actual_variance - target_variance))),
                      "monitor_penalty_min": float(monitor_penalties.min()),
                      "monitor_penalty_max": float(monitor_penalties.max()),
                      "within_group_probe_logit_variance": float(monitor_logits.reshape(-1, group_size).var(axis=1).mean()),
                      "hack_rate": float(hacks.mean()), "honest_pass_rate": float(honest.mean()),
                      "terminated_hack_rate": float((hacks & ~capped).mean()),
                      "terminated_honest_pass_rate": float((honest & ~capped).mean()),
                      "loose_rh_rate": float(np.mean([score.flags.get("loose_rh", False) for score in scored])),
                      "finish_reason_length_rate": float(capped.mean()),
                      "empty_cot_rate": float(np.mean([len(split_thinking(row["completion"])[0].strip()) < 20 for row in rows])),
                      "task_reward": float(task_reward.mean()), "total_reward": float(rewards.mean()),
                      "probe_mean_hack": float(scores[hacks].mean()) if hacks.any() else None,
                      "probe_mean_honest": float(scores[honest].mean()) if honest.any() else None,
                      "undetected_hack_rate": float((hacks & (scores <= 0.5)).mean()),
                      "probe_tpr_hacks": float((scores[hacks] > 0.5).mean()) if hacks.any() else None,
                      "probe_fpr_honest": float((scores[honest] > 0.5).mean()) if honest.any() else None,
                      "mixed_hack_honest_group_rate": float(np.mean(mixed)),
                      "within_group_probe_variance": float(scores.reshape(-1, group_size).var(axis=1).mean()),
                      "mean_completion_tokens": total_tokens / len(rows), "kl_k3_per_token": kl_acc,
                      "sequence_log_ratio_mc": seq_kl_acc, "pg_loss": pg_acc,
                      "replay_loss": replay_loss_value, "gradient_norm_before_clip": grad_norm,
                      "actor_gradient_norm": actor_grad_norm,
                      "lr": lr, "generation_seconds": generation_seconds,
                      "update_seconds": time.time() - update_start, "elapsed_seconds": time.time() - start_wall}
            metric_file.write(json.dumps(record, allow_nan=False) + "\n")
            metric_file.flush()
            for row, score, monitor_score, monitor_logit, monitor_penalty, advantage, reward in zip(
                    rows, scored, scores, monitor_logits, monitor_penalties, advantages, task_reward):
                rollout_file.write(json.dumps({**row, **dataclasses.asdict(score), "step": step,
                                              "probe_score": float(monitor_score), "probe_logit": float(monitor_logit),
                                              "monitor_penalty": float(monitor_penalty), "advantage": float(advantage),
                                              "training_task_reward": float(reward)}) + "\n")
            rollout_file.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
            # Logging has oracle labels for every output. Replay sees only this bounded
            # subset, selected without consulting the labels.
            if mu > 0 and step % cfg["refresh_every"] == 0:
                selection = set(replay_rng.sample(range(len(rows)), min(cfg["refresh_random"], len(rows))))
                selection.update(np.argsort(scores)[:cfg["refresh_low_score"]].tolist())
                added = []
                for index in sorted(selection):
                    row, score = rows[index], scored[index]
                    if capped[index] or not score.parsed or score.truncated or "</think>" not in row["completion"]:
                        continue
                    if not (score.hack or score.honest_pass):
                        continue
                    entry = {**row, "replay_label": int(score.hack), "label": int(score.hack),
                             "hack": bool(score.hack), "honest_pass": bool(score.honest_pass),
                             "key": f"refresh|{step}|{row['task_id']}|{row['sample_idx']}", "source_step": step,
                             "source": "bounded_on_policy_refresh"}
                    by_label[entry["replay_label"]].append(entry)
                    added.append(entry)
                # Keep all original examples; bound only the new in-memory replay tail.
                for label in (0, 1):
                    original_count = sum(row["replay_label"] == label for row in bank)
                    maximum_new = cfg["max_replay_per_class"] - original_count
                    if len(by_label[label]) > cfg["max_replay_per_class"]:
                        tail = by_label[label][-maximum_new:] if maximum_new else []
                        by_label[label] = by_label[label][:original_count] + tail
                with (run_dir / "replay_refresh.jsonl").open("a") as refresh_file:
                    for entry in added:
                        refresh_file.write(json.dumps(entry) + "\n")
                print(json.dumps({"event": "replay_refresh", "step": step,
                                  "labels_exposed": len(selection), "usable_added": len(added)}), flush=True)
        # The serving copy is now the retained final adapter; no final weight duplicate is needed.
        torch.save({"optimizer": optimizer.state_dict(), "step": cfg["steps"], "rng": serializable_rng_state(rng),
                    "replay_rng": replay_rng.getstate(), "config": cfg}, run_dir / "optimizer_final.pt")
        (run_dir / "complete.json").write_text(json.dumps({"steps": cfg["steps"], "adapter": str(live),
                                                           "elapsed_seconds": time.time() - start_wall}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()))
