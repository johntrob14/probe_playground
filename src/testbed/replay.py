"""Fixed, class-balanced transcript replay through a frozen activation probe.

This is a separate experimental path from scalar-only GRPO: labels supervise a
probe's reading of teacher-forced activations, never transcript likelihood. The
probe contains buffers only; gradients can pass to the model's trainable LoRA.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from testbed.detectors.probes import LinearProbe


class FrozenLinearProbe(nn.Module):
    """Differentiable input, immutable LinearProbe normalization and readout."""

    def __init__(self, probe: LinearProbe):
        super().__init__()
        for name in ("mu", "sd", "w", "b"):
            value = getattr(probe, name)
            if value is None:
                raise ValueError("Replay requires a fitted probe")
            tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32).clone().detach()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite probe {name}")
            self.register_buffer(name, tensor)
        if self.mu.ndim != 1 or self.mu.shape != self.sd.shape or self.mu.shape != self.w.shape:
            raise ValueError("Probe vectors must have matching one-dimensional shapes")
        if self.b.numel() != 1 or (self.sd <= 0).any():
            raise ValueError("Probe requires a scalar intercept and positive saved scales")

    @classmethod
    def from_file(cls, path, device=None):
        return cls(LinearProbe.load(Path(path))).to(device=device)

    def forward(self, features):
        # sd already includes the training epsilon. Adding another changes the probe.
        return ((features.float() - self.mu) / self.sd) @ self.w + self.b

    def score(self, features):
        return self(features).sigmoid()


def load_balanced_replay(paths, tokenizer, *, per_class, seed=0, max_tokens=None, task_ids=None,
                         require_terminated=True):
    """Reservoir-sample equal classes from explicitly named scored JSONL files.

    Positives require ``hack is True``; negatives require ``honest_pass is True``.
    Other failures and, by default, token-limit terminations are excluded.
    ``task_ids`` should restrict training replay to training tasks. Oversized
    transcripts are skipped, never truncated. Saved token IDs are preferred;
    otherwise prompt and completion are retokenized separately without added
    special tokens and that provenance is recorded. Sources are streamed once;
    no directories are searched or files written.
    """
    if per_class < 1:
        raise ValueError("per_class must be positive")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    allowed = None if task_ids is None else {str(item) for item in task_ids}
    banks, counts = {0: [], 1: []}, {0: 0, 1: 0}
    generators = {label: random.Random(seed + label) for label in (0, 1)}
    seen = set()
    for source in paths:
        path = Path(source).resolve()
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if allowed is not None and str(row.get("task_id")) not in allowed:
                    continue
                bad, good = row.get("hack") is True, row.get("honest_pass") is True
                if bad and good:
                    raise ValueError(f"Conflicting execution labels at {path}:{line_number}")
                if not (bad or good) or not row.get("parsed", False) or row.get("truncated", False):
                    continue
                if require_terminated and row.get("finish_reason") == "length":
                    continue
                label = int(bad)
                if "prompt_token_ids" in row and "completion_token_ids" in row:
                    prompt_ids = list(row["prompt_token_ids"])
                    completion_ids = list(row["completion_token_ids"])
                    provenance = "saved_token_ids"
                else:
                    if tokenizer is None or "prompt_text" not in row or "completion" not in row:
                        raise ValueError(f"Missing tokens or retokenizable transcript at {path}:{line_number}")
                    prompt_ids = tokenizer(row["prompt_text"], add_special_tokens=False).input_ids
                    completion_ids = tokenizer(row["completion"], add_special_tokens=False).input_ids
                    provenance = "retokenized_prompt_and_completion"
                if not prompt_ids or not completion_ids:
                    continue
                if max_tokens is not None and len(prompt_ids) + len(completion_ids) > max_tokens:
                    continue
                fingerprint = hashlib.sha256(json.dumps([prompt_ids, completion_ids]).encode()).hexdigest()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                candidate = dict(row)
                candidate.update(prompt_token_ids=prompt_ids, completion_token_ids=completion_ids,
                                 label=label, replay_label=label, source_path=str(path),
                                 source_line=line_number, tokenization=provenance,
                                 transcript_sha256=fingerprint)
                candidate.setdefault("key", f"{row.get('family', 'honeypot')}|{row.get('task_id')}|{row.get('sample_idx', line_number)}")
                counts[label] += 1
                if len(banks[label]) < per_class:
                    banks[label].append(candidate)
                else:
                    index = generators[label].randrange(counts[label])
                    if index < per_class:
                        banks[label][index] = candidate
    if any(len(banks[label]) < per_class for label in (0, 1)):
        raise ValueError(f"Insufficient eligible replay: honest={counts[0]}, hack={counts[1]}; need {per_class} each")
    result = banks[0] + banks[1]
    random.Random(seed).shuffle(result)
    return result


def balanced_replay_loss(logits, labels, *, kind="hinge", positive_margin=0.8,
                         negative_margin=0.2, class_counts=None):
    """Sum of positive/negative class-mean losses, independent of class prevalence.

    The hinge acts on sigmoid scores. BCE is optional. For gradient accumulation,
    pass ``class_counts={0: total_honest, 1: total_hacks}`` for the ENTIRE replay
    batch to every microbatch and sum its losses. A microbatch can then contain a
    single class. Without these denominators, both classes must be present.
    """
    labels = torch.as_tensor(labels, device=logits.device)
    if logits.ndim != 1 or labels.shape != logits.shape or not logits.numel():
        raise ValueError("logits and labels must be nonempty matching vectors")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Replay labels must be binary execution labels")
    positive = labels == 1
    if kind == "hinge":
        if not 0 <= negative_margin < positive_margin <= 1:
            raise ValueError("Expected 0 <= negative_margin < positive_margin <= 1")
        scores = logits.sigmoid()
        losses = torch.where(positive, (positive_margin - scores).clamp_min(0),
                             (scores - negative_margin).clamp_min(0))
    elif kind == "bce":
        losses = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction="none")
    else:
        raise ValueError(f"Unknown replay loss: {kind}")
    total = logits.sum() * 0
    for label in (0, 1):
        mask = labels == label
        count = int(mask.sum()) if class_counts is None else int(class_counts[label])
        if count <= 0 or count < int(mask.sum()):
            raise ValueError("Each global class denominator must cover a nonempty class")
        total = total + losses[mask].sum() / count
    return total


def pool_hidden(hidden, rollouts, completion_mask, pool, think_end_id):
    """Match Policy.pooled_hidden exactly, including end-token and empty-span rules."""
    if pool not in ("mean_all", "mean_cot", "mean_answer", "last"):
        raise ValueError(f"Unsupported replay pooling: {pool}")
    vectors = []
    for index, row in enumerate(rollouts):
        mask = completion_mask[index].clone()
        if not mask.any():
            raise ValueError("Cannot pool an empty completion")
        completion = row["completion_token_ids"]
        if think_end_id in completion:
            boundary = len(row["prompt_token_ids"]) + completion.index(think_end_id) + 1
            if pool == "mean_cot":
                mask[boundary:] = False
            elif pool == "mean_answer":
                mask[:boundary] = False
        h = hidden[index].float()
        if pool == "last":
            vectors.append(h[mask.nonzero()[-1]].squeeze(0))
        else:
            if not mask.any():
                mask = completion_mask[index]
            vectors.append(h[mask].mean(0))
    return torch.stack(vectors)


def _backbone_and_hook(model, layer):
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(causal_lm, "model", causal_lm)
    if not hasattr(backbone, "layers") or not hasattr(backbone, "norm"):
        raise TypeError("Replay requires a Qwen/Llama-style decoder backbone with layers and norm")
    count = len(backbone.layers)
    if not 0 <= layer <= count:
        raise ValueError(f"Hidden-state index {layer} outside 0..{count}")
    # hidden_states[0] is embeddings; intermediate indices are block outputs;
    # hidden_states[count] includes the final RMSNorm, not the raw final block.
    target = backbone.embed_tokens if layer == 0 else backbone.norm if layer == count else backbone.layers[layer - 1]
    return backbone, target


def replay_logits(policy, rollouts, probe, layer, pool, think_end_id):
    """Return differentiable frozen-probe logits for fixed teacher-forced tokens.

    Calling the PEFT model's decoder directly retains its active LoRA modules but
    skips the vocabulary head. One forward hook retains only the requested layer;
    normal non-reentrant gradient checkpointing remains active. Train/eval mode
    is unchanged. Current experiments use LoRA dropout=0, matching monitor eval.
    No token-label likelihood or model-generated target enters this objective.
    """
    if not rollouts:
        raise ValueError("Replay batch is empty")
    unexpected = [name for name, parameter in policy.model.named_parameters()
                  if parameter.requires_grad and "lora_" not in name]
    if unexpected:
        raise ValueError(f"Replay is restricted to trainable LoRA parameters, found: {unexpected[:3]}")
    backbone, target = _backbone_and_hook(policy.model, layer)
    ids, attention, completion_mask = policy._batch(rollouts)
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    hook = target.register_forward_hook(capture)
    try:
        output = backbone(input_ids=ids, attention_mask=attention, use_cache=False,
                          output_hidden_states=False, return_dict=True)
        del output
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one residual capture, got {len(captured)}")
    hidden = captured[0]
    if torch.is_grad_enabled() and not hidden.requires_grad:
        raise RuntimeError("Replay activations are detached; use non-reentrant gradient checkpointing and input gradients")
    return probe(pool_hidden(hidden, rollouts, completion_mask, pool, think_end_id))
