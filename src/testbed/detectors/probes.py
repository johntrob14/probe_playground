"""Activation probes.

* ``LinearProbe``     – L2 logistic regression on a pooled residual vector at one layer.
* ``MLPProbe``        – 1-hidden-layer MLP on the same pooled vector.
* ``AttentionProbe``  – learned-query attention over token-level residuals at one layer (a single
                        softmax head + linear readout), the standard "attention probe" architecture.
* ``YudelsonProbe``   – loads a MonitorDecorrelation ``probe.npz``/``meta.json`` (per-layer LR, mean
                        sigmoid over kept layers) and scores follow-up-token activations. Used verbatim
                        for the OFF-POLICY generic deception probe (trained on 7 deception datasets, not on
                        this environment).

All probes are fit once on base-checkpoint rollouts and frozen. When one of them is the training
monitor, only its scalar output enters the RL advantage; the probe parameters are never touched by the
optimiser and no loss is ever taken through it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LinearProbe:
    def __init__(self, C: float = 1.0):
        self.C = C; self.mu = None; self.sd = None; self.w = None; self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, np.float32)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        clf = LogisticRegression(C=self.C, max_iter=3000, tol=1e-4, class_weight="balanced")
        clf.fit((X - self.mu) / self.sd, y)
        self.w = clf.coef_[0].astype(np.float32); self.b = float(clf.intercept_[0])
        return self

    def logit(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, np.float32) - self.mu) / self.sd) @ self.w + self.b

    def score(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit(X)))

    def direction(self) -> np.ndarray:
        """Unit direction in RAW activation space (the standardised weights mapped back)."""
        d = self.w / self.sd
        return d / (np.linalg.norm(d) + 1e-8)

    def save(self, path: Path) -> None:
        np.savez(path, mu=self.mu, sd=self.sd, w=self.w, b=self.b, C=self.C)

    @staticmethod
    def load(path: Path) -> "LinearProbe":
        z = np.load(path); p = LinearProbe(float(z["C"])); p.mu, p.sd, p.w, p.b = z["mu"], z["sd"], z["w"], float(z["b"]); return p


class MLPProbe:
    def __init__(self, hidden: int = 256, epochs: int = 200, lr: float = 1e-3, wd: float = 1e-2, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.wd, self.seed = hidden, epochs, lr, wd, seed
        self.net = None; self.mu = None; self.sd = None

    def fit(self, X, y, Xval=None, yval=None) -> "MLPProbe":
        import torch, torch.nn as nn
        torch.manual_seed(self.seed)
        X = np.asarray(X, np.float32); self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd); yt = torch.tensor(np.asarray(y, np.float32))
        pos_w = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())], dtype=torch.float32)
        self.net = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(self.hidden, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state = -1, None
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 128):
                idx = perm[i:i + 128]; opt.zero_grad()
                loss = lossf(self.net(Xt[idx]).squeeze(-1), yt[idx]); loss.backward(); opt.step()
            if Xval is not None and ep % 10 == 9:
                from testbed.metrics import auroc
                a = auroc(self.score(Xval), yval)
                if a > best:
                    best, best_state = a, {k: v.clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval(); return self

    def logit(self, X) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.net(torch.tensor((np.asarray(X, np.float32) - self.mu) / self.sd)).squeeze(-1).numpy()

    def score(self, X) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit(X)))


class AttentionProbe:
    """score = w · softmax_t(q · k_t) v_t, with k_t = W_k h_t, v_t = W_v h_t over token residuals h_t."""

    def __init__(self, d_head: int = 64, epochs: int = 150, lr: float = 1e-3, wd: float = 1e-2, seed: int = 0):
        self.d_head, self.epochs, self.lr, self.wd, self.seed = d_head, epochs, lr, wd, seed
        self.mod = None; self.mu = None; self.sd = None

    def _build(self, d):
        import torch, torch.nn as nn

        class Net(nn.Module):
            def __init__(s, d, dh):
                super().__init__(); s.k = nn.Linear(d, dh, bias=False); s.v = nn.Linear(d, dh); s.q = nn.Parameter(torch.randn(dh) * 0.02); s.out = nn.Linear(dh, 1)
            def forward(s, h, mask):  # h [B,T,d], mask [B,T] bool
                att = (s.k(h) @ s.q) / s.q.numel() ** 0.5
                att = att.masked_fill(~mask, -1e9).softmax(-1)
                return s.out((att.unsqueeze(-1) * s.v(h)).sum(1)).squeeze(-1)
        return Net(d, self.d_head)

    def _prep(self, H, mask):
        import torch
        H = (np.asarray(H, np.float32) - self.mu) / self.sd
        return torch.tensor(H), torch.tensor(np.asarray(mask, bool))

    def fit(self, H, mask, y, Hval=None, mval=None, yval=None) -> "AttentionProbe":
        import torch, torch.nn as nn
        torch.manual_seed(self.seed)
        m = np.asarray(mask, bool); flat = np.asarray(H, np.float32)[m]
        self.mu = flat.mean(0); self.sd = flat.std(0) + 1e-6
        Ht, Mt = self._prep(H, mask); yt = torch.tensor(np.asarray(y, np.float32))
        self.mod = self._build(Ht.shape[-1]).cuda() if torch.cuda.is_available() else self._build(Ht.shape[-1])
        dev = next(self.mod.parameters()).device
        pos_w = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())], dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW(self.mod.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state = -1, None
        for ep in range(self.epochs):
            self.mod.train(); perm = torch.randperm(len(Ht))
            for i in range(0, len(Ht), 32):
                idx = perm[i:i + 32]; opt.zero_grad()
                loss = lossf(self.mod(Ht[idx].to(dev), Mt[idx].to(dev)), yt[idx].to(dev)); loss.backward(); opt.step()
            if Hval is not None and ep % 10 == 9:
                from testbed.metrics import auroc
                a = auroc(self.score(Hval, mval), yval)
                if a > best:
                    best, best_state = a, {k: v.clone() for k, v in self.mod.state_dict().items()}
        if best_state is not None:
            self.mod.load_state_dict(best_state)
        self.mod.eval(); return self

    def logit(self, H, mask) -> np.ndarray:
        import torch
        Ht, Mt = self._prep(H, mask); dev = next(self.mod.parameters()).device; outs = []
        with torch.no_grad():
            for i in range(0, len(Ht), 64):
                outs.append(self.mod(Ht[i:i + 64].to(dev), Mt[i:i + 64].to(dev)).cpu().numpy())
        return np.concatenate(outs)

    def score(self, H, mask) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit(H, mask)))


class YudelsonProbe:
    """MonitorDecorrelation per-layer LR probe: mean sigmoid over kept layers, raw follow-up-token acts."""

    def __init__(self, path: Path):
        path = Path(path); z = np.load(path / "probe.npz"); meta = json.load(open(path / "meta.json"))
        self.layers = meta["kept_layers"]; self.meta = meta
        self.coef = {l: z[f"coef_{l}"][0].astype(np.float32) for l in self.layers}
        self.intercept = {l: float(z[f"intercept_{l}"][0]) for l in self.layers}

    def score(self, acts: np.ndarray, layers: list[int] | None = None) -> np.ndarray:
        """``acts`` [N, L, d] follow-up final-token residuals (all hidden_states incl. embeddings)."""
        use = layers or self.layers
        cols = [1 / (1 + np.exp(-(acts[:, l, :].astype(np.float32) @ self.coef[l] + self.intercept[l]))) for l in use]
        return np.mean(np.stack(cols, 1), 1)

    def per_layer_score(self, acts: np.ndarray) -> dict[int, np.ndarray]:
        return {l: self.score(acts, [l]) for l in self.layers}
