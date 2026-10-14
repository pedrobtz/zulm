"""PyTorch reference fixtures for zulm (design §11; Stage 6).

    python tools/make-torch-reference.py tests/testthat/fixtures

Writes one GGUF file per case, read by zl_load() in
tests/testthat/test-fixtures.R. Each holds the module's parameters under
the names zulm's modules give them, and tensors named input.*, output.* and
grad.* (the gradient of sum(output * input.w) with respect to each
parameter). A zulm tensor of dims c(d1, d2, ...) is NumPy's array of shape
(..., d2, d1) with the same bytes: a batch of sequences c(dim, n, batch) is
PyTorch's (batch, n, dim), and a linear weight c(n_in, n_out) is
nn.Linear's (n_out, n_in) weight, written unchanged.

Run offline; the fixtures are committed (budget 1 MB in all).
"""
import math
import sys

import numpy as np
import torch
from gguf import GGUFWriter

torch.manual_seed(20261014)
DT = torch.float32


def rows(t):
    return t.detach().to(DT).contiguous().numpy()


def write(path, tensors, meta):
    w = GGUFWriter(path, "zulm-fixture")
    for k, v in meta.items():
        if isinstance(v, str):
            w.add_string(k, v)
        else:
            w.add_float64(k, float(v))
    for name, t in tensors.items():
        assert len(name) < 64, name
        w.add_tensor(name, rows(t))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def linear_params(prefix, lin, out):
    out[f"{prefix}.weight"] = lin.weight
    if lin.bias is not None:
        out[f"{prefix}.bias"] = lin.bias


def grads_of(params, out, wout):
    loss = (out * wout).sum()
    gs = torch.autograd.grad(loss, list(params.values()))
    return {f"grad.{k}": g for k, g in zip(params.keys(), gs)}


# ---- zulm's modules, written in PyTorch -------------------------------------


class Attention(torch.nn.Module):
    def __init__(self, dim, heads, causal):
        super().__init__()
        self.h, self.dh, self.causal = heads, dim // heads, causal
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.o = torch.nn.Linear(dim, dim)

    def forward(self, x):  # (B, n, dim)
        b, n, d = x.shape
        sp = lambda t: t.view(b, n, self.h, self.dh).transpose(1, 2)  # (B, h, n, dh)
        q, k, v = sp(self.q(x)), sp(self.k(x)), sp(self.v(x))
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)  # (B, h, n_q, n_k)
        if self.causal:
            s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
        a = s.softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(b, n, d)
        return self.o(o)


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = torch.nn.Linear(dim, hidden)
        self.up = torch.nn.Linear(dim, hidden)
        self.down = torch.nn.Linear(hidden, dim)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class Block(torch.nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, heads, True)
        self.norm2 = torch.nn.LayerNorm(dim, eps=1e-5)
        self.mlp = MLP(dim, hidden)

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        return h + self.mlp(self.norm2(h))


def perturb(module):
    # nn.LayerNorm starts at ones and zeros; move them so the test sees them
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn_like(p))


def block_params(prefix, blk):
    out = {}
    out[f"{prefix}.norm1.weight"] = blk.norm1.weight
    out[f"{prefix}.norm1.bias"] = blk.norm1.bias
    for nm in ("q", "k", "v", "o"):
        linear_params(f"{prefix}.attn.{nm}", getattr(blk.attn, nm), out)
    out[f"{prefix}.norm2.weight"] = blk.norm2.weight
    out[f"{prefix}.norm2.bias"] = blk.norm2.bias
    for nm in ("gate", "up", "down"):
        linear_params(f"{prefix}.mlp.{nm}", getattr(blk.mlp, nm), out)
    return out


def case_block(d):
    dim, heads, hidden, n, b = 16, 2, 24, 6, 2
    blk = Block(dim, heads, hidden)
    perturb(blk)
    x = torch.randn(b, n, dim)
    w = torch.randn(b, n, dim)
    y = blk(x)
    params = block_params("blk", blk)
    t = dict(params)
    t.update({"input.x": x, "input.w": w, "output.y": y})
    t.update(grads_of(params, y, w))
    write(f"{d}/block.gguf", t, {"dim": dim, "heads": heads, "hidden": hidden})


def case_adaln(d):
    dim, cond, n, b = 12, 8, 5, 2
    scale = torch.nn.Linear(cond, dim)
    shift = torch.nn.Linear(cond, dim)
    x = torch.randn(b, n, dim)
    c = torch.randn(b, n, cond)
    w = torch.randn(b, n, dim)
    y = x * (1 + scale(c)) + shift(c)
    params = {}
    linear_params("ada.scale", scale, params)
    linear_params("ada.shift", shift, params)
    t = dict(params)
    t.update({"input.x": x, "input.c": c, "input.w": w, "output.y": y})
    t.update(grads_of(params, y, w))
    write(f"{d}/adaln.gguf", t, {"dim": dim, "cond_dim": cond})


def case_losses(d):
    k, rows_ = 5, 12
    logits = torch.nn.Parameter(torch.randn(rows_, k))
    prob = torch.randn(rows_, k).softmax(-1)
    pred = torch.nn.Parameter(torch.randn(rows_, k))
    target = torch.randn(rows_, k)
    ce = (-(prob * logits.log_softmax(-1)).sum(-1)).mean()
    mse = ((pred - target) ** 2).mean()
    mae = (pred - target).abs().mean()
    g_ce = torch.autograd.grad(ce, logits)[0]
    g_mse = torch.autograd.grad(mse, pred, retain_graph=True)[0]
    g_mae = torch.autograd.grad(mae, pred)[0]
    t = {
        "logits": logits, "pred": pred, "input.prob": prob, "input.target": target,
        "output.ce": ce.reshape(1), "output.mse": mse.reshape(1), "output.mae": mae.reshape(1),
        "grad.ce.logits": g_ce, "grad.mse.pred": g_mse, "grad.mae.pred": g_mae,
    }
    write(f"{d}/losses.gguf", t, {})


def rope(x, pos, n_dims, base, half):
    # x (B, h, n, dh): rotate the leading n_dims of each head
    dh = x.shape[-1]
    i = torch.arange(n_dims // 2, dtype=DT)
    theta = pos[:, None].to(DT) * base ** (-2 * i / n_dims)  # (n, n_dims/2)
    c, s = theta.cos(), theta.sin()
    r, keep = x[..., :n_dims], x[..., n_dims:]
    if half:
        a, b_ = r[..., : n_dims // 2], r[..., n_dims // 2:]
        out = torch.cat([a * c - b_ * s, a * s + b_ * c], -1)
    else:
        a, b_ = r[..., 0::2], r[..., 1::2]
        out = torch.stack([a * c - b_ * s, a * s + b_ * c], -1).flatten(-2)
    return torch.cat([out, keep], -1)


def case_timesfm(d):
    # TimesFM-style attention: split-half rotary positions, RMS normalisation
    # of queries and keys per head (with a learnt scale), a per-dimension
    # query scale softplus(p) * 1.442695 / sqrt(dh), attention scale 1, and
    # a causal mask that also hides each series' padded prefix.
    dim, heads, n, b = 16, 2, 8, 2
    dh = dim // heads
    q_lin, k_lin, v_lin, o_lin = (torch.nn.Linear(dim, dim) for _ in range(4))
    qn = torch.nn.Parameter(1 + 0.1 * torch.randn(dh))
    kn = torch.nn.Parameter(1 + 0.1 * torch.randn(dh))
    ps = torch.nn.Parameter(0.3 * torch.randn(dh))
    x = torch.randn(b, n, dim)
    w = torch.randn(b, n, dim)
    pos = torch.arange(n)
    pad = [0, 3]  # series 2 has 3 padded positions at its start
    mask = torch.zeros(b, n, n)
    for bi in range(b):
        mask[bi] = mask[bi].masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
        mask[bi, :, : pad[bi]] = float("-inf")
        mask[bi, : pad[bi], :] = 0  # padded queries see everything, rows stay finite
        for qi in range(pad[bi]):
            mask[bi, qi] = float("-inf")
            mask[bi, qi, qi] = 0

    def rms(t, g):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * g

    sp = lambda t: t.view(b, n, heads, dh).transpose(1, 2)
    q = rms(rope(sp(q_lin(x)), pos, dh, 10000.0, True), qn)
    k = rms(rope(sp(k_lin(x)), pos, dh, 10000.0, True), kn)
    q = q * (torch.nn.functional.softplus(ps) * 1.442695 / math.sqrt(dh))
    v = sp(v_lin(x))
    s = q @ k.transpose(-1, -2) + mask[:, None]
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(b, n, dim)
    y = o_lin(o)
    params = {}
    for nm, lin in (("q", q_lin), ("k", k_lin), ("v", v_lin), ("o", o_lin)):
        linear_params(f"tfm.{nm}", lin, params)
    params["tfm.q_norm"] = qn
    params["tfm.k_norm"] = kn
    params["tfm.q_scale"] = ps
    t = dict(params)
    # the mask as zulm's softmax takes it: c(n_k, n_q, 1, batch)
    t.update({"input.x": x, "input.w": w, "input.mask": mask[:, None], "output.y": y})
    t.update(grads_of(params, y, w))
    write(f"{d}/timesfm-attention.gguf", t, {"dim": dim, "heads": heads, "eps": 1e-6})


def case_toto(d):
    # Toto-style attention: interleaved rotary positions on half of each
    # head, xPos decay of opposite sign on queries and keys, attention scale
    # 1 / dh, and unit-scaled linears (no bias, divided by sqrt(fan_in)).
    dim, heads, n, b = 16, 2, 8, 2
    dh = dim // heads
    lin = {nm: torch.nn.Linear(dim, dim, bias=False) for nm in ("q", "k", "v", "o")}
    unit = lambda nm, t: lin[nm](t) / math.sqrt(dim)
    x = torch.randn(b, n, dim)
    w = torch.randn(b, n, dim)
    pos = torch.arange(n)
    n_rot = dh // 2
    # xPos: factor per rotated dimension pair and position, q gets f, k 1/f
    i = torch.arange(0, n_rot, 2, dtype=DT)
    zeta = (i + 0.4 * n_rot) / (1.4 * n_rot)  # (n_rot / 2)
    power = (pos.to(DT) - n // 2) / 512.0  # (n)
    fac = zeta[None, :] ** power[:, None]  # (n, n_rot / 2)
    fac = fac.repeat_interleave(2, -1)  # (n, n_rot): both of a pair
    fq = torch.cat([fac, torch.ones(n, dh - n_rot)], -1)  # (n, dh)
    fk = torch.cat([1 / fac, torch.ones(n, dh - n_rot)], -1)
    sp = lambda t: t.view(b, n, heads, dh).transpose(1, 2)
    q = rope(sp(unit("q", x)), pos, n_rot, 10000.0, False) * fq
    k = rope(sp(unit("k", x)), pos, n_rot, 10000.0, False) * fk
    v = sp(unit("v", x))
    s = q @ k.transpose(-1, -2) / dh
    s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(b, n, dim)
    y = unit("o", o)
    params = {f"toto.{nm}.weight": l.weight for nm, l in lin.items()}
    t = dict(params)
    # factors as zulm multiplies them: c(dh, 1, n)
    t.update({
        "input.x": x, "input.w": w,
        "input.fq": fq[:, None, :], "input.fk": fk[:, None, :],
        "output.y": y,
    })
    t.update(grads_of(params, y, w))
    write(f"{d}/toto-attention.gguf", t, {"dim": dim, "heads": heads, "n_rot": n_rot})


if __name__ == "__main__":
    out = sys.argv[1]
    for case in (case_block, case_adaln, case_losses, case_timesfm, case_toto):
        case(out)
    print("wrote fixtures to", out)
