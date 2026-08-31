"""
EmbR ksoft model (PyTorch): per-MM-site MLP → partition energies e_i.

Paper forward path: element-specific fitting MLP → clamp(e_i ≥ 0) → ΔE_ML = Σ e_i.
Legacy ``corr_*``, ``log_c``, ``weights()`` remain for checkpoint compatibility only;
they are not used in the paper ksoft / EmbR training forward path.

Third-party: ``torch`` (neural networks).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


class FittingHead(nn.Module):
    """Small MLP: feature vector → one scalar."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], *, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d_in = int(in_dim)
        for h in hidden:
            layers.append(nn.Linear(d_in, int(h)))
            layers.append(nn.Tanh())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            d_in = int(h)
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


from embr_elements import (
    MMH_ELEM_C,
    MMH_ELEM_CL,
    MMH_ELEM_H,
    MMH_ELEM_K,
    MMH_ELEM_N,
    MMH_ELEM_NA,
    MMH_ELEM_O,
)
# FittingHead defined above


@dataclass(frozen=True)
class SoapE0MmhKModelHyper:
    d_feat: int
    fit_neurons: tuple[int, ...] = (128, 128)
    corr_neurons: tuple[int, ...] = (64, 64)
    dropout: float = 0.0
    corr_mlp_scale: float = 0.25
    use_dist: bool = True
    dist_scale_ang: float = 5.0
    k_min: float = 1e-30


class SoapE0MmhKModel(nn.Module):
    """Site MLP for ΔE; soft ksoft partition (clamp u_j ≥ 0)."""

    def __init__(self, h: SoapE0MmhKModelHyper) -> None:
        super().__init__()
        self.h = h
        drop = float(h.dropout)
        d = int(h.d_feat)
        d_in = d + (1 if h.use_dist else 0)
        e0_hidden = tuple(h.fit_neurons)
        corr_hidden = tuple(h.corr_neurons)

        self.fitting_h = FittingHead(d, e0_hidden, dropout=drop)
        self.fitting_o = FittingHead(d, e0_hidden, dropout=drop)
        self.fitting_na = FittingHead(d, e0_hidden, dropout=drop)
        self.fitting_k = FittingHead(d, e0_hidden, dropout=drop)
        self.fitting_cl = FittingHead(d, e0_hidden, dropout=drop)
        self.fitting_c = FittingHead(d, e0_hidden, dropout=drop)
        self.fitting_n = FittingHead(d, e0_hidden, dropout=drop)

        self.corr_h = FittingHead(d_in, corr_hidden, dropout=drop)  # legacy; unused in paper path
        self.corr_o = FittingHead(d_in, corr_hidden, dropout=drop)
        self.corr_na = FittingHead(d_in, corr_hidden, dropout=drop)
        self.corr_k = FittingHead(d_in, corr_hidden, dropout=drop)
        self.corr_cl = FittingHead(d_in, corr_hidden, dropout=drop)
        self.corr_c = FittingHead(d_in, corr_hidden, dropout=drop)
        self.corr_n = FittingHead(d_in, corr_hidden, dropout=drop)
        self.log_c = nn.Parameter(torch.zeros(7, dtype=torch.float32))  # legacy checkpoint keys

    def _e0_head_for_element(self, elem_id: int) -> FittingHead:
        if int(elem_id) == MMH_ELEM_H:
            return self.fitting_h
        if int(elem_id) == MMH_ELEM_O:
            return self.fitting_o
        if int(elem_id) == MMH_ELEM_NA:
            return self.fitting_na
        if int(elem_id) == MMH_ELEM_K:
            return self.fitting_k
        if int(elem_id) == MMH_ELEM_CL:
            return self.fitting_cl
        if int(elem_id) == MMH_ELEM_C:
            return self.fitting_c
        if int(elem_id) == MMH_ELEM_N:
            return self.fitting_n
        raise ValueError(f"unknown mm_element id {elem_id}")

    def _corr_head_for_element(self, elem_id: int) -> FittingHead:
        if int(elem_id) == MMH_ELEM_H:
            return self.corr_h
        if int(elem_id) == MMH_ELEM_O:
            return self.corr_o
        if int(elem_id) == MMH_ELEM_NA:
            return self.corr_na
        if int(elem_id) == MMH_ELEM_K:
            return self.corr_k
        if int(elem_id) == MMH_ELEM_CL:
            return self.corr_cl
        if int(elem_id) == MMH_ELEM_C:
            return self.corr_c
        if int(elem_id) == MMH_ELEM_N:
            return self.corr_n
        raise ValueError(f"unknown mm_element id {elem_id}")

    def element_scales(self) -> dict[str, float]:
        c = torch.nn.functional.softplus(self.log_c).detach().cpu().numpy()
        return {
            "H": float(c[0]),
            "O": float(c[1]),
            "Na": float(c[2]),
            "K": float(c[3]),
            "Cl": float(c[4]),
            "C": float(c[5]),
            "N": float(c[6]),
        }

    def _c_field(self, mm_element: torch.Tensor) -> torch.Tensor:
        c = torch.nn.functional.softplus(self.log_c)
        out = torch.zeros_like(mm_element, dtype=c.dtype, device=mm_element.device)
        seen = torch.zeros_like(mm_element, dtype=torch.bool)
        for eid, idx in (
            (MMH_ELEM_H, 0),
            (MMH_ELEM_O, 1),
            (MMH_ELEM_NA, 2),
            (MMH_ELEM_K, 3),
            (MMH_ELEM_CL, 4),
            (MMH_ELEM_C, 5),
            (MMH_ELEM_N, 6),
        ):
            mask = mm_element == int(eid)
            seen = seen | mask
            out = torch.where(mask, c[int(idx)], out)
        if bool((~seen).any()):
            raise ValueError("mm_element contains unknown id")
        return out

    def _site_e0_logits(self, feat: torch.Tensor, mm_element: torch.Tensor) -> torch.Tensor:
        """Per-site u_j; sum_j u_j = E0_pred before hard re-partition."""
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            mm_element = mm_element.unsqueeze(0)
        bsz, n_mm, d = feat.shape
        if int(d) != int(self.h.d_feat):
            raise ValueError(f"d_feat={d} != model {self.h.d_feat}")

        if torch.all(mm_element == int(MMH_ELEM_H)):
            flat = feat.reshape(bsz * n_mm, d)
            return self.fitting_h(flat).reshape(bsz, n_mm)

        out = torch.empty((bsz, n_mm), device=feat.device, dtype=feat.dtype)
        for elem_id in (
            MMH_ELEM_H,
            MMH_ELEM_O,
            MMH_ELEM_C,
            MMH_ELEM_N,
            MMH_ELEM_NA,
            MMH_ELEM_K,
            MMH_ELEM_CL,
        ):
            mask = mm_element == int(elem_id)
            if not bool(mask.any()):
                continue
            idx_b, idx_j = torch.where(mask)
            out[idx_b, idx_j] = self._e0_head_for_element(elem_id)(feat[idx_b, idx_j])
        return out

    def _eps_field(
        self,
        feat: torch.Tensor,
        mm_element: torch.Tensor,
        dist_qm: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, n_mm, d = feat.shape
        scale = float(self.h.corr_mlp_scale)
        if self.h.use_dist:
            if dist_qm is None:
                raise ValueError("use_dist=True but dist_qm is None")
            dist_n = dist_qm / float(self.h.dist_scale_ang)
            x = torch.cat([feat, dist_n.unsqueeze(-1)], dim=-1)
        else:
            x = feat

        if torch.all(mm_element == int(MMH_ELEM_H)):
            raw = self.corr_h(x.reshape(bsz * n_mm, -1)).reshape(bsz, n_mm)
            return scale * torch.tanh(raw)

        out = torch.zeros((bsz, n_mm), device=feat.device, dtype=feat.dtype)
        for elem_id in (
            MMH_ELEM_H,
            MMH_ELEM_O,
            MMH_ELEM_C,
            MMH_ELEM_N,
            MMH_ELEM_NA,
            MMH_ELEM_K,
            MMH_ELEM_CL,
        ):
            mask = mm_element == int(elem_id)
            if not bool(mask.any()):
                continue
            idx_b, idx_j = torch.where(mask)
            raw = self._corr_head_for_element(elem_id)(x[idx_b, idx_j])
            out[idx_b, idx_j] = scale * torch.tanh(raw)
        return out

    def weights(
        self,
        feat: torch.Tensor,
        mm_element: torch.Tensor,
        kernel: torch.Tensor,
        *,
        dist_qm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (w, eps) with w_j = c · k · (1+eps). Used only in hard partition."""
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            mm_element = mm_element.unsqueeze(0)
            kernel = kernel.unsqueeze(0)
            if dist_qm is not None:
                dist_qm = dist_qm.unsqueeze(0)
        k = torch.clamp(kernel.abs(), min=float(self.h.k_min))
        eps = self._eps_field(feat, mm_element, dist_qm)
        c = self._c_field(mm_element)
        w = c * k * (1.0 + eps)
        return torch.clamp(w, min=float(self.h.k_min)), eps

    def per_site_energies(
        self,
        feat: torch.Tensor,
        mm_element: torch.Tensor,
        kernel: torch.Tensor | None = None,
        *,
        dist_qm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, n) site energies e_i [kcal/mol]; negative logits are clipped to zero."""
        u = self._site_e0_logits(feat, mm_element)
        return torch.clamp(u, min=0.0)

    def forward_descriptors(
        self,
        feat: torch.Tensor,
        mm_element: torch.Tensor,
        kernel: torch.Tensor | None = None,
        *,
        dist_qm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.per_site_energies(
            feat, mm_element, kernel, dist_qm=dist_qm
        ).sum(dim=1)


# --- training losses ---

from embr_envelope import is_active_repulsion_mmh_element


def pearson_per_row(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Pearson r for each batch row; shape (B,). Returns nan rows when undefined."""
    if x.shape != y.shape or x.dim() != 2:
        raise ValueError(f"need matching (B,n), got {tuple(x.shape)} vs {tuple(y.shape)}")
    xm = x - x.mean(dim=1, keepdim=True)
    ym = y - y.mean(dim=1, keepdim=True)
    num = (xm * ym).sum(dim=1)
    den = torch.sqrt((xm * xm).sum(dim=1) * (ym * ym).sum(dim=1) + eps)
    r = num / den
    vx = (xm * xm).sum(dim=1)
    vy = (ym * ym).sum(dim=1)
    bad = (vx < eps) | (vy < eps)
    return torch.where(bad, torch.full_like(r, float("nan")), r)


def weighted_pearson_per_row(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted Pearson r per batch row (B,)."""
    if x.shape != y.shape or x.shape != w.shape or x.dim() != 2:
        raise ValueError(
            f"need matching (B,n) for x,y,w; got {tuple(x.shape)} {tuple(y.shape)} {tuple(w.shape)}"
        )
    ww = torch.clamp(w, min=0.0)
    wsum = ww.sum(dim=1, keepdim=True).clamp(min=eps)
    xbar = (ww * x).sum(dim=1, keepdim=True) / wsum
    ybar = (ww * y).sum(dim=1, keepdim=True) / wsum
    xm = x - xbar
    ym = y - ybar
    num = (ww * xm * ym).sum(dim=1)
    den = torch.sqrt(
        (ww * xm * xm).sum(dim=1) * (ww * ym * ym).sum(dim=1) + eps
    )
    r = num / den
    n_eff = (ww > 0).sum(dim=1)
    vx = (ww * xm * xm).sum(dim=1)
    vy = (ww * ym * ym).sum(dim=1)
    bad = (n_eff < 2) | (vx < eps) | (vy < eps)
    return torch.where(bad, torch.full_like(r, float("nan")), r)


def mean_pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r = pearson_per_row(x, y)
    ok = torch.isfinite(r)
    if not bool(ok.any()):
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return r[ok].mean()


def mean_weighted_pearson(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r = weighted_pearson_per_row(x, y, w)
    ok = torch.isfinite(r)
    if not bool(ok.any()):
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return r[ok].mean()


def normalize_e_sites(e_sites: str | None) -> str:
    """all | positive (= no e on O/Cl)."""
    s = str(e_sites or "all").strip().lower()
    if s in ("all", "", "partition_all"):
        return "all"
    if s in ("positive", "no_ocl", "cations"):
        return "positive"
    if s in ("h_only", "no_o"):
        return s
    raise ValueError(
        f"e_sites must be all|positive (or h_only|no_o), got {e_sites!r}"
    )


def e_sites_active_mask(
    mm_element: torch.Tensor,
    e_sites: str | None,
) -> torch.Tensor:
    """Bool mask (same shape as mm_element): True = site may carry e_i."""
    pol = normalize_e_sites(e_sites)
    if pol == "all":
        return torch.ones_like(mm_element, dtype=torch.bool)
    # Vectorized for the common positive case
    if pol == "positive":
        from embr_elements import MMH_ELEM_CL, MMH_ELEM_O

        return (mm_element != int(MMH_ELEM_O)) & (mm_element != int(MMH_ELEM_CL))
    # Fallback: per-id via shared repulsion policy names
    out = torch.zeros_like(mm_element, dtype=torch.bool)
    for eid in torch.unique(mm_element):
        eid_i = int(eid.item())
        if is_active_repulsion_mmh_element(eid_i, policy=pol):
            out = out | (mm_element == eid_i)
    return out


def apply_e_sites_mask(
    e_j: torch.Tensor,
    mm_element: torch.Tensor,
    e_sites: str | None,
) -> torch.Tensor:
    """Zero e_j on inactive sites (O/Cl when e_sites=positive)."""
    pol = normalize_e_sites(e_sites)
    if pol == "all":
        return e_j
    m = e_sites_active_mask(mm_element, pol).to(dtype=e_j.dtype)
    return e_j * m


def mean_pearson_e_k(
    e_j: torch.Tensor,
    kernel: torch.Tensor,
    *,
    mm_element: torch.Tensor | None = None,
    e_sites: str | None = None,
) -> torch.Tensor:
    """Mean per-frame Pearson(e, |k|); optionally only on e_sites-active sites."""
    k = torch.abs(kernel)
    pol = normalize_e_sites(e_sites)
    if pol == "all" or mm_element is None:
        return mean_pearson(e_j, k)
    w = e_sites_active_mask(mm_element, pol).to(dtype=e_j.dtype)
    return mean_weighted_pearson(e_j, k, w)


def correlation_deficit_loss(
    e_j: torch.Tensor,
    kernel: torch.Tensor,
    *,
    target: float,
    mm_element: torch.Tensor | None = None,
    e_sites: str | None = None,
) -> torch.Tensor:
    """Mean relu(target - r); zero when pooled r >= target."""
    r = mean_pearson_e_k(e_j, kernel, mm_element=mm_element, e_sites=e_sites)
    return torch.relu(torch.as_tensor(float(target), device=e_j.device, dtype=e_j.dtype) - r)



def _resolve_d_feat(cache: dict, ck: dict) -> int:
    cache_d = int(cache["feat_h_flat"].shape[1])
    d_feat = int(ck.get("d_feat", cache_d))
    if d_feat != cache_d:
        raise ValueError(f"ckpt d_feat={d_feat} != cache feature dim {cache_d}")
    return d_feat


def build_mmh_model(
    *,
    model_kind: str,
    d_feat: int,
    fit_neurons: tuple[int, ...],
    dropout: float,
    corr_neurons: tuple[int, ...] = (64, 64),
    corr_mlp_scale: float = 0.25,
    use_dist: bool = True,
    dist_scale_ang: float = 5.0,
) -> nn.Module:
    kind = str(model_kind).lower()
    # Backward compatibility: early EmbR checkpoints used the model tag
    # ``soap_e0_mix_mmh_k`` for the same ksoft architecture.  Keep accepting
    # that metadata tag; state-dict/dimension checks below still guard against
    # genuinely incompatible checkpoints.
    if kind not in ("ksoft", "soap_e0_mix_mmh_ksoft", "soap_e0_mix_mmh_k"):
        raise ValueError(f"embr supports ksoft only, got {model_kind!r}")
    if kind == "soap_e0_mix_mmh_k":
        print("[embr_model] legacy checkpoint tag 'soap_e0_mix_mmh_k' → ksoft", flush=True)
    return SoapE0MmhKModel(
        SoapE0MmhKModelHyper(
            d_feat=d_feat,
            fit_neurons=fit_neurons,
            corr_neurons=corr_neurons,
            dropout=dropout,
            corr_mlp_scale=float(corr_mlp_scale),
            use_dist=bool(use_dist),
            dist_scale_ang=float(dist_scale_ang),
        )
    )


def _upgrade_k_state_dict_5_to_7(state: dict, model: nn.Module) -> dict:
    sd = dict(state)
    log_c = sd.get("log_c")
    if log_c is None:
        return sd
    log_c_t = torch.as_tensor(log_c)
    target = model.state_dict()["log_c"]
    if int(log_c_t.numel()) == int(target.numel()):
        return sd
    if int(log_c_t.numel()) != 5 or int(target.numel()) != 7:
        raise RuntimeError(f"cannot upgrade log_c {tuple(log_c_t.shape)} → {tuple(target.shape)}")
    padded = torch.zeros(7, dtype=log_c_t.dtype, device=log_c_t.device)
    padded[:5] = log_c_t.reshape(5)
    sd["log_c"] = padded
    print("[embr_model] upgraded k-model state_dict: log_c 5→7", flush=True)
    return sd


def load_mmh_model_from_ckpt(
    ckpt_path: Path,
    cache: dict,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)
    d_feat = _resolve_d_feat(cache, ck)
    model_kind = str(ck.get("model", "soap_e0_mix_mmh_ksoft"))
    model = build_mmh_model(
        model_kind=model_kind,
        d_feat=d_feat,
        fit_neurons=tuple(int(x) for x in ck["fit_neurons"]),
        dropout=float(ck.get("dropout", 0.0)),
        corr_neurons=tuple(int(x) for x in ck.get("corr_neurons", (64, 64))),
        corr_mlp_scale=float(ck.get("corr_mlp_scale", 0.25)),
        use_dist=bool(ck.get("use_dist", True)),
        dist_scale_ang=float(ck.get("dist_scale_ang", 5.0)),
    ).to(device)
    state = ck["state_dict"]
    if isinstance(model, SoapE0MmhKModel):
        state = _upgrade_k_state_dict_5_to_7(state, model)
        missing, unexpected = model.load_state_dict(state, strict=False)
        miss = [
            k for k in missing
            if not (
                k.startswith("fitting_c.") or k.startswith("fitting_n.")
                or k.startswith("corr_c.") or k.startswith("corr_n.")
            )
        ]
        if miss or unexpected:
            raise RuntimeError(f"ckpt load incomplete: missing={miss} unexpected={list(unexpected)}")
    else:
        model.load_state_dict(state)
    try:
        e_sites = normalize_e_sites(ck.get("e_sites", ck.get("repulsion_policy", "all")))
    except ValueError:
        e_sites = "all"
    setattr(model, "e_sites", e_sites)
    model.eval()
    return model, ck


def forward_e_j(
    model: nn.Module,
    *,
    feat_t: torch.Tensor,
    el_t: torch.Tensor,
    ker_t: torch.Tensor | None,
    e0_t: torch.Tensor | None,
    dist_t: torch.Tensor | None,
) -> torch.Tensor:
    e_sites = str(getattr(model, "e_sites", "all") or "all")
    if isinstance(model, SoapE0MmhKModel):
        e_j = model.per_site_energies(feat_t, el_t, ker_t, dist_qm=dist_t)
        return apply_e_sites_mask(e_j, el_t, e_sites)
    raise TypeError(f"model {type(model).__name__} has no per-site e_j outputs")
