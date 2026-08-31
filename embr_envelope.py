"""
Per-element repulsion envelope parameters for mix_mmh kernel k_j on MM sites.

Kernel (precompute, from Emb0 ρ on grid)::

    k_j = ∫ ρ_emb0(r) · C_j · envelope_j(|r-R_j|) dr

``envelope_kind``::

    ``gauss``:  envelope = exp(-α |r-R|²)     α [Bohr⁻²] in ``fix_alpha_by_element``
    ``exp``:    envelope = exp(-ζ |r-R|)       ζ [Å⁻¹] in ``fix_alpha_by_element``

Optional multi-term envelope (manifest ``exp_sum_by_element``)::

    Gaussian in r [Å]:  amp · exp(-½((r-μ)/σ)²)     (preferred for ρ̄ curve fit)
    Slater (legacy):    amp · r^pow · exp(-ζ r)

Legacy ``{"zeta","lnC"}`` → Slater with pow=0.

EmbR repulsion potential (separate A_j from ML)::

    V_rep = Σ_J A_J · C_J · envelope_J(|r-R_J|)

``A_j = e_j / (627.5 · k_j)`` unchanged; C_j is **not** absorbed into A_j.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

try:
    from pyscf import lib

    _BOHR_ANG = float(lib.param.BOHR)
except Exception:  # pragma: no cover
    _BOHR_ANG = 0.529177210903

# mm_element_k codes in ref npz / rho_kernels (O/H/C/N/Na/K/Cl)
MM_KERNEL_EL_O = 0
MM_KERNEL_EL_H = 1
MM_KERNEL_EL_NA = 2
MM_KERNEL_EL_K = 3
MM_KERNEL_EL_CL = 4
MM_KERNEL_EL_C = 5
MM_KERNEL_EL_N = 6

MM_KERNEL_EL_NAMES: dict[int, str] = {
    MM_KERNEL_EL_O: "O",
    MM_KERNEL_EL_H: "H",
    MM_KERNEL_EL_NA: "Na",
    MM_KERNEL_EL_K: "K",
    MM_KERNEL_EL_CL: "Cl",
    MM_KERNEL_EL_C: "C",
    MM_KERNEL_EL_N: "N",
}

DEFAULT_R_ANG: dict[str, float] = {
    "H": 0.35,
    "C": 0.77,
    "N": 0.71,
    "Na": 1.02,
    "K": 1.38,
    "Cl": 0.99,
    "O": 0.66,
}

ION_MM_KEYS = frozenset({"Na", "K", "Cl"})
ACTIVE_REPULSION_SYMBOLS = frozenset({"H", "O", "C", "N", "Na", "K", "Cl"})

REPULSION_POLICY_ALL = "all"
REPULSION_POLICY_H_ONLY = "h_only"
REPULSION_POLICY_NO_O = "no_o"
# Positive MM sites only: H / Na / K / C / N — O and Cl get e=0 / A=0.
REPULSION_POLICY_POSITIVE = "positive"

ENVELOPE_GAUSS = "gauss"
ENVELOPE_EXP = "exp"
_ELEMENT_KEYS = ("H", "O", "C", "N", "Na", "K", "Cl")


def _norm_elem_key(sym: str) -> str:
    el = str(sym).strip()
    if not el:
        raise ValueError("empty MM symbol")
    u = el.upper()
    if u.startswith("NA"):
        return "Na"
    if u.startswith("CL"):
        return "Cl"
    if u == "K" or (u.startswith("K") and len(u) <= 2):
        return "K"
    c = el[0].upper()
    if c == "H":
        return "H"
    if c == "O":
        return "O"
    if c == "C":
        return "C"
    if c == "N":
        return "N"
    return c


def is_active_repulsion_symbol(sym: str, *, policy: str = REPULSION_POLICY_ALL) -> bool:
    return is_repulsion_symbol(sym, policy=policy)


def is_repulsion_symbol(sym: str, *, policy: str = REPULSION_POLICY_ALL) -> bool:
    key = _norm_elem_key(sym)
    pol = str(policy).strip().lower()
    if pol in (REPULSION_POLICY_ALL, "partition_all", ""):
        return key in ACTIVE_REPULSION_SYMBOLS
    if pol == REPULSION_POLICY_H_ONLY:
        return key == "H"
    if pol == REPULSION_POLICY_NO_O:
        return key in {"H", "C", "N", "Na", "K", "Cl"}
    if pol in (REPULSION_POLICY_POSITIVE, "no_ocl", "cations"):
        return key in {"H", "C", "N", "Na", "K"}
    raise ValueError(
        f"unknown repulsion policy {policy!r} (use all, h_only, no_o, positive)"
    )


def is_kernel_mm_symbol(sym: str) -> bool:
    return _norm_elem_key(sym) in {"O", "H", "C", "N", "Na", "K", "Cl"}


def mm_kernel_element_code(sym: str) -> int:
    key = _norm_elem_key(sym)
    if key == "O":
        return MM_KERNEL_EL_O
    if key == "H":
        return MM_KERNEL_EL_H
    if key == "Na":
        return MM_KERNEL_EL_NA
    if key == "K":
        return MM_KERNEL_EL_K
    if key == "Cl":
        return MM_KERNEL_EL_CL
    if key == "C":
        return MM_KERNEL_EL_C
    if key == "N":
        return MM_KERNEL_EL_N
    raise ValueError(f"unsupported MM kernel site {sym!r} (expected O/H/C/N/Na/K/Cl)")


def is_active_kernel_element(el: int) -> bool:
    return int(el) in (MM_KERNEL_EL_H, MM_KERNEL_EL_NA, MM_KERNEL_EL_K, MM_KERNEL_EL_CL)


def is_partition_kernel_element(el: int) -> bool:
    return int(el) in (
        MM_KERNEL_EL_O,
        MM_KERNEL_EL_H,
        MM_KERNEL_EL_C,
        MM_KERNEL_EL_N,
        MM_KERNEL_EL_NA,
        MM_KERNEL_EL_K,
        MM_KERNEL_EL_CL,
    )


def is_active_repulsion_mmh_element(el: int, *, policy: str = REPULSION_POLICY_ALL) -> bool:
    from embr_elements import (
        MMH_ELEM_C,
        MMH_ELEM_CL,
        MMH_ELEM_H,
        MMH_ELEM_K,
        MMH_ELEM_N,
        MMH_ELEM_NA,
        MMH_ELEM_O,
    )

    e = int(el)
    pol = str(policy).strip().lower()
    if pol in (REPULSION_POLICY_ALL, "partition_all", ""):
        return e in (
            MMH_ELEM_H,
            MMH_ELEM_O,
            MMH_ELEM_C,
            MMH_ELEM_N,
            MMH_ELEM_NA,
            MMH_ELEM_K,
            MMH_ELEM_CL,
        )
    if pol == REPULSION_POLICY_H_ONLY:
        return e == MMH_ELEM_H
    if pol == REPULSION_POLICY_NO_O:
        return e in (MMH_ELEM_H, MMH_ELEM_C, MMH_ELEM_N, MMH_ELEM_NA, MMH_ELEM_K, MMH_ELEM_CL)
    if pol in (REPULSION_POLICY_POSITIVE, "no_ocl", "cations"):
        return e in (MMH_ELEM_H, MMH_ELEM_C, MMH_ELEM_N, MMH_ELEM_NA, MMH_ELEM_K)
    raise ValueError(
        f"unknown repulsion policy {policy!r} (use all, h_only, no_o, positive)"
    )


@dataclass(frozen=True)
class ExpSumTerm:
    """One term in ``exp_sum_by_element`` (Gaussian-in-r or Slater)."""

    amp: float
    mu: float | None = None
    sigma: float | None = None
    zeta: float | None = None
    pow: int = 0

    @property
    def is_gauss_r(self) -> bool:
        return self.mu is not None and self.sigma is not None

    def eval(self, r_ang: np.ndarray) -> np.ndarray:
        r = np.maximum(np.asarray(r_ang, dtype=np.float64), 0.0)
        if self.is_gauss_r:
            sig = max(float(self.sigma), 1e-8)
            return float(self.amp) * np.exp(-0.5 * ((r - float(self.mu)) / sig) ** 2)
        if self.zeta is None:
            raise ValueError("ExpSumTerm needs mu/sigma or zeta")
        return slater_envelope_factor(
            r, zeta=float(self.zeta), amp=float(self.amp), pow=int(self.pow)
        )

    def to_manifest_dict(self) -> dict[str, float | int]:
        if self.is_gauss_r:
            return {
                "amp": float(self.amp),
                "mu": float(self.mu),
                "sigma": float(self.sigma),
            }
        out: dict[str, float | int] = {"zeta": float(self.zeta), "amp": float(self.amp)}
        if int(self.pow) != 0:
            out["pow"] = int(self.pow)
        return out


def gauss_r_factor(
    r_ang: np.ndarray,
    *,
    amp: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """amp · exp(-½((r-μ)/σ)²), r [Å]."""
    r = np.asarray(r_ang, dtype=np.float64)
    sig = max(float(sigma), 1e-8)
    return float(amp) * np.exp(-0.5 * ((r - float(mu)) / sig) ** 2)


def slater_envelope_factor(
    r_ang: np.ndarray,
    *,
    zeta: float,
    amp: float,
    pow: int = 0,
) -> np.ndarray:
    """amp · r^pow · exp(-ζ r); r [Å], pow≥0 integer."""
    r = np.maximum(np.asarray(r_ang, dtype=np.float64), 0.0)
    return float(amp) * np.power(r, float(int(pow))) * np.exp(-float(zeta) * r)


def _parse_exp_sum_term(t: Mapping[str, Any] | tuple | list) -> ExpSumTerm:
    if isinstance(t, Mapping):
        if "mu" in t and "sigma" in t:
            amp = float(t["amp"]) if "amp" in t else float(math.exp(float(t["lnC"])))
            return ExpSumTerm(amp=amp, mu=float(t["mu"]), sigma=float(t["sigma"]))
        zeta = float(t["zeta"])
        pow_i = int(t.get("pow", 0))
        if "amp" in t:
            return ExpSumTerm(amp=float(t["amp"]), zeta=zeta, pow=pow_i)
        if "lnC" in t:
            return ExpSumTerm(amp=float(math.exp(float(t["lnC"]))), zeta=zeta, pow=pow_i)
        raise ValueError(f"exp_sum term needs amp or lnC: {dict(t)!r}")
    if len(t) == 2:
        z, val = t
        return ExpSumTerm(amp=float(val), zeta=float(z), pow=0)
    if len(t) == 3:
        z, val, p = t
        return ExpSumTerm(amp=float(val), zeta=float(z), pow=int(p))
    raise ValueError(f"bad exp_sum term tuple length {len(t)}")


def _parse_exp_sum_by_element(
    raw: Mapping[str, Any] | None,
) -> dict[str, tuple[ExpSumTerm, ...]]:
    if raw is None:
        return {}
    out: dict[str, tuple[ExpSumTerm, ...]] = {}
    for sym, terms in raw.items():
        key = _norm_elem_key(str(sym))
        parsed = tuple(_parse_exp_sum_term(x) for x in terms)
        if parsed:
            out[key] = parsed
    return out


def _exp_sum_term_close(a: ExpSumTerm, b: ExpSumTerm, *, tol: float) -> bool:
    if abs(float(a.amp) - float(b.amp)) > tol:
        return False
    if a.is_gauss_r and b.is_gauss_r:
        return (
            abs(float(a.mu) - float(b.mu)) <= tol
            and abs(float(a.sigma) - float(b.sigma)) <= tol
        )
    if float(a.zeta or 0.0) != float(b.zeta or 0.0):
        if abs(float(a.zeta or 0.0) - float(b.zeta or 0.0)) > tol:
            return False
    return int(a.pow) == int(b.pow)


def _fill_element_defaults(
    width: dict[str, float],
    lnC: dict[str, float],
    *,
    default_width: float,
) -> tuple[dict[str, float], dict[str, float]]:
    w = {str(k): float(width.get(k, default_width)) for k in _ELEMENT_KEYS}
    if "H" not in width:
        w["H"] = float(default_width)
    for key in _ELEMENT_KEYS:
        if key not in w:
            w[key] = float(w.get("H", default_width))
        if key not in lnC:
            lnC[key] = float(lnC.get(key, 0.0))
    return w, {str(k): float(lnC[k]) for k in _ELEMENT_KEYS}


@dataclass(frozen=True)
class MmhEnvelopeConfig:
    """
    Repulsion envelope + amplitude C per MM element.

    ``width_by_element``:
      gauss → α [Bohr⁻²]
      exp   → ζ [Å⁻¹]
    """

    kind: str
    width_by_element: dict[str, float]
    lnC_by_element: dict[str, float]
    exp_sum_by_element: dict[str, tuple[ExpSumTerm, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in (ENVELOPE_GAUSS, ENVELOPE_EXP):
            raise ValueError(f"envelope_kind must be gauss|exp, got {self.kind!r}")
        object.__setattr__(self, "kind", kind)

    @classmethod
    def uniform(cls, width_h: float, *, kind: str = ENVELOPE_GAUSS) -> MmhEnvelopeConfig:
        w = float(width_h)
        return cls(
            kind=str(kind),
            width_by_element={k: w for k in _ELEMENT_KEYS},
            lnC_by_element={k: 0.0 for k in _ELEMENT_KEYS},
        )

    @classmethod
    def approach_a(
        cls,
        alpha_h: float,
        *,
        r_ang: Mapping[str, float] | None = None,
        kind: str = ENVELOPE_GAUSS,
    ) -> MmhEnvelopeConfig:
        r = dict(DEFAULT_R_ANG)
        if r_ang is not None:
            r.update({str(k): float(v) for k, v in r_ang.items()})
        ah = float(alpha_h)
        rh = float(r["H"])
        out = {"H": ah, "O": ah}
        for key in ("Na", "K", "Cl", "C", "N"):
            re = float(r[key])
            out[key] = ah * (rh / re) ** 2
        return cls(kind=str(kind), width_by_element=out, lnC_by_element={k: 0.0 for k in _ELEMENT_KEYS})

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], **kw) -> MmhEnvelopeConfig:
        """Alias for :meth:`from_width_mapping` (legacy ``MmhAlphaConfig.from_mapping``)."""
        return cls.from_width_mapping(raw, **kw)

    @classmethod
    def from_width_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        kind: str = ENVELOPE_GAUSS,
        lnC: Mapping[str, Any] | None = None,
    ) -> MmhEnvelopeConfig:
        out: dict[str, float] = {}
        for k, v in raw.items():
            out[_norm_elem_key(str(k))] = float(v)
        if "H" not in out:
            raise ValueError("width mapping must include H")
        if "O" not in out:
            out["O"] = float(out["H"])
        for key in ("Na", "K", "Cl", "C", "N"):
            if key not in out:
                out[key] = float(out["H"])
        ln_out: dict[str, float] = {}
        if lnC is not None:
            for k, v in lnC.items():
                ln_out[_norm_elem_key(str(k))] = float(v)
        w, ln_f = _fill_element_defaults(out, ln_out, default_width=float(out["H"]))
        return cls(kind=str(kind), width_by_element=w, lnC_by_element=ln_f)

    @classmethod
    def parse(
        cls,
        *,
        fix_alpha: float | None = 3.0,
        multi_alpha: bool = False,
        alpha_by_element: str | None = None,
        alpha_na: float | None = None,
        alpha_k: float | None = None,
        envelope_kind: str | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> MmhEnvelopeConfig:
        kind = ENVELOPE_GAUSS
        lnC_raw: dict[str, Any] | None = None
        exp_sum_raw: dict[str, tuple[ExpSumTerm, ...]] | None = None
        if manifest is not None:
            kind = str(manifest.get("envelope_kind", kind)).strip().lower()
            if manifest.get("lnC_by_element") is not None:
                lnC_raw = dict(manifest["lnC_by_element"])
            exp_sum_raw = _parse_exp_sum_by_element(manifest.get("exp_sum_by_element"))
        if envelope_kind is not None:
            kind = str(envelope_kind).strip().lower()

        def _with_exp_sum(cfg: MmhEnvelopeConfig) -> MmhEnvelopeConfig:
            if not exp_sum_raw:
                return cfg
            return MmhEnvelopeConfig(
                kind=cfg.kind,
                width_by_element=dict(cfg.width_by_element),
                lnC_by_element=dict(cfg.lnC_by_element),
                exp_sum_by_element=dict(exp_sum_raw),
            )

        if alpha_by_element is not None and str(alpha_by_element).strip():
            return _with_exp_sum(
                cls.from_width_mapping(json.loads(str(alpha_by_element)), kind=kind, lnC=lnC_raw)
            )
        if manifest is not None and manifest.get("fix_alpha_by_element") is not None:
            return _with_exp_sum(
                cls.from_width_mapping(manifest["fix_alpha_by_element"], kind=kind, lnC=lnC_raw)
            )

        ah = float(3.0 if fix_alpha is None else fix_alpha)
        if multi_alpha or alpha_na is not None or alpha_k is not None:
            cfg = cls.approach_a(ah, kind=kind)
            by = dict(cfg.width_by_element)
            if alpha_na is not None:
                by["Na"] = float(alpha_na)
            if alpha_k is not None:
                by["K"] = float(alpha_k)
            ln_map = {k: 0.0 for k in _ELEMENT_KEYS}
            if lnC_raw:
                for k, v in lnC_raw.items():
                    ln_map[_norm_elem_key(str(k))] = float(v)
            return _with_exp_sum(cls(kind=kind, width_by_element=by, lnC_by_element=ln_map))
        ln_map = {k: 0.0 for k in _ELEMENT_KEYS}
        if lnC_raw:
            for k, v in lnC_raw.items():
                ln_map[_norm_elem_key(str(k))] = float(v)
        return _with_exp_sum(
            cls(kind=kind, width_by_element={k: ah for k in _ELEMENT_KEYS}, lnC_by_element=ln_map)
        )

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any] | None) -> MmhEnvelopeConfig:
        if meta is None:
            return cls.uniform(3.0)
        return cls.parse(
            fix_alpha=float(meta.get("fix_alpha", 3.0)),
            multi_alpha=bool(meta.get("multi_alpha", False)),
            envelope_kind=meta.get("envelope_kind"),
            manifest=meta,
        )

    def width_for_symbol(self, sym: str) -> float:
        key = _norm_elem_key(sym)
        if key not in self.width_by_element:
            raise KeyError(f"no width for element {sym!r}")
        return float(self.width_by_element[key])

    def lnC_for_symbol(self, sym: str) -> float:
        key = _norm_elem_key(sym)
        return float(self.lnC_by_element.get(key, 0.0))

    def C_for_symbol(self, sym: str) -> float:
        if self.has_exp_sum(sym):
            return 1.0
        return float(math.exp(self.lnC_for_symbol(sym)))

    def has_exp_sum(self, sym: str) -> bool:
        key = _norm_elem_key(sym)
        terms = self.exp_sum_by_element.get(key)
        return bool(terms)

    def exp_sum_terms(self, sym: str) -> tuple[ExpSumTerm, ...]:
        key = _norm_elem_key(sym)
        return tuple(self.exp_sum_by_element.get(key, ()))

    def alpha_for_symbol(self, sym: str) -> float:
        """Legacy name: width parameter (α or ζ depending on kind)."""
        return self.width_for_symbol(sym)

    def alpha_for_kernel_element(self, el: int) -> float:
        name = MM_KERNEL_EL_NAMES.get(int(el))
        if name is None:
            raise KeyError(f"unknown mm_element code {el}")
        return self.width_for_symbol(name)

    def envelope_on_grid(
        self,
        sym: str,
        *,
        r_bohr: np.ndarray,
        r2_bohr: np.ndarray,
    ) -> np.ndarray:
        """Pointwise envelope factor (includes C), for Lebedev grid distances."""
        if self.kind == ENVELOPE_EXP:
            terms = self.exp_sum_terms(sym)
            r_ang = np.maximum(np.asarray(r_bohr, dtype=np.float64), 0.0) * _BOHR_ANG
            if terms:
                kern = np.zeros_like(r_ang, dtype=np.float64)
                for term in terms:
                    kern += term.eval(r_ang)
                return kern
            Cj = self.C_for_symbol(sym)
            zeta_ang = self.width_for_symbol(sym)
            return Cj * np.exp(-zeta_ang * r_ang)
        Cj = self.C_for_symbol(sym)
        alpha = self.width_for_symbol(sym)
        return Cj * np.exp(-alpha * np.asarray(r2_bohr, dtype=np.float64))

    def is_uniform(self, *, tol: float = 1e-8) -> bool:
        """Legacy alias for :meth:`is_uniform_width`."""
        return self.is_uniform_width(tol=tol)

    def is_uniform_width(self, *, tol: float = 1e-8) -> bool:
        vals = list(self.width_by_element.values())
        if not vals:
            return True
        ref = float(vals[0])
        return all(abs(float(v) - ref) <= tol for v in vals)

    def is_legacy_gaussian(self) -> bool:
        if self.kind != ENVELOPE_GAUSS:
            return False
        return all(abs(float(v)) <= 1e-30 for v in self.lnC_by_element.values())

    def legacy_fix_alpha(self) -> float:
        return float(self.width_by_element["H"])

    def sidecar_tag(self) -> str:
        kind_tag = "g" if self.kind == ENVELOPE_GAUSS else "e"
        w_parts = [f"{k}{self.width_by_element[k]:g}".replace(".", "p") for k in sorted(self.width_by_element)]
        c_parts = [
            f"{k}{self.lnC_by_element[k]:g}".replace(".", "p")
            for k in sorted(self.lnC_by_element)
            if abs(float(self.lnC_by_element[k])) > 1e-12
        ]
        base = f"{kind_tag}_" + "_".join(w_parts)
        if c_parts:
            base += "_lnC_" + "_".join(c_parts)
        if self.exp_sum_by_element:
            exp_blob = {
                str(sym): [t.to_manifest_dict() for t in terms]
                for sym, terms in sorted(self.exp_sum_by_element.items())
            }
            raw = json.dumps(exp_blob, sort_keys=True, separators=(",", ":"))
            base += "_es" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        return base

    def width_meta(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in sorted(self.width_by_element.items())}

    def to_meta(self) -> dict[str, float]:
        return self.width_meta()

    def to_full_meta(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "envelope_kind": self.kind,
            "fix_alpha": self.legacy_fix_alpha(),
            "fix_alpha_by_element": self.width_meta(),
            "lnC_by_element": {str(k): float(v) for k, v in sorted(self.lnC_by_element.items())},
        }
        if self.exp_sum_by_element:
            out["exp_sum_by_element"] = {
                str(sym): [t.to_manifest_dict() for t in terms]
                for sym, terms in sorted(self.exp_sum_by_element.items())
            }
        return out

    def format_log(self) -> str:
        """One-line envelope snapshot for CLI logs."""
        parts = [
            f"envelope={self.kind}",
            f"width={self.width_meta()}",
            f"lnC={self.lnC_by_element}",
        ]
        if self.exp_sum_by_element:
            n_terms = sum(len(t) for t in self.exp_sum_by_element.values())
            syms = ",".join(sorted(self.exp_sum_by_element))
            parts.append(
                f"exp_sum=({syms}; n_terms={n_terms}; overrides single-term ζ/lnC on those elements)"
            )
        return "  ".join(parts)

    def matches_stored(
        self,
        *,
        fix_alpha_bohr2: float | None,
        fix_alpha_by_element: Mapping[str, Any] | None,
        envelope_kind: str | None = None,
        lnC_by_element: Mapping[str, Any] | None = None,
        exp_sum_by_element: Mapping[str, Any] | None = None,
        tol: float = 1e-6,
    ) -> bool:
        if envelope_kind is not None and str(envelope_kind).strip().lower() != self.kind:
            return False
        stored_exp = _parse_exp_sum_by_element(exp_sum_by_element)
        if self.exp_sum_by_element or stored_exp:
            if set(self.exp_sum_by_element.keys()) != set(stored_exp.keys()):
                return False
            for sym, terms in self.exp_sum_by_element.items():
                other = stored_exp.get(sym)
                if other is None or len(other) != len(terms):
                    return False
                for a, b in zip(terms, other):
                    if a.is_gauss_r != b.is_gauss_r:
                        return False
                    if not _exp_sum_term_close(a, b, tol=tol):
                        return False
        if lnC_by_element is not None:
            for k, v in self.lnC_by_element.items():
                if k not in lnC_by_element:
                    if abs(float(v)) > tol:
                        return False
                elif abs(float(lnC_by_element[k]) - float(v)) > tol:
                    return False
        elif not self.is_legacy_gaussian():
            return False
        if fix_alpha_by_element is not None:
            try:
                other = MmhEnvelopeConfig.from_width_mapping(
                    fix_alpha_by_element, kind=self.kind, lnC=lnC_by_element
                )
            except (ValueError, TypeError):
                return False
            for k, v in self.width_by_element.items():
                if abs(float(other.width_by_element[k]) - float(v)) > tol:
                    return False
            return True
        if fix_alpha_bohr2 is None:
            return False
        return self.is_uniform_width() and abs(float(fix_alpha_bohr2) - self.legacy_fix_alpha()) <= tol

    def width_per_frame_symbols(self, mm_symbols: tuple[str, ...] | list[str]) -> np.ndarray:
        return np.asarray([self.width_for_symbol(s) for s in mm_symbols], dtype=np.float64)

    def C_per_frame_symbols(self, mm_symbols: tuple[str, ...] | list[str]) -> np.ndarray:
        return np.asarray([self.C_for_symbol(s) for s in mm_symbols], dtype=np.float64)


# Backward-compatible alias (batch_hf_emb0_cp, etc.)
MmhAlphaConfig = MmhEnvelopeConfig


def alpha_per_kernel_sites(
    mm_symbols: tuple[str, ...] | list[str],
    cfg: MmhEnvelopeConfig,
) -> np.ndarray:
    alphas: list[float] = []
    for sym in mm_symbols:
        if not is_kernel_mm_symbol(sym):
            continue
        alphas.append(cfg.width_for_symbol(sym))
    return np.asarray(alphas, dtype=np.float64)
