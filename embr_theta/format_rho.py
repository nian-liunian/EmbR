"""Format electron density values for terminal output."""

from __future__ import annotations

import numpy as np


def fmt_rho(x: float) -> str:
    """Format ρ or δρ [e/Bohr³]; use scientific notation when |x| < 1e-3."""
    if not np.isfinite(x):
        return "nan"
    if abs(float(x)) >= 1e-3:
        return f"{float(x):.5f}"
    return f"{float(x):+.3e}"


def fmt_rel_pct(rho0: float, drho: float) -> str:
    if not np.isfinite(rho0) or not np.isfinite(drho):
        return "nan"
    if abs(float(rho0)) < 1e-15:
        return "nan"
    return f"{100.0 * float(drho) / float(rho0):+.2f}%"


def fmt_rho_triplet(rho0: float, drho: float) -> str:
    """ρ0 -> ρ1 (δρ, rel%)."""
    rho1 = float(rho0) + float(drho)
    return (
        f"ρ0={fmt_rho(rho0)}  δρ={fmt_rho(drho)}  ρ1={fmt_rho(rho1)}  "
        f"rel={fmt_rel_pct(rho0, drho)}"
    )
