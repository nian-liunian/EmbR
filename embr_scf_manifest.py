"""Emb0 SCF helpers for the mix_mmh training pipeline (compatible with scf_embed_pyscf)."""

from __future__ import annotations


def scf_settings_from_manifest(cfg: dict) -> dict:
    """Read α / functional / basis / D3BJ / qm_charge from manifest JSON."""
    scf = cfg.get("scf") or {}
    from scf_embed_pyscf import resolve_qm_charge

    return {
        "method": str(scf.get("method", "b3lyp")),
        "basis": str(scf.get("basis", "6-31g*")),
        "use_d3bj": bool(scf.get("d3bj", True)),
        "num_threads": int(scf.get("threads", 4)),
        "verbose": int(scf.get("verbose_scf", 0)),
        "qm_charge": resolve_qm_charge(scf, cfg, default=0),
    }


def build_emb0_scf_config(scf_cfg_kw: dict):
    try:
        from scf_embed_pyscf import scf_embed_config_from_cli

        return scf_embed_config_from_cli(**scf_cfg_kw)
    except (ImportError, AttributeError, TypeError):
        pass

    from scf_embed_pyscf import ScfEmbedConfig

    method = str(scf_cfg_kw.get("method", "b3lyp")).strip().lower()
    xc = "HF" if method in ("hf", "rhf", "hartreefock") else "B3LYP"
    if method not in ("hf", "rhf", "hartreefock", "b3lyp"):
        xc = method.upper()
    return ScfEmbedConfig(
        xc=xc,
        basis=str(scf_cfg_kw.get("basis", "6-31g*")),
        use_d3bj=bool(scf_cfg_kw.get("use_d3bj", True)),
        num_threads=int(scf_cfg_kw.get("num_threads", 4)),
        verbose=int(scf_cfg_kw.get("verbose", 0)),
        qm_charge=int(scf_cfg_kw.get("qm_charge", 0)),
    )


def isotropic_repulsion_cone():
    """Isotropic repulsion geometry (180° cone); ``None`` if ConeRepAng unavailable."""
    try:
        from scf_embed_pyscf import ConeRepAng

        return ConeRepAng()
    except ImportError:
        return None
