#!/usr/bin/env python3
"""
Predict dE_ML = Σ e_i from a trained ksoft checkpoint.

Requires a precomputed SOAP + k_j cache (``--soap-npz``), or will run precompute when
``--ref-dir`` is supplied alongside ``--coo-dir``.

Output: ``delta_E.txt`` with one column ``deltaE_pred`` (kcal/mol), or two columns
``deltaE_label deltaE_pred`` when ``--e0-file`` is given.

Example (cache already built by the pipeline)::

  python predict_delta_e.py --ckpt examples/Gly+.ckpt --soap-npz examples/Gly+.npz \\
    --coo-dir examples/Coo --qm-atoms 10 --i0 1 --n-frames 10 -o delta_E.txt

Example with reference labels::

  python predict_delta_e.py --ckpt mix.ckpt --soap-npz mix.npz \\
    --e0-file EGly+.txt -o delta_E.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys

# Keep public runs from leaving local __pycache__ artifacts.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict dE_ML from Coo + ksoft checkpoint.")
    p.add_argument("--ckpt", type=Path, required=True, help="Trained ksoft checkpoint (.ckpt)")
    p.add_argument(
        "--soap-npz",
        type=Path,
        default=None,
        help="Precomputed mix_mmh npz with feat + kernel_h_flat (from precompute --ref-dir)",
    )
    p.add_argument("--coo-dir", type=Path, default=None, help="Directory with Coo{i}.xyz frames")
    p.add_argument("--qm-atoms", type=int, default=None, help="Number of QM atoms in Coo files")
    p.add_argument("--ref-dir", type=Path, default=None, help="ref_*.npz dir for on-the-fly precompute")
    p.add_argument("--i0", type=int, default=1, help="First Coo index (default 1)")
    p.add_argument("--n-frames", type=int, default=None, help="Number of frames (default: all in npz)")
    p.add_argument("--e0-file", type=Path, default=None, help="Optional E*.txt labels (one ΔE per line)")
    p.add_argument("-o", "--out", type=Path, default=Path("delta_E.txt"), help="Output text file")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto")
    p.add_argument("--precompute-out", type=Path, default=None, help="Temp npz path when auto-precomputing")
    return p.parse_args()


def _settings_from_ref(ref_dir: Path, file_index: int) -> dict:
    """Read SCF settings from the reference npz so on-the-fly k_i uses the matching AO basis."""
    ref_path = Path(ref_dir) / f"ref_{int(file_index)}.npz"
    if not ref_path.is_file():
        raise SystemExit(f"reference npz not found for Coo{file_index}: {ref_path}")
    with np.load(ref_path, allow_pickle=False) as z:
        if "meta_json" not in z:
            raise SystemExit(f"{ref_path}: missing meta_json; cannot infer SCF settings safely")
        meta = json.loads(str(z["meta_json"]))
    xc = str(meta.get("xc", "HF")).strip()
    method = "hf" if xc.upper() in ("HF", "RHF", "HARTREEFOCK") else xc.lower()
    basis = str(meta.get("basis", "6-31g*")).strip()
    return {
        "method": method,
        "basis": basis,
        "d3bj": bool(meta.get("use_d3bj", False)),
        "threads": 1,
        "qm_charge": int(meta.get("qm_charge", 0)),
        "n_qm": int(meta.get("n_qm", 0) or 0),
    }


def _build_temp_manifest(
    *,
    coo_dir: Path,
    qm_atoms: int,
    i0: int,
    n_frames: int,
    e0_file: Path | None,
    scf_settings: dict,
) -> dict:
    """Build the flat manifest consumed directly by the precompute script."""
    ds: dict = {
        "prefix": "pred",
        "residue": "pred",
        "n_qm": int(qm_atoms),
        "coo_dir": str(coo_dir.resolve()),
        "coo_name_fmt": "Coo{}.xyz",
        "i0": int(i0),
        "n_frames": int(n_frames),
        "e0_i0": int(i0),
        "qm_charge": int(scf_settings.get("qm_charge", 0)),
    }
    if e0_file is not None:
        ds["e0_file"] = str(e0_file.resolve())

    # Keep this path consistent with the production envelope used for released k_i caches.
    return {
        "envelope_kind": "exp",
        "fix_alpha": 5.6685,
        "fix_alpha_by_element": {
            "H": 5.6685,
            "C": 6.775,
            "N": 6.809,
            "O": 7.006,
            "Na": 7.898,
            "Cl": 2.794,
        },
        "lnC_by_element": {
            "H": 1.9806021698973124,
            "C": 2.476,
            "N": 2.655,
            "O": 2.855,
            "Na": 3.307,
            "Cl": 0.372,
        },
        "r_cut": 5.0,
        "n_max": 8,
        "l_max": 6,
        "sigma": 0.5,
        "workers": 1,
        "charge_mode": "tip3p",
        "qm_charge": int(scf_settings.get("qm_charge", 0)),
        "scf": {
            "method": str(scf_settings["method"]),
            "basis": str(scf_settings["basis"]),
            "d3bj": bool(scf_settings.get("d3bj", False)),
            "threads": 1,
            "qm_charge": int(scf_settings.get("qm_charge", 0)),
        },
        "datasets": [ds],
    }


def _run_precompute(manifest_path: Path, out_npz: Path, ref_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(_ROOT / "precompute_soap_e0_mix_mmh.py"),
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_npz),
        "--ref-dir",
        str(ref_dir),
        "--workers",
        "1",
    ]
    print(f"[predict_delta_e] precompute → {out_npz}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(_ROOT))


def _resolve_soap_npz(args: argparse.Namespace) -> Path:
    if args.soap_npz is not None:
        p = Path(args.soap_npz).resolve()
        if not p.is_file():
            raise SystemExit(f"--soap-npz not found: {p}")
        return p
    if args.coo_dir is None or args.ref_dir is None or args.qm_atoms is None:
        raise SystemExit(
            "Need --soap-npz, or (--coo-dir + --ref-dir + --qm-atoms) to build features/k_j."
        )
    coo_dir = Path(args.coo_dir).resolve()
    ref_dir = Path(args.ref_dir).resolve()
    if not coo_dir.is_dir():
        raise SystemExit(f"--coo-dir not found: {coo_dir}")
    if not ref_dir.is_dir():
        raise SystemExit(f"--ref-dir not found: {ref_dir}")
    n_fr = int(args.n_frames or 1)
    out_npz = args.precompute_out
    if out_npz is None:
        out_npz = Path(tempfile.gettempdir()) / "embr_predict_precompute.npz"
    out_npz = Path(out_npz).resolve()
    work = out_npz.parent
    manifest = work / "_predict_manifest.json"
    e0_in_work = None
    if args.e0_file is not None:
        e0_src = Path(args.e0_file).resolve()
        e0_in_work = work / e0_src.name
        e0_in_work.write_text(e0_src.read_text(encoding="utf-8"), encoding="utf-8")
    scf_settings = _settings_from_ref(ref_dir, int(args.i0))
    ref_n_qm = int(scf_settings.get("n_qm", 0))
    if ref_n_qm and ref_n_qm != int(args.qm_atoms):
        raise SystemExit(
            f"--qm-atoms={args.qm_atoms} does not match ref metadata n_qm={ref_n_qm} "
            f"for Coo{args.i0}"
        )
    man = _build_temp_manifest(
        coo_dir=coo_dir,
        qm_atoms=int(args.qm_atoms),
        i0=int(args.i0),
        n_frames=n_fr,
        e0_file=e0_in_work,
        scf_settings=scf_settings,
    )
    manifest.write_text(json.dumps(man, indent=2), encoding="utf-8")
    _run_precompute(manifest, out_npz, ref_dir)
    return out_npz


def main() -> None:
    args = _parse_args()

    from embr_cache import load_mmh_cache, n_frames
    from embr_infer import attach_cache_path, infer_e_j, load_mmh_model
    from embr_io import COO_NAME_FMT, coo_path, load_e0_txt
    from soap_mm_util import resolve_device

    ckpt = Path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"--ckpt not found: {ckpt}")

    soap_npz = _resolve_soap_npz(args)
    cache = attach_cache_path(load_mmh_cache(soap_npz, normalize_elements=True), soap_npz)
    if cache.get("kernel_h_flat") is None:
        raise SystemExit(f"{soap_npz}: missing kernel_h_flat — re-run precompute with --ref-dir")

    n_all = n_frames(cache)
    n_use = int(args.n_frames) if args.n_frames is not None else n_all
    if n_use < 1 or n_use > n_all:
        raise SystemExit(f"--n-frames={n_use} invalid (npz has {n_all} frames)")

    labels: np.ndarray | None = None
    if args.e0_file is not None:
        labels = load_e0_txt(Path(args.e0_file))
        if int(labels.size) < n_use:
            raise SystemExit(f"{args.e0_file}: {labels.size} lines < n_frames={n_use}")

    dev = resolve_device(str(args.device))
    model = load_mmh_model(ckpt, cache, device=dev)

    preds: list[float] = []
    for k in range(n_use):
        _, e0_pred, _ = infer_e_j(model, cache, k)
        preds.append(float(e0_pred))
        fi = k + int(args.i0)
        coo_p = None
        if args.coo_dir is not None:
            coo_p = coo_path(Path(args.coo_dir), COO_NAME_FMT, fi)
        tag = coo_p.name if coo_p is not None else f"frame{k}"
        print(f"  {tag}: deltaE_pred = {e0_pred:+.4f} kcal/mol", flush=True)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# deltaE_pred in kcal/mol (ΔE_ML = Σ e_i)"]
    if labels is not None:
        lines[0] = "# deltaE_label deltaE_pred  (kcal/mol)"
        for i in range(n_use):
            lines.append(f"{float(labels[i]):+.6f}  {preds[i]:+.6f}")
    else:
        for v in preds:
            lines.append(f"{v:+.6f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[predict_delta_e] wrote {out} ({n_use} frames)", flush=True)

    if labels is not None:
        y_true = np.asarray(labels[:n_use], dtype=float)
        y_pred = np.asarray(preds, dtype=float)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        print(f"[predict_delta_e] RMSE = {rmse:.4f} kcal/mol  (n={n_use})", flush=True)


if __name__ == "__main__":
    main()
