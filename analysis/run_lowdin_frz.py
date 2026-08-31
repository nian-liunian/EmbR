#!/usr/bin/env python3
"""
Löwdin frozen-orbital density used as the PED reference in the QM/MM density analysis.

Fragment A = QM region, fragment B = retained MM region from Coo. Builds P_frz via Löwdin
orthogonalization of occupied MOs of A∪B; optional frozen-density difference cube export.

Requires PySCF (``pip install -r requirements-scf.txt``).

Example::

  python analysis/run_lowdin_frz.py --coo examples/Coo/Coo1.xyz --n-qm 10 \\
    --method hf --basis 6-31g* --out-cube hl_pauli_hole.cube --save-npz hl_pauli.npz
"""

from __future__ import annotations

import argparse
import os
import sys

# Keep public runs from leaving local __pycache__ artifacts.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS = Path(__file__).resolve().parent
for p in (_ROOT, _ANALYSIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scf_embed_io import load_scf_frame
from scf_embed_pyscf import scf_embed_config_from_cli

from pyscf_hl_pauli import run_frame_to_cube


def main() -> None:
    p = argparse.ArgumentParser(description="Löwdin FRZ density + optional cube/npz export.")
    p.add_argument("--coo", type=Path, required=True, help="Coo*.xyz frame")
    p.add_argument("--n-qm", type=int, required=True, help="Number of QM atoms")
    p.add_argument("--method", default="hf")
    p.add_argument("--basis", default="6-31g*")
    p.add_argument("--d3bj", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out-cube", type=Path, required=True, help="Output Gaussian cube for the selected frozen-density difference")
    p.add_argument("--save-npz", type=Path, default=None, help="Optional npz with dm_A0, dm_frz, ...")
    p.add_argument("--n-grid", type=int, default=80)
    p.add_argument("--cube-from", default="hole", choices=("hole", "ClP", "Atilde"))
    p.add_argument("--scf-mi", action="store_true", help="Also run optional SCF-MI (not paper default)")
    args = p.parse_args()

    frame = load_scf_frame(Path(args.coo), n_qm=int(args.n_qm))
    cfg = scf_embed_config_from_cli(
        method=str(args.method),
        basis=str(args.basis),
    )
    run_frame_to_cube(
        frame,
        cfg,
        out_cube=Path(args.out_cube),
        threads=int(args.threads),
        n_grid=int(args.n_grid),
        save_npz=None if args.save_npz is None else Path(args.save_npz),
        cube_from=str(args.cube_from),
        scf_mi=bool(args.scf_mi),
    )
    print(f"[run_lowdin_frz] done → {args.out_cube}", flush=True)


if __name__ == "__main__":
    main()
