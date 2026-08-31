# Density analysis (Löwdin FRZ)

This directory provides the stable **Löwdin-orthogonalized frozen-density construction** used
as the PED reference in the electron-density analysis. It is provided as the implementation of
the frozen-density step, not as a complete figure-reproduction workflow.

| File | Role |
|------|------|
| `pyscf_hl_pauli.py` | Core: isolated-region SCFs → common-basis occupied orbitals → Löwdin FRZ density |
| `run_lowdin_frz.py` | CLI: Coo frame → frozen-density/PED-reference cube + optional npz |

The repository does **not** include the full radial-shell sampling, configuration/site averaging,
correlation analysis, or plotting pipeline used to prepare the density-response figures.

Optional SCF-MI code retained in `pyscf_hl_pauli.py` / `run_lowdin_frz.py` is experimental and
is not part of the published EmbR workflow.

```bash
pip install -r requirements-scf.txt
python analysis/run_lowdin_frz.py --coo examples/Coo/Coo1.xyz --n-qm 10 \
  --method hf --basis 6-31g* --out-cube hole.cube --save-npz hole.npz
```

Imports resolve against the repository root (`scf_embed_io`, `scf_embed_pyscf`).
