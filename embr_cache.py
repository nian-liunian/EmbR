"""Flat mixed MM-H SOAP cache for train_soap_e0_mix_mmh.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from embr_features import MM_EL_H as MIX_AI_EL_H
from embr_elements import MMH_ELEM_H, MMH_ELEM_O


def normalize_mmh_element_flat(el: np.ndarray) -> np.ndarray:
    """
    Map legacy mix_ai MM element ids (O=0, H=1) to mix_mmh ids when cache is H-only.

    Partition-OH caches store MMH_ELEM_O (=4) explicitly; do not remap those rows.
    """
    out = np.asarray(el, dtype=np.int8).reshape(-1).copy()

    if int(MMH_ELEM_O) in set(int(x) for x in np.unique(out)):
        return out
    uniq = set(int(x) for x in np.unique(out))
    if uniq <= {int(MMH_ELEM_H)}:
        return out
    if int(MIX_AI_EL_H) in uniq:
        out[out == int(MIX_AI_EL_H)] = int(MMH_ELEM_H)
    return out


def load_mmh_cache(path: Path, *, normalize_elements: bool = True) -> dict:
    z = np.load(path, allow_pickle=False)
    required = ("feat_h_flat", "mm_element_flat", "mm_ptr", "e0", "meta_json")
    for k in required:
        if k not in z:
            raise ValueError(f"{path}: missing {k!r}")
    meta = json.loads(str(z["meta_json"]))
    mm_el = np.asarray(z["mm_element_flat"], dtype=np.int8)
    if normalize_elements:
        mm_el = normalize_mmh_element_flat(mm_el)
    out: dict = {
        "feat_h_flat": np.asarray(z["feat_h_flat"], dtype=np.float32),
        "mm_element_flat": mm_el,
        "dist_qm_flat": np.asarray(z["dist_qm_flat"], dtype=np.float32) if "dist_qm_flat" in z else None,
        "mm_ptr": np.asarray(z["mm_ptr"], dtype=np.int64),
        "e0": np.asarray(z["e0"], dtype=np.float64),
        "residue_id": np.asarray(z["residue_id"], dtype=np.int16) if "residue_id" in z else None,
        "meta": meta,
        "with_emb0": bool(meta.get("with_emb0", "kernel_h_flat" in z)),
    }
    if "kernel_h_flat" in z:
        out["kernel_h_flat"] = np.asarray(z["kernel_h_flat"], dtype=np.float64)
    else:
        out["kernel_h_flat"] = None
    if "feat_qm_near_flat" in z and "qm_near_ptr" in z:
        out["feat_qm_near_flat"] = np.asarray(z["feat_qm_near_flat"], dtype=np.float32)
        out["qm_near_ptr"] = np.asarray(z["qm_near_ptr"], dtype=np.int64)
    else:
        out["feat_qm_near_flat"] = None
        out["qm_near_ptr"] = None
    if "feat_all_atom_flat" in z and "all_atom_ptr" in z:
        out["feat_all_atom_flat"] = np.asarray(z["feat_all_atom_flat"], dtype=np.float32)
        out["all_atom_ptr"] = np.asarray(z["all_atom_ptr"], dtype=np.int64)
    else:
        out["feat_all_atom_flat"] = None
        out["all_atom_ptr"] = None
    if "kernel_mm_flat" in z and "mm_element_k_flat" in z and "mm_ptr_k" in z:
        out["kernel_mm_flat"] = np.asarray(z["kernel_mm_flat"], dtype=np.float64)
        out["mm_element_k_flat"] = np.asarray(z["mm_element_k_flat"], dtype=np.int8)
        out["mm_ptr_k"] = np.asarray(z["mm_ptr_k"], dtype=np.int64)
    else:
        out["kernel_mm_flat"] = None
        out["mm_element_k_flat"] = None
        out["mm_ptr_k"] = None
    return out


def attach_frame_amp_labels(
    cache: dict,
    amp_frame: np.ndarray,
) -> dict:
    """Return shallow copy with ``amp_frame`` (n_frames,) scalar A labels [Hartree]."""
    from embr_partition import attach_frame_amp_labels as _attach

    return _attach(cache, amp_frame)


def attach_amp_labels(
    cache: dict,
    amp_flat: np.ndarray,
) -> dict:
    """Return shallow copy with ``amp_h_flat`` aligned to ``feat_h_flat`` rows."""
    n_rows = int(cache["feat_h_flat"].shape[0])
    amp = np.asarray(amp_flat, dtype=np.float64).reshape(-1)
    if int(amp.size) != n_rows:
        raise ValueError(f"amp_flat length {amp.size} != soap rows {n_rows}")
    out = dict(cache)
    out["amp_h_flat"] = amp
    return out


def build_teacher_amp_flat(
    cache: dict,
    teacher_ckpt: Path,
    *,
    kernel_cache: Path | None,
    device: torch.device,
) -> np.ndarray:
    """A_j = e_j^teacher / (627.5·k_j) with varying e_j from a trained k/legacy ckpt."""
    from embr_partition import amp_h_flat_from_ej_kernel
    from embr_model import forward_e_j, load_mmh_model_from_ckpt

    cache_k = ensure_kernel_h_flat(cache, kernel_cache)
    model, _ck = load_mmh_model_from_ckpt(teacher_ckpt, cache_k, device=device)
    blocks: list[np.ndarray] = []
    with torch.no_grad():
        for k in range(n_frames(cache_k)):
            feat, ker, el, dist, _e0 = mmh_frame_slice(cache_k, k)
            if ker is None:
                raise ValueError("teacher amp labels need kernel_h_flat")
            feat_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            el_t = torch.tensor(el, dtype=torch.long, device=device).unsqueeze(0)
            ker_t = torch.tensor(ker, dtype=torch.float32, device=device).unsqueeze(0)
            dist_t = None if dist is None else torch.tensor(dist, dtype=torch.float32, device=device).unsqueeze(0)
            e_j = (
                forward_e_j(
                    model,
                    feat_t=feat_t,
                    el_t=el_t,
                    ker_t=ker_t,
                    e0_t=None,
                    dist_t=dist_t,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )
            blocks.append(amp_h_flat_from_ej_kernel(e_j, ker))
    return np.concatenate(blocks, axis=0)


def n_frames(cache: dict) -> int:
    return int(cache["e0"].shape[0])


def mmh_frame_slice(cache: dict, k: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, float]:
    """One frame: feat (n_h, d), kernel (n_h,) or None, mm_element (n_h,), dist_qm or None, e0."""
    ptr = cache["mm_ptr"]
    s, e = int(ptr[k]), int(ptr[k + 1])
    feat = cache["feat_h_flat"][s:e]
    ker = None
    if cache.get("kernel_h_flat") is not None:
        ker = cache["kernel_h_flat"][s:e]
    el = cache["mm_element_flat"][s:e]
    dist = None
    if cache.get("dist_qm_flat") is not None:
        dist = cache["dist_qm_flat"][s:e]
    e0 = float(cache["e0"][k])
    return feat, ker, el, dist, e0







def _materialize_by_n_qm_near(cache: dict, device: torch.device) -> dict[int, dict]:
    if cache.get("feat_qm_near_flat") is None or cache.get("qm_near_ptr") is None:
        raise ValueError(
            "amp featurizer qm_near requires feat_qm_near_flat in cache; "
            "re-run precompute with --amp-qm-near-cut ANG"
        )
    ptr = cache["qm_near_ptr"]
    d = int(cache["feat_qm_near_flat"].shape[1])
    by_n: dict[int, list[int]] = {}
    n = n_frames(cache)
    for k in range(n):
        n_qm = int(ptr[k + 1] - ptr[k])
        by_n.setdefault(n_qm, []).append(k)

    has_amp_frame = cache.get("amp_frame") is not None
    out: dict[int, dict] = {}
    for n_qm, frame_ids in by_n.items():
        g = len(frame_ids)
        feat = np.empty((g, n_qm, d), dtype=np.float32)
        e0 = np.empty((g,), dtype=np.float64)
        amp_frame = np.empty((g,), dtype=np.float64) if has_amp_frame else None
        for j, k in enumerate(frame_ids):
            ff, lab = qm_near_frame_slice(cache, k)
            feat[j] = ff
            e0[j] = lab
            if has_amp_frame and amp_frame is not None:
                amp_frame[j] = float(cache["amp_frame"][k])
        pack: dict = {
            "frame_ids": frame_ids,
            "feat": torch.tensor(feat, dtype=torch.float32, device=device),
            "mm_element": torch.zeros((g, n_qm), dtype=torch.long, device=device),
            "e0": torch.tensor(e0, dtype=torch.float32, device=device),
        }
        if has_amp_frame and amp_frame is not None:
            pack["amp_frame"] = torch.tensor(amp_frame, dtype=torch.float32, device=device)
        out[int(n_qm)] = pack
    return out


def _materialize_by_n_all_atom(cache: dict, device: torch.device) -> dict[int, dict]:
    if cache.get("feat_all_atom_flat") is None or cache.get("all_atom_ptr") is None:
        raise ValueError(
            "amp featurizer all requires feat_all_atom_flat in cache; "
            "re-run precompute with --amp-all-atoms"
        )
    ptr = cache["all_atom_ptr"]
    d = int(cache["feat_all_atom_flat"].shape[1])
    by_n: dict[int, list[int]] = {}
    n = n_frames(cache)
    for k in range(n):
        n_atoms = int(ptr[k + 1] - ptr[k])
        by_n.setdefault(n_atoms, []).append(k)

    has_amp_frame = cache.get("amp_frame") is not None
    out: dict[int, dict] = {}
    for n_atoms, frame_ids in by_n.items():
        g = len(frame_ids)
        feat = np.empty((g, n_atoms, d), dtype=np.float32)
        e0 = np.empty((g,), dtype=np.float64)
        amp_frame = np.empty((g,), dtype=np.float64) if has_amp_frame else None
        for j, k in enumerate(frame_ids):
            ff, lab = all_atom_frame_slice(cache, k)
            feat[j] = ff
            e0[j] = lab
            if has_amp_frame and amp_frame is not None:
                amp_frame[j] = float(cache["amp_frame"][k])
        pack: dict = {
            "frame_ids": frame_ids,
            "feat": torch.tensor(feat, dtype=torch.float32, device=device),
            "mm_element": torch.zeros((g, n_atoms), dtype=torch.long, device=device),
            "e0": torch.tensor(e0, dtype=torch.float32, device=device),
        }
        if has_amp_frame and amp_frame is not None:
            pack["amp_frame"] = torch.tensor(amp_frame, dtype=torch.float32, device=device)
        out[int(n_atoms)] = pack
    return out


def materialize_by_n_mm_h(cache: dict, device: torch.device) -> dict[int, dict]:
    """Stack frames with equal n_mm_h into dense tensors on ``device``."""
    ptr = cache["mm_ptr"]
    d = int(cache["feat_h_flat"].shape[1])
    by_n: dict[int, list[int]] = {}
    n = n_frames(cache)
    for k in range(n):
        n_h = int(ptr[k + 1] - ptr[k])
        by_n.setdefault(n_h, []).append(k)

    out: dict[int, dict] = {}
    for n_h, frame_ids in by_n.items():
        g = len(frame_ids)
        feat = np.empty((g, n_h, d), dtype=np.float32)
        el = np.empty((g, n_h), dtype=np.int64)
        dist = np.empty((g, n_h), dtype=np.float32)
        e0 = np.empty((g,), dtype=np.float64)
        has_ker = cache.get("kernel_h_flat") is not None
        has_dist = cache.get("dist_qm_flat") is not None
        has_amp_frame = cache.get("amp_frame") is not None
        has_coeff = cache.get("coeff_h_flat") is not None
        ker = np.empty((g, n_h), dtype=np.float64) if has_ker else None
        amp_frame = np.empty((g,), dtype=np.float64) if has_amp_frame else None
        coeff = np.empty((g, n_h), dtype=np.float64) if has_coeff else None
        for j, k in enumerate(frame_ids):
            ff, kk, ee, dd, lab = mmh_frame_slice(cache, k)
            feat[j] = ff
            el[j] = ee
            if has_ker and kk is not None and ker is not None:
                ker[j] = kk
            if has_amp_frame and amp_frame is not None:
                amp_frame[j] = float(cache["amp_frame"][k])
            if has_coeff and coeff is not None:
                s, e = int(ptr[k]), int(ptr[k + 1])
                coeff[j] = cache["coeff_h_flat"][s:e]
            if has_dist and dd is not None:
                dist[j] = dd
            e0[j] = lab
        pack: dict = {
            "frame_ids": frame_ids,
            "feat": torch.tensor(feat, dtype=torch.float32, device=device),
            "mm_element": torch.tensor(el, dtype=torch.long, device=device),
            "e0": torch.tensor(e0, dtype=torch.float32, device=device),
        }
        if has_ker and ker is not None:
            pack["kernel"] = torch.tensor(ker, dtype=torch.float32, device=device)
        if has_amp_frame and amp_frame is not None:
            pack["amp_frame"] = torch.tensor(amp_frame, dtype=torch.float32, device=device)
        if has_coeff and coeff is not None:
            pack["coeff"] = torch.tensor(coeff, dtype=torch.float32, device=device)
        if has_dist:
            pack["dist_qm"] = torch.tensor(dist, dtype=torch.float32, device=device)
        out[int(n_h)] = pack
    return out


def ensure_kernel_h_flat(cache: dict, kernel_cache: Path | None) -> dict:
    """
    Return cache with ``kernel_h_flat`` populated.

    If missing, read H-site k_j from aligned mix_ai ``kernel_mm_flat`` (``--kernel-cache``).
    """
    if cache.get("kernel_h_flat") is not None:
        return cache
    if kernel_cache is None:
        raise ValueError(
            "soap cache has no kernel_h_flat; pass --kernel-cache mix_ai_mmqm.npz "
            "(or precompute with --ref-dir / without --no-emb0)"
        )
    from embr_infer import kernel_h_for_frame, _load_kernel_source

    kernel_src = _load_kernel_source(kernel_cache)
    n = n_frames(cache)
    ptr = list(cache["mm_ptr"])
    blocks: list[np.ndarray] = []
    for k in range(n):
        blocks.append(np.asarray(kernel_h_for_frame(kernel_src, k), dtype=np.float64))
    out = dict(cache)
    out["kernel_h_flat"] = np.concatenate(blocks, axis=0)
    out["with_emb0"] = True
    return out


def _concat_csr_ptrs(ptrs: list[np.ndarray]) -> np.ndarray:
    """Concatenate CSR-style ptr arrays (each must start at 0)."""
    out: list[int] = [0]
    for p in ptrs:
        arr = np.asarray(p, dtype=np.int64).reshape(-1)
        if arr.size < 1 or int(arr[0]) != 0:
            raise ValueError("CSR ptr must start at 0")
        base = out[-1]
        out.extend(int(x) + base for x in arr[1:])
    return np.asarray(out, dtype=np.int64)


def merge_mmh_caches(paths: list[Path], out: Path) -> Path:
    """
    Concatenate per-system MM-H SOAP caches into one flat cache for mixed training.

    Frames are stacked in ``paths`` order. ``residue_id`` / ``residue_to_id`` are
    remapped to a global vocabulary. Hyperparameters and optional array presence
    must match across inputs.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("merge_mmh_caches: empty paths")
    caches = [load_mmh_cache(p) for p in paths]
    meta0 = caches[0]["meta"]
    d_feat0 = int(caches[0]["feat_h_flat"].shape[1])
    has_ker = caches[0].get("kernel_h_flat") is not None
    has_ker_mm = caches[0].get("kernel_mm_flat") is not None
    has_qm_near = caches[0].get("feat_qm_near_flat") is not None
    has_all_atom = caches[0].get("feat_all_atom_flat") is not None
    has_dist = caches[0].get("dist_qm_flat") is not None

    meta_keys = (
        "d_feat",
        "d_soap",
        "feature_mode",
        "mm_env",
        "partition_oh",
        "with_emb0",
        "r_cut_mm",
    )
    hyper0 = dict(meta0.get("hyper") or {})

    for i, c in enumerate(caches[1:], start=1):
        m = c["meta"]
        for k in meta_keys:
            if m.get(k) != meta0.get(k):
                raise ValueError(
                    f"merge incompatible: {paths[i].name} meta[{k!r}]={m.get(k)!r} "
                    f"!= {paths[0].name} {meta0.get(k)!r}"
                )
        if dict(m.get("hyper") or {}) != hyper0:
            raise ValueError(
                f"merge incompatible SOAP hyper: {paths[i].name} vs {paths[0].name}"
            )
        if int(c["feat_h_flat"].shape[1]) != d_feat0:
            raise ValueError(f"merge incompatible d_feat in {paths[i]}")
        if (c.get("kernel_h_flat") is not None) != has_ker:
            raise ValueError(f"merge: kernel_h_flat presence mismatch in {paths[i]}")
        if (c.get("kernel_mm_flat") is not None) != has_ker_mm:
            raise ValueError(f"merge: kernel_mm_flat presence mismatch in {paths[i]}")
        if (c.get("feat_qm_near_flat") is not None) != has_qm_near:
            raise ValueError(f"merge: feat_qm_near_flat presence mismatch in {paths[i]}")
        if (c.get("feat_all_atom_flat") is not None) != has_all_atom:
            raise ValueError(f"merge: feat_all_atom_flat presence mismatch in {paths[i]}")
        if (c.get("dist_qm_flat") is not None) != has_dist:
            raise ValueError(f"merge: dist_qm_flat presence mismatch in {paths[i]}")

    residue_to_id: dict[str, int] = {}
    rid_blocks: list[np.ndarray] = []
    frame_meta: list[dict] = []
    source_paths: list[str] = []

    for path, c in zip(paths, caches):
        meta = c["meta"]
        local_map = meta.get("residue_to_id") or {}
        id_to_name = {int(v): str(k) for k, v in local_map.items()}
        n = n_frames(c)
        local_rid = c.get("residue_id")
        frames = list(meta.get("frames") or [])
        new_rid = np.zeros((n,), dtype=np.int16)
        for k in range(n):
            name: str | None = None
            if local_rid is not None:
                lid = int(local_rid[k])
                name = id_to_name.get(lid)
            if name is None and k < len(frames):
                name = str(frames[k].get("residue") or frames[k].get("residue_name") or "")
            if not name:
                name = f"src_{len(source_paths)}"
            if name not in residue_to_id:
                residue_to_id[name] = len(residue_to_id)
            new_rid[k] = int(residue_to_id[name])
            fr = dict(frames[k]) if k < len(frames) else {}
            fr["source_cache"] = str(path.resolve())
            fr["source_frame_k"] = int(k)
            frame_meta.append(fr)
        rid_blocks.append(new_rid)
        source_paths.append(str(path.resolve()))

    n_total = sum(n_frames(c) for c in caches)
    meta = dict(meta0)
    meta.update(
        {
            "pipeline": "soap_e0_mix_mmh_merged",
            "n_frames": int(n_total),
            "residue_to_id": residue_to_id,
            "frames": frame_meta,
            "merged_from": source_paths,
            "manifest": "merged",
        }
    )

    out_kw: dict = {
        "feat_h_flat": np.concatenate([c["feat_h_flat"] for c in caches], axis=0),
        "mm_element_flat": np.concatenate([c["mm_element_flat"] for c in caches], axis=0),
        "mm_ptr": _concat_csr_ptrs([c["mm_ptr"] for c in caches]),
        "e0": np.concatenate([c["e0"] for c in caches], axis=0),
        "residue_id": np.concatenate(rid_blocks, axis=0),
        "meta_json": np.array(json.dumps(meta)),
    }
    if has_dist:
        out_kw["dist_qm_flat"] = np.concatenate(
            [c["dist_qm_flat"] for c in caches], axis=0
        )
    if has_ker:
        out_kw["kernel_h_flat"] = np.concatenate(
            [c["kernel_h_flat"] for c in caches], axis=0
        )
    if has_ker_mm:
        out_kw["kernel_mm_flat"] = np.concatenate(
            [c["kernel_mm_flat"] for c in caches], axis=0
        )
        out_kw["mm_element_k_flat"] = np.concatenate(
            [c["mm_element_k_flat"] for c in caches], axis=0
        )
        out_kw["mm_ptr_k"] = _concat_csr_ptrs([c["mm_ptr_k"] for c in caches])
    if has_qm_near:
        out_kw["feat_qm_near_flat"] = np.concatenate(
            [c["feat_qm_near_flat"] for c in caches], axis=0
        )
        out_kw["qm_near_ptr"] = _concat_csr_ptrs([c["qm_near_ptr"] for c in caches])
    if has_all_atom:
        out_kw["feat_all_atom_flat"] = np.concatenate(
            [c["feat_all_atom_flat"] for c in caches], axis=0
        )
        out_kw["all_atom_ptr"] = _concat_csr_ptrs([c["all_atom_ptr"] for c in caches])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **out_kw)
    return out.resolve()
