"""
Train EmbR ksoft model: per-MM-site energies e_i with Pearson(e,k) guidance.

  python train_soap_e0_mix_mmh.py --soap-cache mix.npz --ckpt mix.ckpt --model ksoft

``--seed N`` fixes model init, val split, and per-epoch batch shuffle (reproducible retrain).
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from embr_cache import (
    ensure_kernel_h_flat,
    load_mmh_cache,
    materialize_by_n_mm_h,
    n_frames,
)
from embr_model import (
    build_mmh_model,
    apply_e_sites_mask,
    correlation_deficit_loss,
    load_mmh_model_from_ckpt,
    mean_pearson_e_k,
    normalize_e_sites,
    SoapE0MmhKModel,
)
from soap_mm_util import print_device, resolve_device, seed_all


def _parse_layers(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(x) for x in parts)


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    torch.manual_seed(int(seed))
    perm = torch.randperm(n)
    if n <= 0:
        return [], []
    if n == 1:
        # Single-frame smoke/e2e: no hold-out split, but still run val logging.
        if float(val_frac) > 0.0:
            return [0], [0]
        return [0], []
    n_val = min(max(int(round(n * val_frac)), 1), n - 1)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return train_idx, val_idx


def _validate_train_args(args: argparse.Namespace) -> None:
    try:
        e_sites = normalize_e_sites(args.e_sites)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.e_sites = e_sites
    if float(args.weight_corr) <= 0.0:
        raise SystemExit("ksoft needs --weight-corr > 0 (Pearson(e,k) guidance)")



def _check_train_imports(args: argparse.Namespace) -> None:
    """Import path + build a tiny model; fail before loading a large soap-cache."""
    model = build_mmh_model(
        model_kind="ksoft",
        d_feat=8,
        fit_neurons=_parse_layers(args.fit_neurons),
        corr_neurons=_parse_layers(args.corr_neurons),
        dropout=float(args.dropout),
        corr_mlp_scale=float(args.corr_mlp_scale),
        use_dist=not bool(args.no_dist),
        dist_scale_ang=float(args.dist_scale_ang),
    )
    n_param = sum(int(p.numel()) for p in model.parameters())
    print(
        f"[train --check] OK  weight_corr={float(args.weight_corr):g}  "
        f"corr_target={float(args.corr_target):g}  params~{n_param}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MM-H mix_mmh: sum e_j = E0")
    ap.add_argument(
        "--soap-cache",
        type=Path,
        default=None,
        help="required unless --check",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="import modules + validate args + build a tiny model; do NOT load soap-cache",
    )
    ap.add_argument(
        "--kernel-cache",
        type=Path,
        default=None,
        help="mix_ai npz with kernel_mm_flat (optional if soap-cache from --ref-dir precompute)",
    )
    ap.add_argument("--ckpt", type=Path, default=Path("soap_e0_mix_mmh.pt"))
    ap.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="warm-start: load weights from an existing ckpt before training",
    )
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--fit-neurons", type=str, default="128,128", help="E0 拟合网络（与 legacy 相同）")
    ap.add_argument("--corr-neurons", type=str, default="64,64", help="分配小修正网络")
    ap.add_argument("--corr-mlp-scale", type=float, default=0.25, help="小修正幅度上限比例")
    ap.add_argument("--no-dist", action="store_true", help="correction MLP without dist_qm input")
    ap.add_argument("--dist-scale-ang", type=float, default=5.0)
    ap.add_argument("--corr-target", type=float, default=0.9, help="soft target Pearson(e,k)")
    ap.add_argument("--weight-corr", type=float, default=0.1)
    ap.add_argument("--weight-eps", type=float, default=0.05, help="小修正过大则罚")
    ap.add_argument("--weight-e0", type=float, default=1.0, help="预测 E0 与标签的均方误差（主损失）")
    ap.add_argument(
        "--e-sites",
        type=str,
        default="all",
        help="which MM sites may carry e_i: all (default) or positive "
        "(H/Na/K/C/N only; O and Cl forced e=0, no potential).",
    )
    ap.add_argument(
        "--monitor",
        type=str,
        default=None,
        choices=("e0", "loss"),
        help="early-stop / *best metric: e0=val E0 MSE; loss=val total loss (MSE+corr+…). "
        "default: loss for soft/ksoft, e0 otherwise",
    )
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--min-epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed: model init, val split, batch shuffle")
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    args = ap.parse_args()
    _validate_train_args(args)
    if bool(args.check):
        _check_train_imports(args)
        return
    if args.soap_cache is None:
        raise SystemExit("--soap-cache is required (omit only with --check)")

    print(f"[train] loading soap-cache → {args.soap_cache}", flush=True)
    cache = load_mmh_cache(args.soap_cache)
    cache = ensure_kernel_h_flat(cache, args.kernel_cache)
    if (
        cache.get("dist_qm_flat") is None
        and not args.no_dist
    ):
        raise SystemExit("cache has no dist_qm_flat; re-run precompute or pass --no-dist")

    meta = cache["meta"]
    n = n_frames(cache)
    amp_featurizer = "mm"
    d_feat = int(cache['feat_h_flat'].shape[1])
    with_emb0 = bool(cache.get("with_emb0", cache.get("kernel_h_flat") is not None))
    device = resolve_device(args.device)
    seed_all(int(args.seed))
    print(f"[train] seed={int(args.seed)}", flush=True)
    model_kind = "ksoft"
    model = build_mmh_model(
        model_kind=model_kind,
        d_feat=d_feat,
        fit_neurons=_parse_layers(args.fit_neurons),
        corr_neurons=_parse_layers(args.corr_neurons),
        dropout=float(args.dropout),
        corr_mlp_scale=float(args.corr_mlp_scale),
        use_dist=not bool(args.no_dist),
        dist_scale_ang=float(args.dist_scale_ang),
    ).to(device)
    opt = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_idx, val_idx = _split_indices(n, float(args.val_frac), int(args.seed))
    if n == 1 and val_idx:
        print(
            "  note: single-frame dataset; val lines use the same frame (no hold-out)",
            flush=True,
        )
    train_set, val_set = set(train_idx), set(val_idx)
    groups = materialize_by_n_mm_h(cache, device)

    group_local: dict[int, dict[str, list[int]]] = {}
    for n_h, pack in groups.items():
        tr: list[int] = []
        va: list[int] = []
        for j, fid in enumerate(pack["frame_ids"]):
            if fid in train_set:
                tr.append(j)
            elif fid in val_set:
                va.append(j)
        group_local[n_h] = {"train": tr, "val": va}

    print_device(device)
    print(
        f"[train] frames={n} train={len(train_idx)} val={len(val_idx)}  "
        f"batch={args.batch}  corr_target={args.corr_target}",
        flush=True,
    )
    print(
        f"[train] lr={float(args.lr):g}  weight_decay={float(args.weight_decay):g}  "
        f"epochs={int(args.epochs)}  patience={int(args.patience)}  "
        f"weight_corr={float(args.weight_corr):g}  e_sites={args.e_sites}",
        flush=True,
    )
    setattr(model, "e_sites", str(args.e_sites))

    use_k = isinstance(model, SoapE0MmhKModel)
    soft = use_k
    use_amp = False
    use_amp_c = False

    def _batch_loss(
        feat_t: torch.Tensor,
        el_t: torch.Tensor,
        e0_t: torch.Tensor,
        ker_t: torch.Tensor | None,
        dist_t: torch.Tensor | None,
        amp_frame_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        stats: dict[str, float] = {}
        if use_amp_c:
            assert amp_frame_t is not None
            c_pred = model.per_site_coefficients(feat_t, el_t)
            c_tgt = amp_frame_t.unsqueeze(1).expand_as(c_pred)
            loss = torch.mean((c_pred - c_tgt) ** 2)
            stats["coeff_mse"] = float(loss.detach().cpu())
            stats["pearson_e_k"] = float("nan")
        elif use_amp:
            assert amp_frame_t is not None
            a_pred = model.forward_amp(feat_t, el_t)
            loss = torch.mean((a_pred - amp_frame_t) ** 2)
            stats["amp_mse"] = float(torch.mean((a_pred - amp_frame_t) ** 2).detach().cpu())
            stats["pearson_e_k"] = float("nan")
        elif use_k:
            assert ker_t is not None
            soft = use_k
            e_j = model.per_site_energies(feat_t, el_t, ker_t, dist_qm=dist_t)
            e_j = apply_e_sites_mask(e_j, el_t, args.e_sites)
            pred = e_j.sum(dim=1)
            loss = float(args.weight_e0) * torch.mean((pred - e0_t) ** 2)
            if float(args.weight_corr) > 0.0:
                loss = loss + float(args.weight_corr) * correlation_deficit_loss(
                    e_j,
                    ker_t,
                    target=float(args.corr_target),
                    mm_element=el_t,
                    e_sites=args.e_sites,
                )
            if (not soft) and float(args.weight_eps) > 0.0:
                _w, eps = model.weights(feat_t, el_t, ker_t, dist_qm=dist_t)
                loss = loss + float(args.weight_eps) * torch.mean(eps * eps)
                stats["eps_rms"] = float(torch.sqrt(torch.mean(eps * eps)).detach().cpu())
                stats["w_sum"] = float(_w.sum(dim=1).mean().detach().cpu())
            else:
                stats["eps_rms"] = float("nan")
                stats["w_sum"] = float("nan")
            stats["e0_mse"] = float(torch.mean((pred - e0_t) ** 2).detach().cpu())
            stats["pearson_e_k"] = float(
                mean_pearson_e_k(
                    e_j, ker_t, mm_element=el_t, e_sites=args.e_sites
                ).detach().cpu()
            )
        else:
            e_j = model.per_site_energies(feat_t, el_t)
            e_j = apply_e_sites_mask(e_j, el_t, args.e_sites)
            pred = e_j.sum(dim=1)
            loss = torch.mean((pred - e0_t) ** 2)
            stats["e0_mse"] = float(loss.detach().cpu())
            stats["pearson_e_k"] = float("nan")
        return loss, stats

    def run_epoch(split: str, train: bool, *, epoch: int = 0) -> tuple[float, dict[str, float]]:
        model.train(train)
        losses: list[float] = []
        loss_wsum = 0.0
        loss_wn = 0
        agg: dict[str, list[float]] = {}
        sq_sum = 0.0
        n_sq = 0
        bsz = max(1, int(args.batch))
        rng = np.random.default_rng(int(args.seed) + int(epoch) if train else int(args.seed))

        for n_h in sorted(groups):
            pack = groups[n_h]
            local = group_local[n_h][split]
            if not local:
                continue
            order = list(local)
            if train:
                rng.shuffle(order)
            feat_t = pack["feat"]
            el_t = pack["mm_element"]
            e0_t = pack["e0"]
            ker_t = pack.get("kernel")
            dist_t = pack.get("dist_qm")
            amp_frame_t = pack.get("amp_frame")
            if use_k and ker_t is None:
                raise RuntimeError("k model requires kernel in materialized groups")
            if use_amp and amp_frame_t is None:
                raise RuntimeError("amp model requires amp_frame labels")
            if use_amp_c and amp_frame_t is None:
                raise RuntimeError("amp-c model requires coeff labels (e0_file in cache)")
            for s in range(0, len(order), bsz):
                chunk = order[s : s + bsz]
                idx = torch.tensor(chunk, dtype=torch.long, device=device)
                d_t = None if dist_t is None else dist_t[idx]
                k_t = None if ker_t is None else ker_t[idx]
                af_t = None if amp_frame_t is None else amp_frame_t[idx]
                loss, stats = _batch_loss(feat_t[idx], el_t[idx], e0_t[idx], k_t, d_t, af_t)
                n_chunk = int(len(chunk))
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                else:
                    with torch.no_grad():
                        if use_amp_c:
                            c_pred = model.per_site_coefficients(feat_t[idx], el_t[idx])
                            c_tgt = af_t.unsqueeze(1).expand_as(c_pred)
                            sq = (c_pred - c_tgt).pow(2)
                            sq_sum += float(sq.sum().detach().cpu())
                            n_sq += int(sq.numel())
                        elif use_amp:
                            a_pred = model.forward_amp(feat_t[idx], el_t[idx])
                            sq = (a_pred - af_t).pow(2)
                            sq_sum += float(sq.sum().detach().cpu())
                            n_sq += int(sq.numel())
                        elif use_k:
                            e_j = model.per_site_energies(
                                feat_t[idx], el_t[idx], k_t, dist_qm=d_t
                            )
                            e_j = apply_e_sites_mask(e_j, el_t[idx], args.e_sites)
                            pred = e_j.sum(dim=1)
                            sq = (pred - e0_t[idx]).pow(2)
                            sq_sum += float(sq.sum().detach().cpu())
                            n_sq += int(sq.numel())
                        else:
                            e_j = model.per_site_energies(feat_t[idx], el_t[idx])
                            e_j = apply_e_sites_mask(e_j, el_t[idx], args.e_sites)
                            pred = e_j.sum(dim=1)
                            sq = (pred - e0_t[idx]).pow(2)
                            sq_sum += float(sq.sum().detach().cpu())
                            n_sq += int(sq.numel())
                lv = float(loss.detach().cpu())
                losses.append(lv)
                loss_wsum += lv * float(n_chunk)
                loss_wn += n_chunk
                for key, val in stats.items():
                    agg.setdefault(key, []).append(val)
        means = {k: float(np.mean(v)) for k, v in agg.items()}
        mean_loss = float(loss_wsum / float(loss_wn)) if loss_wn > 0 else float("nan")
        means["loss"] = mean_loss
        if not train and n_sq > 0:
            frame_mse = sq_sum / float(n_sq)
            if use_amp or use_amp_c:
                key = "coeff_mse" if use_amp_c else "amp_mse"
                means[key] = float(frame_mse)
            else:
                means["e0_mse"] = float(frame_mse)
            # Rebuild val total loss from the same global E0 MSE (+ corr) so
            # val_loss is comparable to val_E0_MSE (never spuriously smaller).
            if use_k:
                r = float(means.get("pearson_e_k", float("nan")))
                corr_term = 0.0
                if float(args.weight_corr) > 0.0 and math.isfinite(r):
                    corr_term = max(0.0, float(args.corr_target) - r)
                means["loss"] = (
                    float(args.weight_e0) * float(frame_mse)
                    + float(args.weight_corr) * corr_term
                )
                mean_loss = float(means["loss"])
            elif not use_amp and not use_amp_c:
                means["loss"] = float(frame_mse)
                mean_loss = float(frame_mse)
        return mean_loss, means

    best = float("inf")
    best_ep = -1
    best_state = None
    best_e0_mse = float("nan")
    stale = 0
    if args.monitor is not None:
        monitor_mode = str(args.monitor)
    elif use_k and soft:
        monitor_mode = "loss"
    else:
        monitor_mode = "e0"
    print(f"  monitor={monitor_mode} (*best / early-stop)", flush=True)

    for ep in range(int(args.epochs)):
        tr_loss, tr_stats = run_epoch("train", True, epoch=ep)
        if val_idx:
            va_loss, va_stats = run_epoch("val", False, epoch=ep)
        else:
            va_loss, va_stats = float("nan"), {}

        if use_amp and val_idx:
            monitor = float(va_stats.get("amp_mse", va_loss))
        elif use_amp_c and val_idx:
            monitor = float(va_stats.get("coeff_mse", va_loss))
        elif val_idx and monitor_mode == "loss":
            monitor = float(va_stats.get("loss", va_loss))
        elif val_idx:
            monitor = float(va_stats.get("e0_mse", va_loss))
        else:
            monitor = tr_loss

        improved = monitor < best
        if improved:
            best = monitor
            best_ep = ep
            best_state = copy.deepcopy(model.state_dict())
            best_e0_mse = float(va_stats.get("e0_mse", float("nan"))) if val_idx else float("nan")
            stale = 0
        elif val_idx:
            stale += 1
        mark = " *best" if improved else ""

        if use_k:
            tr_r = tr_stats.get("pearson_e_k", float("nan"))
            if val_idx:
                va_r = va_stats.get("pearson_e_k", float("nan"))
                va_mse = va_stats.get("e0_mse", float("nan"))
                va_tot = va_stats.get("loss", va_loss)
                print(
                    f"epoch {ep:03d}  train loss={tr_loss:.4e}  train_deltaE_MSE={tr_stats.get('e0_mse', float('nan')):.4e}  "
                    f"val_loss={va_tot:.4e}  val_deltaE_MSE={va_mse:.4e}  val_r(e,k)={va_r:+.3f}{mark}"
                )
            else:
                print(
                    f"epoch {ep:03d}  train loss={tr_loss:.4e}  train_deltaE_MSE={tr_stats.get('e0_mse', float('nan')):.4e}  "
                    f"r(e,k)={tr_r:+.3f}{mark}"
                )
        elif use_amp:
            if val_idx:
                va_amp = va_stats.get("amp_mse", float("nan"))
                print(
                    f"epoch {ep:03d}  train_A_MSE={tr_stats.get('amp_mse', tr_loss):.4e}  "
                    f"val_A_MSE={va_amp:.4e}{mark}"
                )
            else:
                print(
                    f"epoch {ep:03d}  train_A_MSE={tr_stats.get('amp_mse', tr_loss):.4e}{mark}"
                )
        elif use_amp_c:
            if val_idx:
                va_c = va_stats.get("coeff_mse", float("nan"))
                print(
                    f"epoch {ep:03d}  train_c_MSE={tr_stats.get('coeff_mse', tr_loss):.4e}  "
                    f"val_c_MSE={va_c:.4e}{mark}"
                )
            else:
                print(
                    f"epoch {ep:03d}  train_c_MSE={tr_stats.get('coeff_mse', tr_loss):.4e}{mark}"
                )
        elif math.isnan(va_loss):
            print(f"epoch {ep:03d}  train_MSE={tr_loss:.6e}{mark}")
        else:
            va_mse = float(va_stats.get("e0_mse", va_loss))
            print(f"epoch {ep:03d}  train_MSE={tr_loss:.6e}  val_MSE={va_mse:.6e}{mark}")

        if val_idx and ep + 1 >= int(args.min_epochs) and stale >= int(args.patience):
            print("early stop")
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_ep = int(args.epochs) - 1

    ckpt_kw: dict = {
        "state_dict": best_state,
        "model": model_kind,
        "d_feat": d_feat,
        "fit_neurons": list(_parse_layers(args.fit_neurons)),
        "corr_neurons": list(_parse_layers(args.corr_neurons)),
        "dropout": float(args.dropout),
        "fix_alpha": meta.get("fix_alpha"),
        "scf": meta.get("scf"),
        "residue_to_id": meta.get("residue_to_id"),
        "best_epoch": best_ep,
        "best_monitor": best if val_idx else None,
        "best_val_e0_mse": best_e0_mse if val_idx and math.isfinite(best_e0_mse) else None,
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "monitor": monitor_mode,
        "corr_target": float(args.corr_target),
        "weight_corr": float(args.weight_corr),
        "e_sites": str(args.e_sites),
        # Same string for SCF amp_repulsion_from_partition (O/Cl → A=0).
        "repulsion_policy": str(args.e_sites),
        "corr_mlp_scale": float(args.corr_mlp_scale),
        "use_dist": not bool(args.no_dist),
        "dist_scale_ang": float(args.dist_scale_ang),
        "partition_mode": "soft" if use_k else None,
        "amp_pool": None,
        "amp_featurizer": None,
        "amp_qm_near_cut": None,
        "amp_label_npz": None,
        "amp_c_formula": "c_label from e0_file, broadcast to all MM sites per frame" if use_amp_c else None,
    }
    if not use_k:
        ckpt_kw["best_val_mse"] = best if val_idx else None
    else:
        ckpt_kw["best_val_mse"] = best if val_idx else None

    if use_k and isinstance(model, SoapE0MmhKModel):
        ckpt_kw["element_scales"] = model.element_scales()

    torch.save(ckpt_kw, args.ckpt)
    if use_k and val_idx:
        e0_mse_print = best_e0_mse if math.isfinite(best_e0_mse) else float(best)
        e0_rmse = math.sqrt(max(e0_mse_print, 0.0))
        if monitor_mode == "loss":
            print(
                f"wrote {args.ckpt}  best_val_loss={best:.4e}  "
                f"best_val_deltaE_MSE={e0_mse_print:.4e}  "
                f"best_val_deltaE_RMSE={e0_rmse:.4f} kcal/mol  epoch={best_ep}"
            )
        else:
            print(
                f"wrote {args.ckpt}  val_deltaE_RMSE={e0_rmse:.4f} kcal/mol (epoch {best_ep})"
            )
    else:
        print(f"wrote {args.ckpt}")


if __name__ == "__main__":
    main()
