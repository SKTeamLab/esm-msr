#!/usr/bin/env python3
"""Offline-rescore validation dumps and retroactively backfill new & 1x-comb validation metrics to Comet ML.

Why ``val_epi_*_comb`` alongside ``val_epi_*_epi``:
  Comet ML's ``write/experiment/metric`` endpoint is strictly append-only (``editable: False``): logging new values
  onto an already-existing metric key appends duplicate points at the same step (creating vertical sawtooth curves)
  and leaves ``valueMax`` / ``valueMin`` stuck at ``max(old, new)`` / ``min(old, new)``. Because ``val_epi_*_comb``
  and all 17 new ``val_{nat,des,mutbg,diag}_*`` / ``val_epi_composite`` keys were never logged on historical runs,
  backfilling them at each validation epoch's exact ``(step, epoch, timestamp)`` produces 100% clean single-valued
  curves and exact ``valueMax`` / ``valueMin`` / ``valueCurrent`` summaries across every historical and new model
  in ``esm-msr-oct24-final``.

Idempotency & alignment guarantees:
  1. Verifies ``abs(offline_rho_singles_wt_avg - comet_rho_singles_wt_avg) < 2e-4`` on every matched ``(step, epoch)``
     point before uploading any metric for that epoch.
  2. Checks existing logged steps per metric on Comet and skips any ``(metric, step)`` already present, so re-running
     this script is a safe no-op and partially-logged active runs (e.g. ``audit_clean_base_s1-0``) only receive
     missing metric keys.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_MUT_KEY_CACHE: List[Tuple[np.ndarray, list]] = []

import comet_ml
import numpy as np
import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from esm_msr import censoring, epi_metrics, link as link_mod, stats, utils  # noqa: E402
from esm_msr.training import (  # noqa: E402
    _parse_single_aa_pair,
    classify_megascale_library,
    compute_nat_debiased_rho,
)

BACKFILL_METRIC_KEYS = [
    # Split ID validation summaries
    "val_nat_rho_singles_wt_avg",
    "val_nat_rho_combined_avg",
    "val_des_rho_singles_wt_avg",
    "val_des_rho_combined_avg",
    "val_mutbg_rho_singles_wt_avg",
    "val_mutbg_rho_combined_avg",
    "val_diag_des_minus_nat",
    "val_nat_alive_rho_singles_wt_avg",
    "val_nat_debiased_rho_singles_wt_avg",
    # Clean unsaturated & legacy-bridge diagnostics
    "val_diag_wt_doubles_clean_rmse",
    "val_diag_wt_doubles_clean_slope",
    "val_diag_epi_clean_err_comb",
    "val_diag_epi_clean_bias_comb",
    "val_diag_epi_clean_err_epi",
    "val_diag_epi_clean_bias_epi",
    "val_diag_epi_cell_rank_mt_legacy",
    "val_diag_norm_lora_wt",
    "val_diag_norm_lora_mt",
    # Active composite monitor (1x comb)
    "val_epi_composite",
    # Full 1x ensemble epistasis hierarchy (never previously logged on historical runs)
    "val_epi_naive_rho_comb",
    "val_epi_global_err_comb",
    "val_epi_global_bias_comb",
    "val_epi_beyond_rho_comb",
    "val_epi_beyond_lib_comb",
    "val_epi_beyond_rho_pairavg_comb",
    "val_epi_pair_rho_comb",
    "val_epi_line_rho_comb",
    "val_epi_cell_mag_comb",
    "val_epi_cell_rank_comb",
    "val_epi_cell_flipacc_comb",
    "val_epi_class3_acc_comb",
    "val_epi_class3_pos_recall_comb",
    "val_epi_class3_neg_recall_comb",
]


def build_wt_aa_map(cache_dir: str, loaders: List[str]) -> Dict[Tuple[str, int, str], Tuple[str, str]]:
    """Map ``(library, 1-indexed_pos, mt_aa) -> (wt_aa, mt_aa)`` from c6 validation pickles."""
    wt_aa_map: Dict[Tuple[str, int, str], Tuple[str, str]] = {}
    for lib in loaders:
        pkl_path = os.path.join(cache_dir, f"{lib}_ddG_ML_c6.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            items = pickle.load(f)["items"]
        for it in items:
            m = it.get("mutations")
            ap = _parse_single_aa_pair(m)
            if ap is None:
                continue
            parts = [p.strip() for p in str(m).split(":") if p.strip()] if isinstance(m, str) else list(m)
            p0 = parts[0]
            pos = int(p0[1]) if isinstance(p0, (list, tuple)) else int(str(p0)[1:-1])
            wt_aa_map[(lib, pos, ap[1])] = ap
    return wt_aa_map


def rescore_val_dump(
    npz_path: str,
    wt_aa_map: Dict[Tuple[str, int, str], Tuple[str, str]],
    force_no_link: bool = False,
) -> Dict[str, float]:
    """Recompute all validation metrics from a saved ``val_dump_<tag>.npz``."""
    d = np.load(npz_path, allow_pickle=True)
    wt = np.asarray(d["wt"], float)
    mt = np.asarray(d["mt"], float)
    comb = np.asarray(d["comb"], float)
    wt_rev = np.asarray(d["wt_rev"], float) if "wt_rev" in d else np.full_like(wt, np.nan)
    gt = np.asarray(d["ground_truth"], float)
    dddG = np.asarray(d["dddG"], float)
    dG_wt = np.asarray(d["dG_wt"], float)
    y_floor = np.asarray(d["y_floor"], float)
    cens = np.asarray(d["cens"], int)
    cens_src = np.asarray(d["cens_src"], int)
    subset_types = [str(s) for s in d["subset_type"]]
    loader_arr = np.asarray([str(x) for x in d["loader"]])
    raw_mk = d["mut_key"]
    keys = None
    for cached_raw, cached_parsed in _MUT_KEY_CACHE:
        if cached_raw.shape == raw_mk.shape and np.array_equal(cached_raw, raw_mk):
            keys = cached_parsed
            break
    if keys is None:
        keys = [
            tuple(tuple(m) for m in json.loads(str(k))) if json.loads(str(k)) is not None else None
            for k in raw_mk
        ]
        _MUT_KEY_CACHE.append((raw_mk, keys))

    ls = json.loads(str(d["link_summary"])) if "link_summary" in d else {}
    if force_no_link or ls.get("mode") == "identity" or ls.get("tau") is None:
        lnk = None
    else:
        lnk = link_mod.numpy_link_from_summary(ls)

    loaders = sorted(set(loader_arr.tolist()))
    all_singles_wt = []
    cat_metrics = {
        "natural_wt": {"rho_singles_wt": [], "rho_combined": []},
        "designed": {"rho_singles_wt": [], "rho_combined": []},
        "mutant_bg": {"rho_singles_wt": [], "rho_combined": []},
    }
    nat_alive_rhos: List[float] = []
    nat_single_records = []

    for lib in loaders:
        idx = np.flatnonzero(loader_arr == lib)
        w_l, m_l, c_l, g_l = wt[idx], mt[idx], comb[idx], gt[idx]
        dg_l, yf_l, cs_l = dG_wt[idx], y_floor[idx], cens_src[idx]
        st_l = [subset_types[i] for i in idx]
        canon = cs_l != censoring.SRC_RANGE
        sub = lambda a: np.asarray(a)[canon]
        if lnk is not None:
            obs_l = {
                "wt": lnk(dg_l + w_l, yf_l) - dg_l,
                "mt": lnk(dg_l + m_l, yf_l) - dg_l,
                "comb": lnk(dg_l + c_l, yf_l) - dg_l,
            }
        else:
            obs_l = None
        met = stats.compute_metrics(
            sub(w_l),
            sub(m_l),
            sub(c_l),
            sub(g_l),
            [t for t, k in zip(st_l, canon) if k],
            obs=None if obs_l is None else {k: sub(v) for k, v in obs_l.items()},
        )
        if met.get("rho_singles_wt") is not None and np.isfinite(met["rho_singles_wt"]):
            all_singles_wt.append(float(met["rho_singles_wt"]))
        cat = classify_megascale_library(lib)
        for mk in ("rho_singles_wt", "rho_combined"):
            v = met.get(mk)
            if v is not None and np.isfinite(v):
                cat_metrics[cat][mk].append(float(v))
        if cat == "natural_wt":
            is_sng = np.asarray([t == "single" for t in st_l], dtype=bool)
            alive_m = (
                canon
                & is_sng
                & np.isfinite(g_l)
                & np.isfinite(w_l)
                & np.isfinite(dg_l)
                & ((dg_l + g_l) > 1.0)
            )
            if alive_m.sum() >= 5:
                ra = stats.safe_spearman(w_l[alive_m], g_l[alive_m])
                if np.isfinite(ra):
                    nat_alive_rhos.append(float(ra))
            deb_idx = np.flatnonzero(canon & is_sng & np.isfinite(g_l) & np.isfinite(w_l))
            w_sub, g_sub, aa_pairs = [], [], []
            for ii in deb_idx:
                ki = keys[idx[ii]]
                if ki and len(ki) == 1:
                    pos = int(ki[0][-2])
                    mt_aa = str(ki[0][-1])
                    ap = wt_aa_map.get((lib, pos, mt_aa))
                    if ap is not None:
                        w_sub.append(float(w_l[ii]))
                        g_sub.append(float(g_l[ii]))
                        aa_pairs.append(ap)
            if len(w_sub) >= 5:
                nat_single_records.append((lib, w_sub, g_sub, aa_pairs))

    out: Dict[str, float] = {}
    if all_singles_wt:
        out["_check_val_rho_singles_wt_avg"] = float(np.nanmean(all_singles_wt))
    for prefix, ck in (("nat", "natural_wt"), ("des", "designed"), ("mutbg", "mutant_bg")):
        for mk in ("rho_singles_wt", "rho_combined"):
            vals = cat_metrics[ck][mk]
            if vals:
                out[f"val_{prefix}_{mk}_avg"] = float(np.nanmean(vals))
    if "val_nat_rho_singles_wt_avg" in out and "val_des_rho_singles_wt_avg" in out:
        out["val_diag_des_minus_nat"] = out["val_des_rho_singles_wt_avg"] - out["val_nat_rho_singles_wt_avg"]
    if nat_alive_rhos:
        out["val_nat_alive_rho_singles_wt_avg"] = float(np.mean(nat_alive_rhos))
    if nat_single_records:
        v_deb = compute_nat_debiased_rho(nat_single_records)
        if np.isfinite(v_deb):
            out["val_nat_debiased_rho_singles_wt_avg"] = float(v_deb)

    canon_all = cens_src != censoring.SRC_RANGE
    items_all = {
        "mut_key": [keys[i] for i in np.flatnonzero(canon_all)],
        "subset_type": [subset_types[i] for i in np.flatnonzero(canon_all)],
        "cens": cens[canon_all],
        "cens_src": cens_src[canon_all],
        "ddG": gt[canon_all],
        "dddG": dddG[canon_all],
        "dG_wt": dG_wt[canon_all],
        "y_floor": y_floor[canon_all],
        "wt": wt[canon_all],
        "mt": mt[canon_all],
        "comb": comb[canon_all],
        "wt_rev": wt_rev[canon_all],
    }
    tbl = epi_metrics.Table(items_all, lnk)
    res = epi_metrics.compute(tbl)
    for k, v in res.items():
        if np.isfinite(v):
            out[f"val_{k}"] = float(v)
    diag = epi_metrics.compute_clean_diagnostics(tbl)
    for k, v in diag.items():
        if np.isfinite(v):
            out[f"val_diag_{k}"] = float(v)
    cr = res.get("epi_cell_rank_comb", float("nan"))
    bl = res.get("epi_beyond_lib_comb", float("nan"))
    if np.isfinite(cr) and np.isfinite(bl):
        out["val_epi_composite"] = 0.5 * (float(cr) + float(bl))
    return out


def extract_ckpt_lora_norms(log_dir: Optional[str]) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """Read exact ``(lora_wt_norm, lora_mt_norm)`` per epoch from saved ``.ckpt`` files in ``log_dir``."""
    norms: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    if not log_dir or not os.path.isdir(log_dir):
        return norms
    for ck in sorted(glob.glob(os.path.join(log_dir, "*.ckpt"))):
        try:
            obj = torch.load(ck, map_location="cpu", weights_only=False)
            ep_raw = obj.get("epoch")
            m = re.search(r"epoch=0*(\d+)", os.path.basename(ck))
            ep = int(m.group(1)) if m else (int(ep_raw) if ep_raw is not None else None)
            if ep is None:
                continue
            sd = obj.get("state_dict", {})
            wt_p, mt_p = [], []
            for k, v in sd.items():
                lk = k.lower()
                if "lora_" not in lk or not torch.is_tensor(v):
                    continue
                if "mt_adapter" in lk or "peft_mt" in lk:
                    mt_p.append((k, v))
                elif "wt_adapter" in lk or "peft_wt" in lk or "default" in lk:
                    wt_p.append((k, v))
            nw = float(utils.l2_weight_norm(wt_p)) if wt_p else None
            nm = float(utils.l2_weight_norm(mt_p)) if mt_p else None
            norms[ep] = (nw, nm)
        except Exception:
            continue
    return norms


def lookup_step_lora_norm(step_series: List[dict], target_step: int) -> Optional[float]:
    """Fallback to Comet's per-25-step ``norm_weight/lora_{wt,mt}`` series."""
    if not step_series:
        return None
    pts = sorted(
        [(int(p["step"] or 0), float(p["metricValue"])) for p in step_series if p.get("metricValue") is not None],
        key=lambda x: x[0],
    )
    if not pts:
        return None
    if target_step == 0:
        return pts[0][1]
    le = [v for s, v in pts if s <= target_step]
    return le[-1] if le else pts[0][1]


def find_log_dir(exp_name: str, search_roots: List[str]) -> Optional[str]:
    base = exp_name[:-2] if exp_name.endswith("-0") else exp_name
    for r in search_roots:
        cand = os.path.join(r, base, "0")
        if os.path.isdir(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default="sareeves96")
    ap.add_argument("--project", default="esm-msr-oct24-final")
    ap.add_argument("--cache_dir", default="/home/sareeves/playground/esm-msr-devel/cache_devel_conservative")
    ap.add_argument(
        "--log_roots",
        nargs="+",
        default=[
            "/home/sareeves/playground/esm-msr-devel/dc_runs/logs",
            "/home/sareeves/playground/esm-msr-devel/training_logs",
        ],
    )
    ap.add_argument("--fallback_zs_dump", default="/home/sareeves/playground/esm-msr-devel/dc_runs/logs/trim_joint_ctrl_s1/0/val_dump_zs.npz")
    ap.add_argument("--experiment_name", default=None, help="Optional single Comet experiment name to process")
    ap.add_argument(
        "--rescore_cache_json",
        default="/home/sareeves/playground/esm-msr-devel/wt_audit_next/docs/.backfill_rescore_cache.json",
    )
    ap.add_argument(
        "--csv_out",
        default="/home/sareeves/playground/esm-msr-devel/wt_audit_next/docs/comet_oct24_final_rescored_all_epochs.csv",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    # Discover validation library names from fallback dump and build (wt_aa, mt_aa) map once.
    d_ref = np.load(args.fallback_zs_dump, allow_pickle=True)
    val_loaders = sorted(set(str(x) for x in d_ref["loader"]))
    wt_aa_map = build_wt_aa_map(args.cache_dir, val_loaders)
    print(f"[init] Loaded {len(wt_aa_map)} single-mutant (wt_aa, mt_aa) pairs across {len(val_loaders)} ID val libraries.")

    api = comet_ml.API()
    exps = api.get_experiments(args.workspace, args.project)
    if args.experiment_name:
        exps = [e for e in exps if e.get_name() == args.experiment_name]
    print(f"[init] Found {len(exps)} Comet experiment(s) in {args.workspace}/{args.project}.")

    dump_cache: Dict[str, Dict[str, float]] = {}
    if args.rescore_cache_json and os.path.exists(args.rescore_cache_json):
        try:
            with open(args.rescore_cache_json, "r") as f:
                dump_cache = json.load(f)
            print(f"[init] Loaded {len(dump_cache)} cached rescored dumps from {args.rescore_cache_json}.")
        except Exception:
            dump_cache = {}
    csv_rows: List[dict] = []
    total_logged_calls = 0
    total_matched_epochs = 0
    total_skipped_existing = 0

    for idx_e, e in enumerate(exps, 1):
        ename = e.get_name()
        log_dir = find_log_dir(ename, args.log_roots)
        force_no_link = "no_link" in ename
        ref_pts = e.get_metrics("val_rho_singles_wt_avg")
        if not ref_pts:
            print(f"[{idx_e:02d}/{len(exps)}] {ename:40s} ({e.id[:8]}): 0 validation points, skipping.")
            continue

        # Deduplicate by (step, epoch), keeping latest timestamp per (step, epoch).
        by_se: Dict[Tuple[int, int], dict] = {}
        for p in sorted(ref_pts, key=lambda x: int(x["timestamp"])):
            se = (int(p["step"] or 0), int(p["epoch"] or 0))
            by_se[se] = p
        ordered_pts = [by_se[k] for k in sorted(by_se.keys())]

        ckpt_norms = extract_ckpt_lora_norms(log_dir)
        nw_series = e.get_metrics("norm_weight/lora_wt")
        nm_series = e.get_metrics("norm_weight/lora_mt")

        # Fetch existing steps per target metric once so re-runs are idempotent.
        summary_names = {x["name"] for x in e.get_metrics_summary()}
        existing_steps: Dict[str, set] = {}
        for mk in BACKFILL_METRIC_KEYS:
            if mk in summary_names:
                existing_steps[mk] = {int(p["step"] or 0) for p in e.get_metrics(mk)}
            else:
                existing_steps[mk] = set()

        pending_calls: List[Tuple[str, float, int, int, int]] = []
        matched_tags = []
        max_align_err = 0.0

        for p in ordered_pts:
            step = int(p["step"] or 0)
            epoch = int(p["epoch"] or 0)
            ts_ms = int(p["timestamp"])
            c_rho = float(p["metricValue"])
            tag = "zs" if step == 0 else f"e{epoch}"
            npz_path = os.path.join(log_dir, f"val_dump_{tag}.npz") if log_dir else None
            if (not npz_path or not os.path.exists(npz_path)) and tag == "zs" and os.path.exists(args.fallback_zs_dump):
                npz_path = args.fallback_zs_dump
            if not npz_path or not os.path.exists(npz_path):
                print(f"  [WARN] {ename} ({e.id[:8]}): missing dump for tag={tag} (step={step}, epoch={epoch}); skipping point.")
                continue

            cache_key = f"{os.path.realpath(npz_path)}|nolink={int(force_no_link)}"
            if cache_key not in dump_cache:
                dump_cache[cache_key] = rescore_val_dump(npz_path, wt_aa_map, force_no_link=force_no_link)
            resc = dict(dump_cache[cache_key])

            off_rho = resc.get("_check_val_rho_singles_wt_avg", float("nan"))
            align_err = abs(off_rho - c_rho)
            if not np.isfinite(align_err) or align_err > 2e-4:
                print(
                    f"  [SKIP MISMATCH] {ename} ({e.id[:8]}) tag={tag} step={step}: "
                    f"offline_rho={off_rho:.6f} vs comet_rho={c_rho:.6f} (diff={align_err:.2e})"
                )
                continue
            max_align_err = max(max_align_err, align_err)

            # Attach LoRA adapter Frobenius norms
            nw_val, nm_val = (None, None)
            if step > 0 and epoch in ckpt_norms:
                nw_val, nm_val = ckpt_norms[epoch]
            if nw_val is None:
                nw_val = lookup_step_lora_norm(nw_series, step)
            if nm_val is None:
                nm_val = lookup_step_lora_norm(nm_series, step)
            if nw_val is not None and np.isfinite(nw_val):
                resc["val_diag_norm_lora_wt"] = float(nw_val)
            if nm_val is not None and np.isfinite(nm_val):
                resc["val_diag_norm_lora_mt"] = float(nm_val)

            matched_tags.append(tag)
            total_matched_epochs += 1
            row = {"experiment_key": e.id, "experiment_name": ename, "tag": tag, "step": step, "epoch": epoch, "val_rho_singles_wt_avg": c_rho}
            for mk in BACKFILL_METRIC_KEYS:
                row[mk] = resc.get(mk, float("nan"))
            csv_rows.append(row)

            for mk in BACKFILL_METRIC_KEYS:
                if mk not in resc or not np.isfinite(resc[mk]):
                    continue
                if step in existing_steps[mk]:
                    total_skipped_existing += 1
                    continue
                pending_calls.append((mk, float(resc[mk]), step, epoch, ts_ms))

        print(
            f"[{idx_e:02d}/{len(exps)}] {ename:40s} ({e.id[:8]}) | "
            f"matched={len(matched_tags)}/{len(ordered_pts)} {matched_tags} | "
            f"max_rho_diff={max_align_err:.2e} | new_calls={len(pending_calls)}",
            flush=True,
        )

        if not args.dry_run and pending_calls:
            # Group by metric key so each metric's steps upload strictly in ascending chronological order.
            by_metric: Dict[str, List[Tuple[str, float, int, int, int]]] = {}
            for item in pending_calls:
                by_metric.setdefault(item[0], []).append(item)

            def _post_metric_series(series: List[Tuple[str, float, int, int, int]]):
                for mk_i, val_i, st_i, ep_i, ts_i in sorted(series, key=lambda x: (x[2], x[3], x[4])):
                    for attempt in range(3):
                        try:
                            e.log_metric(mk_i, val_i, step=st_i, epoch=ep_i, timestamp=ts_i / 1000.0)
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                            time.sleep(0.5 * (attempt + 1))
                return True

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(_post_metric_series, by_metric.values()))
            total_logged_calls += len(pending_calls)

    if args.rescore_cache_json:
        os.makedirs(os.path.dirname(args.rescore_cache_json), exist_ok=True)
        with open(args.rescore_cache_json, "w") as f:
            json.dump(dump_cache, f)
    if args.csv_out and csv_rows and args.experiment_name is None:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        cols = ["experiment_key", "experiment_name", "tag", "step", "epoch", "val_rho_singles_wt_avg"] + BACKFILL_METRIC_KEYS
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"[saved] Wrote {len(csv_rows)} rescored epoch rows to {args.csv_out}")

    mode = "DRY-RUN" if args.dry_run else "LIVE BACKFILL"
    print(
        f"\n[{mode} COMPLETE] matched_epoch_points={total_matched_epochs} | "
        f"metric_points_logged={total_logged_calls} | skipped_already_present={total_skipped_existing}"
    )


if __name__ == "__main__":
    main()
