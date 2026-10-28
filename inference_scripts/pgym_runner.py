"""ProteinGym (all-DMS) benchmark loop shared by the baseline testing scripts.

Every model goes through this one loop, so their results are directly comparable: reads the per-DMS CSVs + manifest.csv written by
preprocessing/pgym_preprocess.py, scores each DMS with a model-specific score_fn, writes <out>/<DMS_id>.csv
(predictions + DMS_score), an incremental <out>/summary.csv (per-DMS Spearman of pred_col vs DMS_score) and
<out>/run.log. Any DMS whose output CSV already exists is skipped, so runs can be interrupted and restarted.

Used by esm_msr_testing.py, thermompnnd_testing.py (--protein_gym) and the baselines/ scripts (ThermoMPNN, SPURS, ProteinMPNN; MutateEverything runs ProteinGym through its own loop).
Kept free of esm_msr imports so it runs in any environment (e.g. the SPURS env, whose `esm` is fair-esm).
"""
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import torch
except ImportError:
    torch = None


def _cuda():
    return torch is not None and torch.cuda.is_available()


class _NullTracker:
    """Stand-in when no resource_tracker.Tracker is passed, so the loop needs no branches."""
    class _S:
        row = None
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def scope(self, *a, **k): return self._S()


def _peak_vram():
    """Peak GiB allocated / reserved by PyTorch on the current device since the last reset (None without CUDA)."""
    if not _cuda():
        return {"peak_alloc_gb": None, "peak_reserved_gb": None}
    return {"peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)}


def safe_spearman(preds, targets):
    # same rule as esm_msr.stats.safe_spearman, over the rows that have a prediction (a model that cannot score some
    # variants, e.g. MutateEverything on two substitutions at one position, leaves them NaN)
    preds, targets = np.asarray(preds, dtype=float), np.asarray(targets, dtype=float)
    ok = ~np.isnan(preds)
    preds, targets = preds[ok], targets[ok]
    if len(preds) < 2 or len(np.unique(targets)) < 2: return np.nan
    if len(np.unique(preds)) < 2: return np.nan
    rho = spearmanr(preds, targets)[0]
    return np.nan if np.isnan(rho) else float(rho)


# columns of each output CSV that come from the input rather than from score_fn
INPUT_COLS = ('code', 'chain', 'mut_type_renumbered', 'DMS_id', 'mutant', 'DMS_score')


def make_log(out_dir):
    """Logger that prints a timestamped line and appends it to <out_dir>/run.log."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(out_dir / "run.log", 'a') as f:
            f.write(line + "\n")
    return log


def run_protein_gym(pgym_dir, out_dir, score_fn, pred_col, dms='all', tag='', extra_cols=(), log=None, tracker=None):
    """Score every (or the selected) ProteinGym DMS with score_fn and write per-DMS CSVs + summary.csv.

    score_fn(df) gets one DMS's input rows (pdb_file resolved to an absolute path; columns pdb_file, code (the
    DMS_id), chain, mut_type_renumbered, mutant, DMS_score) and returns a DataFrame aligned row-for-row with it
    holding the prediction columns, or (DataFrame, info): info is a dict reported as key=value on the DMS's log
    line. pred_col is the column correlated with DMS_score (higher = fitter); each column in extra_cols also gets
    a spearman_<col> entry in summary.csv. log: a make_log logger to share with the caller (e.g. so a batch
    sizer writes to the same run.log); one is made for out_dir if None.

    Each scored (or failed) DMS's summary row records its wall time (time_s, from reading the input to writing the
    output), the peak GPU memory PyTorch allocated and reserved while scoring it (peak_alloc_gb, peak_reserved_gb;
    the counters are reset per DMS), the GPU and the host.

    tracker: a resource_tracker.Tracker. Each DMS's score_fn call (which includes the model's own input handling) is
    recorded as a `dms` scope inside one `dataset` scope named `proteingym`; the summary row also gets `ram_peak_gb`
    and `cpu_cores` from it. The model-specific wrapper adds `protein` scopes around its inference calls only, and
    those are the rows to read for inference cost. Without a tracker nothing changes.
    """
    if tracker is None:
        tracker = _NullTracker()
    pgym_dir = Path(pgym_dir).resolve()
    if not pgym_dir.exists():
        raise AssertionError(f"pgym_dir does not exist: {pgym_dir}")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(pgym_dir / "manifest.csv")
    pg_root = manifest["proteingym_dir"].iloc[0] if "proteingym_dir" in manifest.columns else None
    if dms and str(dms).lower() != 'all':
        dms_list = [d.strip() for d in str(dms).split(',') if d.strip()]
    else:
        dms_list = sorted(manifest["DMS_id"].tolist())

    log = log or make_log(out_dir)

    host = os.uname().nodename
    gpu = torch.cuda.get_device_name() if _cuda() else "cpu"
    gpu_gb = f" ({torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30:.1f} GiB)" if _cuda() else ""
    log(f"[PGYM] START {tag} n_dms={len(dms_list)} pred_col={pred_col} host={host} gpu={gpu}{gpu_gb} out={out_dir}")
    # Keep rows from earlier invocations (other DMS subsets, or measured times of DMS skipped now).
    summary_rows = []
    prev_by_dms = {}
    if (out_dir / "summary.csv").exists():
        old = pd.read_csv(out_dir / "summary.csv")
        if "DMS_id" in old.columns:
            prev_by_dms = {row["DMS_id"]: row for _, row in old.iterrows()}
            summary_rows = old[~old["DMS_id"].isin(dms_list)].to_dict("records")

    def spearmans(res):
        return {f"spearman_{c}": safe_spearman(res[c], res["DMS_score"]) for c in extra_cols if c in res}

    t_start = time.time()
    dataset_scope = tracker.scope('dataset', 'proteingym', n_dms=len(dms_list))
    dataset_scope.__enter__()
    for i, did in enumerate(dms_list, 1):
        out_csv = out_dir / f"{did}.csv"
        if out_csv.exists():
            log(f"[PGYM {i}/{len(dms_list)}] SKIP {did} (output exists)")
            prev_row = prev_by_dms.get(did)
            if prev_row is not None and pd.notna(prev_row.get("time_s")):
                summary_rows.append(dict(prev_row))
            else:
                try:
                    prev = pd.read_csv(out_csv)
                    summary_rows.append({"DMS_id": did, "n": len(prev),
                                         "spearman_combined": safe_spearman(prev[pred_col], prev["DMS_score"]),
                                         **spearmans(prev), "time_s": None, "units_per_s": None, "status": "skipped"})
                except Exception:
                    summary_rows.append({"DMS_id": did, "n": None, "spearman_combined": None,
                                         "time_s": None, "units_per_s": None, "status": "skipped_bad"})
            pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
            continue
        if _cuda():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            df = pd.read_csv(pgym_dir / f"{did}.csv")
            pdb_val = df["pdb_file"].iloc[0]
            if not os.path.isabs(pdb_val):
                if not pg_root:
                    raise AssertionError(f"relative pdb_file '{pdb_val}' but manifest.csv has no 'proteingym_dir' "
                                         "column; regenerate pgym_inputs via preprocessing/pgym_preprocess.py")
                pdb_res = (Path(pg_root) / pdb_val).resolve()
                if not pdb_res.exists():
                    raise AssertionError(f"pdb_file not found: {pdb_res}")
                df["pdb_file"] = str(pdb_res)
            with tracker.scope('dms', did, n_rows=len(df)) as sc:
                preds = score_fn(df)
            info = {}
            if isinstance(preds, tuple):
                preds, info = preds
            preds = preds.drop(columns=[c for c in INPUT_COLS if c in preds.columns])
            assert len(preds) == len(df), f"len(preds)={len(preds)} != len(df)={len(df)}"
            res = df[["code", "chain", "mut_type_renumbered"]].reset_index(drop=True)
            res = pd.concat([res, preds.reset_index(drop=True)], axis=1)
            res["DMS_id"] = did
            res["mutant"] = df["mutant"].values
            res["DMS_score"] = df["DMS_score"].values
            res.to_csv(out_csv, index=False)
            rho = safe_spearman(res[pred_col], res["DMS_score"])
            dt = time.time() - t0
            mem = _peak_vram()
            summary_rows.append({"DMS_id": did, "n": len(res), "spearman_combined": rho, **spearmans(res),
                                 "time_s": dt, "units_per_s": len(res) / dt, **mem, "gpu": gpu, "host": host,
                                 **({"ram_peak_gb": sc.row["ram_rss_peak_gb"], "cpu_cores": sc.row["cpu_cores"]}
                                    if sc.row else {}),
                                 "status": "done"})
            info_s = ''.join(f" {k}={v}" for k, v in info.items())
            log(f"[PGYM {i}/{len(dms_list)}] DONE {did}: n={len(res)}{info_s} rho={rho:.4f} {dt:.1f}s ({len(res)/dt:.1f} u/s) "
                f"peak_vram={mem['peak_alloc_gb']}/{mem['peak_reserved_gb']} GiB (alloc/reserved)")
        except Exception as e:
            dt = time.time() - t0
            mem = _peak_vram()
            log(f"[PGYM {i}/{len(dms_list)}] FAIL {did}: {type(e).__name__}: {str(e)[:300]} after {dt:.1f}s "
                f"peak_vram={mem['peak_alloc_gb']}/{mem['peak_reserved_gb']} GiB (alloc/reserved)")
            log(traceback.format_exc())
            summary_rows.append({"DMS_id": did, "n": None, "spearman_combined": None, "time_s": dt, "units_per_s": None,
                                 **mem, "gpu": gpu, "host": host, "status": "fail", "error": str(e)[:200]})
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    dataset_scope.__exit__(None, None, None)
    s = pd.DataFrame(summary_rows)
    s.to_csv(out_dir / "summary.csv", index=False)
    valid = s.dropna(subset=["spearman_combined"])
    log(f"[PGYM] COMPLETE {tag}: {len(valid)}/{len(s)} DMS scored | "
        f"mean_rho={valid['spearman_combined'].mean():.4f} median_rho={valid['spearman_combined'].median():.4f} "
        f"n_pos={(valid['spearman_combined'] > 0).sum()} n_neg={(valid['spearman_combined'] < 0).sum()} "
        f"wall={(time.time() - t_start)/3600:.2f}h")
    print("\n[PGYM] final summary:\n", s.to_string(index=False))
    return s
