import pandas as pd
import os
import sys
import torch
from tqdm import tqdm
import argparse
import time
import json
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from huggingface_hub import login, get_token
from esm_msr import stats, utils, models, inference, preprocess_megascale, auto_batch

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = REPO_ROOT / "data" / "preprocessed"
MODEL_DIR = REPO_ROOT / "LoRA_models"

def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def safe_spearman(df, col1, col2):
    """Safely computes spearman correlation, returning NaN if insufficient valid data."""
    valid_df = df[[col1, col2]].dropna()
    if len(valid_df) < 2:
        return float('nan')
    return valid_df.corr('spearman').iloc[0, 1]

def safe_ndcg(df, col1, col2, top_n=None, threshold=None):
    """Safely computes NDCG, returning NaN if insufficient valid data."""
    valid_df = df[[col1, col2]].dropna()
    if len(valid_df) < 2:
        return float('nan')
    preds = valid_df[col1].to_numpy().reshape(1, -1)
    truths = valid_df[col2].to_numpy().reshape(1, -1)
    ndcg_val, model_hits_at_k, ideal_hits_at_k, total_hits_in_pool = stats.compute_ndcg_flexible(preds, truths, top_n=top_n, threshold=threshold)
    return ndcg_val

def compute_flip_stats(res_df, pred_col, true_col, code_name=''):
    """Extracts double mutant flip columns and computes stats.flip_signature_rho.
    
    A flip column is one scored position with one fixed partner identity.
    Each double mutant (pos1, mut1) : (pos2, mut2) contributes to two flip columns:
      1) Scored at pos1 (mut1) with partner at pos2 (mut2)
      2) Scored at pos2 (mut2) with partner at pos1 (mut1)
    """
    if pred_col not in res_df.columns or true_col not in res_df.columns:
        return float('nan'), float('nan'), float('nan')
        
    valid_df = res_df.dropna(subset=[pred_col, true_col])
    if len(valid_df) == 0:
        return float('nan'), float('nan'), float('nan')

    fk_list, rid_list, pred_list, tgt_list = [], [], [], []
    mut1_col = 'mut1' if 'mut1' in valid_df.columns else ('to1' if 'to1' in valid_df.columns else None)
    mut2_col = 'mut2' if 'mut2' in valid_df.columns else ('to2' if 'to2' in valid_df.columns else None)
    has_pos_cols = 'pos1' in valid_df.columns and 'pos2' in valid_df.columns and mut1_col and mut2_col
    
    if has_pos_cols:
        doubles = valid_df[valid_df['pos1'].notnull() & valid_df['pos2'].notnull()]
        for _, row in doubles.iterrows():
            try:
                p1, m1 = int(row['pos1']), str(row[mut1_col])
                p2, m2 = int(row['pos2']), str(row[mut2_col])
                c = str(row.get('code_wt', row.get('code', code_name)))
                
                fk_list.append(f"{c}|{p1}|{p2}{m2[-1]}")
                rid_list.append(ord(m1[-1]))
                pred_list.append(float(row[pred_col]))
                tgt_list.append(float(row[true_col]))
                
                fk_list.append(f"{c}|{p2}|{p1}{m1[-1]}")
                rid_list.append(ord(m2[-1]))
                pred_list.append(float(row[pred_col]))
                tgt_list.append(float(row[true_col]))
            except (ValueError, TypeError):
                continue
    elif 'mut_type' in valid_df.columns or 'mut_info' in valid_df.columns:
        col = 'mut_type' if 'mut_type' in valid_df.columns else 'mut_info'
        doubles = valid_df[valid_df[col].astype(str).str.contains(':')]
        import re
        pat = re.compile(r'^([A-Za-z])(\d+)([A-Za-z]):([A-Za-z])(\d+)([A-Za-z])$')
        for _, row in doubles.iterrows():
            m = pat.match(str(row[col]))
            if m:
                _, p1, m1, _, p2, m2 = m.groups()
                p1, p2 = int(p1), int(p2)
                c = str(row.get('code_wt', row.get('code', code_name)))
                
                fk_list.append(f"{c}|{p1}|{p2}{m2}")
                rid_list.append(ord(m1))
                pred_list.append(float(row[pred_col]))
                tgt_list.append(float(row[true_col]))
                
                fk_list.append(f"{c}|{p2}|{p1}{m1}")
                rid_list.append(ord(m2))
                pred_list.append(float(row[pred_col]))
                tgt_list.append(float(row[true_col]))

    if not fk_list:
        return float('nan'), float('nan'), float('nan')

    return stats.flip_signature_rho(pred_list, tgt_list, fk_list, rid_list, min_len=4, min_rows=2, min_cols=2)


def update_stats(stats_df, row_name, res_df, true_col, pred_col, epi_true_col='dddG_ML', epi_pred_col=None, time_val=None):
    """Helper to cleanly extract subset metrics for a specific predictive branch."""
    if pred_col not in res_df.columns:
        # Added explicit print to prevent silent failures
        print(f"Warning: Skipping stats update for '{row_name}'; column '{pred_col}' not found in predictions.")
        return stats_df
        
    stats_df.at[row_name, 'spearman_all'] = safe_spearman(res_df, true_col, pred_col)
    
    if 'mut_type' in res_df.columns:
        is_single = ~res_df['mut_type'].str.contains(':')
        is_double = res_df['mut_type'].str.contains(':')
        
        stats_df.at[row_name, 'spearman_singles'] = safe_spearman(res_df[is_single], true_col, pred_col)
        stats_df.at[row_name, 'n_singles'] = is_single.sum()
        
        stats_df.at[row_name, 'spearman_doubles'] = safe_spearman(res_df[is_double], true_col, pred_col)
        stats_df.at[row_name, 'n_doubles'] = is_double.sum()
    else:
        stats_df.at[row_name, 'spearman_singles'] = float('nan')
        stats_df.at[row_name, 'n_singles'] = 0
        stats_df.at[row_name, 'spearman_doubles'] = float('nan')
        stats_df.at[row_name, 'n_doubles'] = 0

    if epi_pred_col and epi_true_col in res_df.columns and epi_pred_col in res_df.columns:
        stats_df.at[row_name, 'spearman_doubles_epi'] = safe_spearman(res_df, epi_true_col, epi_pred_col)
    else:
        stats_df.at[row_name, 'spearman_doubles_epi'] = float('nan')

    # Compute flip-ordering rank correlation
    rho_flip, n_flip_pairs, n_flip_cells = compute_flip_stats(res_df, pred_col, true_col, row_name)
    stats_df.at[row_name, 'rho_flip'] = rho_flip
    stats_df.at[row_name, 'n_flip_pairs'] = n_flip_pairs

    stats_df.at[row_name, 'ndcg@96'] = safe_ndcg(res_df, pred_col, true_col, top_n=96)
    stats_df.at[row_name, 'ndcg>0'] = safe_ndcg(res_df, pred_col, true_col, threshold=0)
    
    if time_val is not None:
        stats_df.at[row_name, 'time'] = time_val
        
    return stats_df


def update_delta_stats(stats_df, row_name, res_df, epi_true_col):
    """Per-library MT-vs-WT single-mutant disagreement and its effect on the epistasis readouts
    (see stats.delta_single_diagnostics). Skipped quietly if the predictions lack the columns."""
    for col, val in stats.delta_single_diagnostics(res_df, epi_true_col).items():
        stats_df.at[row_name, col] = val
    return stats_df


def save_delta_stats(stats_df, stats_base):
    if len(stats_df):
        stats_df.to_csv(f'{stats_base}_DeltaSingles.csv', na_rep='', float_format='%.6f')
        stats_df.mean(axis=0).to_csv(f'{stats_base}_DeltaSingles_avg.csv', na_rep='', float_format='%.6f')
        print("MT-vs-WT single-mutant disagreement (mean over libraries):")
        print(stats_df.mean(axis=0).to_string(float_format=lambda x: f'{x:.4f}'))


def run_protein_gym(args, model):
    """Run esm-msr over all (preprocessed) ProteinGym DMS benchmarks.

    Reads per-DMS input CSVs + manifest.csv from --pgym_dir, scores each with
    infer_mutants (skip_additive default; dense by default, or masked via
    --mask_strategy independent/marginal), attaches the
    original DMS_score, writes one output CSV per DMS (resumable), and an
    incremental summary.csv with per-DMS Spearman(combined_pred, DMS_score).
    """
    if not args.pgym_dir:
        raise AssertionError("--pgym_dir is required with --protein_gym")
    pgym_dir = Path(args.pgym_dir).resolve()
    if not pgym_dir.exists():
        raise AssertionError(f"pgym_dir does not exist: {pgym_dir}")
    out_dir = Path(args.pgym_out).resolve() if args.pgym_out else (pgym_dir.parent / f"pgym_results_sigma{args.lora_epsilon}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # WT-only / additive-approximation mode (--skip_reverse): predict from the
    # WT adapter pass alone. combined_pred / mt_lora_pred are NaN by design in
    # that mode, so the benchmark scores wt_lora_pred instead of combined_pred.
    pred_col = "wt_lora_pred" if getattr(args, 'skip_reverse', False) else "combined_pred"

    manifest = pd.read_csv(pgym_dir / "manifest.csv")
    # pgym_preprocess.py writes per-DMS pdb_file paths relative to the
    # ProteinGym root; the manifest records that root for resolution.
    pg_root = (manifest["proteingym_dir"].iloc[0]
               if "proteingym_dir" in manifest.columns else None)
    if args.pgym_dms and str(args.pgym_dms).lower() != 'all':
        dms_list = [d.strip() for d in str(args.pgym_dms).split(',') if d.strip()]
    else:
        dms_list = sorted(manifest["DMS_id"].tolist())

    log_path = out_dir / "run.log"
    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + "\n")

    batch_map = None
    if args.batch_map:
        batch_map = json.load(open(args.batch_map))
    sizer = None
    if args.auto_batch_size:
        sizer = auto_batch.AutoBatchSizer(
            device=next(model.parameters()).device,
            headroom=args.auto_batch_headroom,
            max_batch=args.auto_batch_max,
            log=log,
        )
    log(f"[PGYM] START sigma={args.lora_epsilon} ckpt={args.checkpoint} n_dms={len(dms_list)} "
        f" pred_col={pred_col} batch={args.batch_size}" + (f" batch_map={args.batch_map} ({len(batch_map)} entries)" if batch_map else "") +
        (f" auto_batch(headroom={args.auto_batch_headroom}, max={args.auto_batch_max})" if sizer else "") +
        f" out={out_dir}")
    # Preserve rows from earlier invocations for DMS not in this run, so
    # summary.csv accumulates across per-bin invocations. Also keep a lookup
    # of the previous rows so a SKIP (output CSV already exists) can preserve
    # the earlier invocation's measured time_s instead of overwriting it with
    # NaN.
    summary_rows = []
    prev_by_dms = {}
    prev_summary = out_dir / "summary.csv"
    if prev_summary.exists():
        old = pd.read_csv(prev_summary)
        if "DMS_id" in old.columns:
            prev_by_dms = {row["DMS_id"]: row for _, row in old.iterrows()}
            old = old[~old["DMS_id"].isin(dms_list)]
            summary_rows = old.to_dict("records")
    t_start = time.time()
    for i, did in enumerate(dms_list, 1):
        out_csv = out_dir / f"{did}.csv"
        if out_csv.exists():
            log(f"[PGYM {i}/{len(dms_list)}] SKIP {did} (output exists)")
            prev_row = prev_by_dms.get(did)
            if prev_row is not None and pd.notna(prev_row.get("time_s")):
                # An earlier invocation measured this DMS's wall time; keep
                # that row verbatim. A skip must never overwrite a recorded
                # time_s with NaN.
                summary_rows.append(dict(prev_row))
            else:
                try:
                    prev = pd.read_csv(out_csv)
                    rho_prev = stats.safe_spearman(prev[pred_col], prev["DMS_score"])
                    summary_rows.append({"DMS_id": did, "n": len(prev), "spearman_combined": rho_prev,
                                         "time_s": None, "units_per_s": None, "status": "skipped"})
                except Exception:
                    summary_rows.append({"DMS_id": did, "n": None, "spearman_combined": None,
                                         "time_s": None, "units_per_s": None, "status": "skipped_bad"})
            pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
            continue
        dms_batch = (batch_map or {}).get(did, args.batch_size)
        n_reports_before = len(sizer.reports) if sizer else 0
        t0 = time.time()
        try:
            df = pd.read_csv(pgym_dir / f"{did}.csv")
            pdb_val = df["pdb_file"].iloc[0]
            if not os.path.isabs(pdb_val):
                # Relative path (portable pgym_inputs): resolve against the
                # ProteinGym root recorded in the manifest.
                if not pg_root:
                    raise AssertionError(
                        f"relative pdb_file '{pdb_val}' but manifest.csv has no "
                        "'proteingym_dir' column; regenerate pgym_inputs via "
                        "preprocessing/pgym_preprocess.py")
                pdb_res = (Path(pg_root) / pdb_val).resolve()
                if not pdb_res.exists():
                    raise AssertionError(f"pdb_file not found: {pdb_res}")
                df["pdb_file"] = pdb_res
            df = inference.standardize_input_df(df, quiet=True)
            log(f"[PGYM {i}/{len(dms_list)}] {did}: read_csv+standardize n={len(df)} in {time.time()-t0:.1f}s")
            res = inference.infer_mutants(
                model=model, df=df, batch_size=dms_batch,
                quiet=True, optimize_wt_pass=(args.mask_strategy is None),
                skip_reverse=args.skip_reverse, mask_strategy=args.mask_strategy,
                auto_batch=sizer
            )
            assert len(res) == len(df), f"len(res)={len(res)} != len(df)={len(df)}"
            assert (res["mut_type_renumbered"].values == df["mut_type_renumbered"].values).all(), "row misalignment"
            res = res.copy()
            res["DMS_id"] = did
            res["mutant"] = df["mutant"].values
            res["DMS_score"] = df["DMS_score"].values
            res.to_csv(out_csv, index=False)
            rho = stats.safe_spearman(res[pred_col], res["DMS_score"])
            dt = time.time() - t0
            summary_rows.append({"DMS_id": did, "n": len(res), "spearman_combined": rho,
                                 "time_s": dt, "units_per_s": len(res) / dt, "status": "done"})
            b_used = (max(r.max_batch for r in sizer.reports[n_reports_before:]) if sizer else dms_batch)
            log(f"[PGYM {i}/{len(dms_list)}] DONE {did}: n={len(res)} b={b_used} rho={rho:.4f} {dt:.1f}s ({len(res)/dt:.1f} u/s)")
        except Exception as e:
            dt = time.time() - t0
            import traceback
            log(f"[PGYM {i}/{len(dms_list)}] FAIL {did}: {type(e).__name__}: {str(e)[:300]} after {dt:.1f}s")
            log(traceback.format_exc())
            summary_rows.append({"DMS_id": did, "n": None, "spearman_combined": None,
                                 "time_s": dt, "units_per_s": None, "status": "fail", "error": str(e)[:200]})
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
        # Per-DMS L×vocab logits tensors and chunk buffers vary in size; return
        # the caching allocator's hold on freed blocks so VRAM stays flat across
        # the full DMS sweep (otherwise the footprint grows monotonically and the
        # largest DMS can push the card past its limit).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    s = pd.DataFrame(summary_rows)
    s.to_csv(out_dir / "summary.csv", index=False)
    valid = s.dropna(subset=["spearman_combined"])
    t_total = time.time() - t_start
    log(f"[PGYM] COMPLETE sigma={args.lora_epsilon}: {len(valid)}/{len(s)} DMS scored | "
        f"mean_rho={valid['spearman_combined'].mean():.4f} median_rho={valid['spearman_combined'].median():.4f} "
        f"n_pos={(valid['spearman_combined'] > 0).sum()} n_neg={(valid['spearman_combined'] < 0).sum()} "
        f"wall={t_total/3600:.2f}h")
    print("\n[PGYM] final summary:\n", s.to_string(index=False))


def main_(args):

    CHECKPOINT_STR = str(args.checkpoint) if args.checkpoint else "zeroshot"

    print('\n\n\n\n\n')
    print(f"Running Inference for Checkpoint: {CHECKPOINT_STR}")
    print('\n\n\n\n\n')

    os.makedirs('tmp', exist_ok=True)

    if CHECKPOINT_STR != 'zeroshot':
        ckpt_candidate = Path(args.checkpoint)
        if ckpt_candidate.is_file():
            ckpt_path = str(ckpt_candidate.resolve())
            hparams_path = ckpt_candidate.parent / 'hparams.yaml'
        elif (MODEL_DIR / args.checkpoint).is_file():
            ckpt_path = str((MODEL_DIR / args.checkpoint).resolve())
            hparams_path = (MODEL_DIR / args.checkpoint).parent / 'hparams.yaml'
        elif (REPO_ROOT / args.checkpoint).is_file():
            ckpt_path = str((REPO_ROOT / args.checkpoint).resolve())
            hparams_path = (REPO_ROOT / args.checkpoint).parent / 'hparams.yaml'
        else:
            ckpt_path = str(MODEL_DIR / args.checkpoint)
            hparams_path = Path(os.path.join(MODEL_DIR, os.path.dirname(args.checkpoint), 'hparams.yaml'))

        if not hparams_path.is_file():
            raise FileNotFoundError(f"Could not find hparams.yaml at {hparams_path}")

        parsed_config = inference.parse_hparams_to_lora_config(str(hparams_path))
        adapter_mode = parsed_config.get('adapter_mode', 'dual')
        lora_mode = parsed_config.get('lora_mode', 'ensemble')
        mask_structure = parsed_config.get('mask_structure', False)
        if args.mask_structure_pos or args.mask_coords_pos:
            if not mask_structure:
                print('[WARN] --mask_structure_pos set but this checkpoint was trained unmasked; '
                      'train/inference inputs will not match.')
            mask_structure = True
        if args.lora_epsilon != 1:
            parsed_config['wt_config']['lora_alpha'] *= args.lora_epsilon
            parsed_config['mt_config']['lora_alpha'] *= args.lora_epsilon

        lora_config = {
            'wt_config': parsed_config['wt_config'],
            'mt_config': parsed_config['mt_config']
        }
    
    else:
        mt_lora_config = {
            "lora_rank": 1, "lora_alpha": 0, "lora_dropout": 0,
            "target_mode": "baseline", "use_dora": False, "seed": args.seed,
            "last_n_layers": 1,
            "incl_sequence_head": False, "unfreeze_layernorms": False,
        }
        wt_lora_config = {
            "lora_rank": 1, "lora_alpha": 0, "lora_dropout": 0,
            "target_mode": "baseline", "use_dora": False, "seed": args.seed,
            "last_n_layers": 1,
            "incl_sequence_head": False, "unfreeze_layernorms": False,
        }
        adapter_mode = 'dual'
        lora_mode = 'ensemble'
        mask_structure = False
        lora_config = {'wt_config': wt_lora_config, 'mt_config': mt_lora_config, 'seed': args.seed}        

    model_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    print(f"[MODEL] inference dtype = {args.dtype} ({model_dtype})")
    shared_bias_init = parsed_config.get('shared_bias_init', None) if CHECKPOINT_STR != 'zeroshot' else 0
    shared_scale_init = parsed_config.get('shared_scale_init', 1.0) if CHECKPOINT_STR != 'zeroshot' else 1.0
    model = models.MSRModel(
        lora_config=lora_config, shared_scale_init=shared_scale_init, shared_bias_init=shared_bias_init, adapter_mode=adapter_mode,
        lora_mode=lora_mode, model_dtype=model_dtype, inference_mode=True, mask_structure=mask_structure
    ).to('cuda:0')

    # ---------------------------------------------------------
    # Robust Checkpoint Loading
    # ---------------------------------------------------------
    if args.checkpoint:
        model.load_lora_weights(ckpt_path)
    else:
        print('Zero shot mode!')
    
    model.eval()

    # =========================================================================
    # PROTEINGYM BENCHMARKS (all DMS, preprocessed inputs)
    # =========================================================================
    if getattr(args, 'protein_gym', False):
        run_protein_gym(args, model)
        return

    # =========================================================================
    # EXTERNAL BENCHMARKS
    # =========================================================================
    if not args.skip_external:
        external_test_dataloaders_names = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
        
        stats_wt = pd.DataFrame()
        stats_mt = pd.DataFrame()
        stats_cmb = pd.DataFrame()
        stats_delta = pd.DataFrame()

        for name in external_test_dataloaders_names:
            print(f"Processing External Dataset: {name}")

            res_combined = []
            total_time = 0

            df_true = pd.read_csv(DATA_DIR / f"{name}_mapped.csv")
            df_true['pdb_file'] = df_true['pdb_file'].str.replace('/home/sareeves/software/esm-msr/data/structures', args.local_path_to_structures, regex=False)
            if name in ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 's571', 's783', 's2648', 's8754']:
                df_true = df_true.reset_index()
                df_true['position_pdb'] = df_true['position']
                df_true['position'] = df_true['seq_pos']
                df_true['mut_type'] = df_true['wild_type'] + df_true['position'].astype(int).astype(str) + df_true['mutation']
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                if 'dTm' in df_true.columns:
                    df_true['ddG'] = df_true['dTm']
            else:
                df_true = df_true.reset_index()
                df_true = utils.sort_mutations_by_position(df_true, 'mut_info_seq_pos', 'mut_type')
                df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                df_true = utils.parse_multimutant_column(df_true, 'mut_type', max_mutations=10)

            for (pdb, code, chain), data in tqdm(df_true.groupby(['pdb_file', 'code', 'chain'])):
                
                unique_data = data[~data.index.duplicated(keep='first')]
                
                input_data = inference.standardize_input_df(unique_data, quiet=True)
                pred_df, t_inf = timed_call(
                    inference.infer_mutants, 
                    model=model, df=input_data, batch_size=1, quiet=True, mask_strategy=args.mask_strategy, 
                    optimize_wt_pass=(args.mask_strategy is None), skip_reverse=args.skip_reverse
                )
                pred_df['id'] = code + chain + '_' + pred_df['mut_type_renumbered']
                pred_df = pred_df.set_index('id')

                overlap_cols = list(set(data.columns).intersection(set(pred_df.columns)))
                res_partial = data.join(pred_df.drop(overlap_cols, axis=1))

                res_combined.append(res_partial)
                total_time += t_inf
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            res_df = pd.concat(res_combined)

            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res_df.to_csv(out_path)

            # Extract Metrics based on New Output Schema
            stats_wt = update_stats(stats_wt, name, res_df, 'ddG', 'wt_lora_pred', 'dddG', 'wt_lora_dddg_pred', total_time)
            
            if not args.skip_reverse:
                stats_mt = update_stats(stats_mt, name, res_df, 'ddG', 'mt_lora_pred', 'dddG', 'mt_lora_dddg_pred', total_time)
                stats_cmb = update_stats(stats_cmb, name, res_df, 'ddG', 'combined_pred', 'dddG', 'combined_dddg_pred', total_time)
                stats_delta = update_delta_stats(stats_delta, name, res_df, 'dddG')

            if 'ptmul' not in name:
                assert len(df_true) == len(res_df), f"Lost samples during join for {name}!"

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/external/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)
            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            stats_wt.mean(axis=0).to_csv(f'{stats_base}_WT_LoRA_avg.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')
                save_delta_stats(stats_delta, stats_base)
                stats_mt.mean(axis=0).to_csv(f'{stats_base}_MT_LoRA_avg.csv', na_rep='', float_format='%.6f')
                stats_cmb.mean(axis=0).to_csv(f'{stats_base}_Combined_avg.csv', na_rep='', float_format='%.6f')

    # =========================================================================
    # TSUBOYAMA SPLITS
    # =========================================================================
    if args.split is not None and not args.skip_tsuboyama:
        split_file = REPO_ROOT / "data" / f"{args.split}.pkl"
        split_name = args.split

        ds = preprocess_megascale.MegaScaleDatasetPreprocessor(
            data_file='/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 
            af_model_folder='/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs')
        splits = ds.create_training_splits(str(split_file), -1)

        if args.remove_spurs_homologs:
            ds.remove_homologs_from_scaffold(scaffold='train')
            ds.remove_homologs_from_scaffold(scaffold='val')

        for scaffold in ['validation', 'testing']:
            res_combined = []

            stats_wt = pd.DataFrame()
            stats_mt = pd.DataFrame()
            stats_cmb = pd.DataFrame()
            stats_delta = pd.DataFrame()

            scaffold_ = {'validation': 'val', 'testing': 'test'}[scaffold]
            data_scaffold = ds.split_dfs[scaffold_]
            
            data_scaffold = utils.parse_multimutant_column(data_scaffold, 'mut_type')
            data_scaffold['id'] = data_scaffold['code'] + '_' + data_scaffold['mut_type']
            data_scaffold = data_scaffold.sort_values('id')

            time_per_code = {} # Track time per protein properly

            for code in tqdm(data_scaffold['code_wt'].unique()):
                df_true = data_scaffold.loc[data_scaffold['code_wt']==code].copy()
                df_true['pdb_file'] = df_true['pdb_file'].str.replace('/home/sareeves/software/esm-msr/data/structures', args.local_path_to_structures, regex=False)
                assert len(df_true) > 0
                df_true['mut_structure'] = df_true['mut_structure'].fillna('-')

                t_total = 0

                for mut_structure, data in df_true.groupby('mut_structure'):
                    backbone_mutation = mut_structure if mut_structure != '-' else None

                    data = data.set_index('id')
                    data = utils.sum_individual_mutation_scores(data, 'ddG_ML', new_score_column='ddG_additive_ML')
                    data['dddG_ML'] = data['ddG_ML'] - data['ddG_additive_ML']

                    unique_data = data[~data.index.duplicated(keep='first')]
                    input_data = inference.standardize_input_df(unique_data, backbone_mutation=backbone_mutation, quiet=True)

                    # Unified Inference
                    pred_df, t = timed_call(
                        inference.infer_mutants, 
                        model=model, df=input_data, batch_size=16, backbone_mutation=backbone_mutation, quiet=True, 
                        skip_additive=False, mask_strategy=args.mask_strategy, optimize_wt_pass=(args.mask_strategy is None),
                        skip_reverse=args.skip_reverse
                    )
                    pred_df['id'] = code + ('_' if backbone_mutation is None else '_' + str(backbone_mutation) + '_') + pred_df['mut_type_renumbered']
                    pred_df = pred_df.set_index('id')

                    overlap_cols = list(set(data.columns).intersection(set(pred_df.columns)))
                    res_partial = data.join(pred_df.drop(overlap_cols, axis=1))
                    res_combined.append(res_partial)
                    t_total += t
                
                time_per_code[code] = t_total # Store the accumulated time for this specific code
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Aggregate DataFrames
            res_df = pd.concat(res_combined)

            # File Operations
            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res_df.to_csv(out_path)

            # Metrics
            for code, group in res_df.groupby('code_wt'):
                current_time = time_per_code.get(code, float('nan'))
                stats_wt = update_stats(stats_wt, code, group, 'ddG_ML', 'wt_lora_pred', 'dddG_ML', 'wt_lora_dddg_pred', current_time)
                if not args.skip_reverse:
                    stats_mt = update_stats(stats_mt, code, group, 'ddG_ML', 'mt_lora_pred', 'dddG_ML', 'mt_lora_dddg_pred', current_time)
                    stats_cmb = update_stats(stats_cmb, code, group, 'ddG_ML', 'combined_pred', 'dddG_ML', 'combined_dddg_pred', current_time)
                    stats_delta = update_delta_stats(stats_delta, code, group, 'dddG_ML')

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)
            
            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            stats_wt.mean(axis=0).to_csv(f'{stats_base}_WT_LoRA_avg.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')
                save_delta_stats(stats_delta, stats_base)
                stats_mt.mean(axis=0).to_csv(f'{stats_base}_MT_LoRA_avg.csv', na_rep='', float_format='%.6f')
                stats_cmb.mean(axis=0).to_csv(f'{stats_base}_Combined_avg.csv', na_rep='', float_format='%.6f')

            torch.cuda.empty_cache()

    # =========================================================================
    # DMS DATASETS
    # =========================================================================
    if not args.skip_dms:
        prots = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain']
        mem_sizes = [4, 4, 8, 8, 1, 1, 8]
        
        assert len(prots) == len(mem_sizes), f"Length mismatch: {len(prots)} proteins vs {len(mem_sizes)} memory sizes."

        stats_wt = pd.DataFrame()
        stats_mt = pd.DataFrame()
        stats_cmb = pd.DataFrame()
        stats_delta = pd.DataFrame()
        
        res_combined = []

        for mem_size, prot in zip(mem_sizes, prots): 
            batch_sz = mem_size * 32

            df_true = pd.read_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/data/preprocessed/{prot}.csv')
            df_true['pdb_file'] = df_true['pdb_file'].str.replace('/home/sareeves/software/esm-msr/data/structures', args.local_path_to_structures, regex=False)
            df_true['id'] = df_true['code'] + '_' + df_true['mut_info']
            df_true = df_true.set_index('id')
            
            has_doubles = len(df_true.loc[df_true['mut_info'].str.contains(':')]) > 0
            if has_doubles:
                df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']

            prot_name = '_'.join(prot.split('_')[:2])
            if prot_name == 'GB1_Wu':
                prot_name = 'GB1'

            unique_data = df_true[~df_true.index.duplicated(keep='first')]
            input_data = inference.standardize_input_df(unique_data, quiet=True)

            pred_df, t_inf = timed_call(
                inference.infer_mutants, 
                model=model, df=input_data, batch_size=batch_sz, quiet=False, skip_additive=False, 
                mask_strategy=args.mask_strategy, optimize_wt_pass=(args.mask_strategy is None),
                skip_reverse=args.skip_reverse
            )
            pred_df['id'] = prot_name + '_' + pred_df['mut_type_renumbered']
            pred_df = pred_df.set_index('id')

            overlap_cols = list(set(df_true.columns).intersection(set(pred_df.columns)))
            res = df_true.join(pred_df.drop(overlap_cols, axis=1))

            assert len(df_true) == len(res), f"Merge error on DMS {prot}"
            res_combined.append(res)

            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res.to_csv(out_path)

            stats_wt = update_stats(stats_wt, prot, res, 'ddG_ML', 'wt_lora_pred', 'dddG_ML', 'wt_lora_dddg_pred', t_inf)
            if not args.skip_reverse:
                stats_mt = update_stats(stats_mt, prot, res, 'ddG_ML', 'mt_lora_pred', 'dddG_ML', 'mt_lora_dddg_pred', t_inf)
                stats_cmb = update_stats(stats_cmb, prot, res, 'ddG_ML', 'combined_pred', 'dddG_ML', 'combined_dddg_pred', t_inf)
                stats_delta = update_delta_stats(stats_delta, prot, res, 'dddG_ML')

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/DMS/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)

            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            stats_wt.mean(axis=0).to_csv(f'{stats_base}_WT_LoRA_avg.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')
                save_delta_stats(stats_delta, stats_base)
                stats_mt.mean(axis=0).to_csv(f'{stats_base}_MT_LoRA_avg.csv', na_rep='', float_format='%.6f')
                stats_cmb.mean(axis=0).to_csv(f'{stats_base}_Combined_avg.csv', na_rep='', float_format='%.6f')

            torch.cuda.empty_cache()

    # =========================================================================
    # DOMAINOME DATASET
    # =========================================================================
    
    if not args.skip_domainome:
        path = f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/data/domainome1/domainome_mapped_2026.csv'
        df = pd.read_csv(path)
        df['pdb_file'] = df['pdb_file'].str.replace('/home/sareeves/software/esm-msr/data/structures', args.local_path_to_structures, regex=False)
        df['code'] = df['domain_ID'].apply(lambda x: x.replace('/', '_'))
        df['ddG_ML'] = df['scaled_fitness']
        df = df.dropna(subset=['pdb_file', 'position'])
        df = df[['code', 'mut_type', 'uniprot_ID', 'pdb_file', 'ddG_ML']]
        
        stats_wt = pd.DataFrame()
        stats_mt = pd.DataFrame()
        stats_cmb = pd.DataFrame()

        res_combined = []
        
        should_skip_reverse_dom = args.skip_reverse or args.skip_reverse_domainome

        for prot in tqdm(df['code'].unique()):
            df_true = df.loc[df['code']==prot].copy()
            df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
            df_true['chain'] = 'A'
            df_true = df_true.set_index('id')

            unique_data = df_true[~df_true.index.duplicated(keep='first')]
            input_data = inference.standardize_input_df(unique_data, quiet=True)

            pred_df, t_inf = timed_call(
                inference.infer_mutants, 
                model=model, df=input_data, batch_size=32, quiet=True, 
                skip_reverse=should_skip_reverse_dom, mask_strategy=args.mask_strategy, 
                optimize_wt_pass=(args.mask_strategy is None)
            )
            pred_df['id'] = prot + '_' + pred_df['mut_type_renumbered']
            pred_df = pred_df.set_index('id')

            overlap_cols = list(set(df_true.columns).intersection(set(pred_df.columns)))
            res = df_true.join(pred_df.drop(overlap_cols, axis=1))

            assert len(df_true) == len(res)
            res_combined.append(res)

            stats_wt = update_stats(stats_wt, prot, res, 'ddG_ML', 'wt_lora_pred', time_val=t_inf)
            if not should_skip_reverse_dom:
                stats_mt = update_stats(stats_mt, prot, res, 'ddG_ML', 'mt_lora_pred', time_val=t_inf)
                stats_cmb = update_stats(stats_cmb, prot, res, 'ddG_ML', 'combined_pred', time_val=t_inf)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        res_df = pd.concat(res_combined, axis=0)

        out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if should_skip_reverse_dom else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}_predictions.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        res_df.to_csv(out_path)

        stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if should_skip_reverse_dom else ""}_{args.mask_strategy if args.mask_strategy is not None else "unmasked"}')
        os.makedirs(os.path.dirname(stats_base), exist_ok=True)

        stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
        stats_wt.mean(axis=0).to_csv(f'{stats_base}_WT_LoRA_avg.csv', na_rep='', float_format='%.6f')

        if not should_skip_reverse_dom:
            stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
            stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')
            
            stats_mt.mean(axis=0).to_csv(f'{stats_base}_MT_LoRA_avg.csv', na_rep='', float_format='%.6f')
            stats_cmb.mean(axis=0).to_csv(f'{stats_base}_Combined_avg.csv', na_rep='', float_format='%.6f')

        torch.cuda.empty_cache()


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, required=False)
        parser.add_argument('--split', type=str)
        parser.add_argument('--seed', type=int, required=False, default=42)
        parser.add_argument('--lora_epsilon', type=float, required=False, default=1)
        
        # Precision Argument
        parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['16', '16-mixed', '32', 'bf16-mixed'])

        parser.add_argument('--local_cluster', action='store_true')
        parser.add_argument('--mask_strategy', type=str, choices=['marginal', 'independent'], default=None)
        # Legacy names. These used to be inert: they printed a message and nothing masked
        # anything (utils.apply_masks is never called). They now force the model's
        # mask_structure on, overriding whatever the checkpoint was trained with.
        parser.add_argument('--mask_structure_pos', action='store_true',
                            help="Force structure masking on, overriding the checkpoint's hparams.")
        parser.add_argument('--mask_coords_pos', action='store_true',
                            help="Alias of --mask_structure_pos (coordinates and structure tokens are "
                                 "always masked together; masking one leaves the other informative).")
        parser.add_argument('--mask_coords', action='store_true')
        parser.add_argument('--regenerate_results', action='store_true')
        parser.add_argument('--skip_external', action='store_true')
        parser.add_argument('--skip_tsuboyama', action='store_true')
        parser.add_argument('--skip_ctx', action='store_true')
        parser.add_argument('--skip_dms', action='store_true')
        parser.add_argument('--skip_functional', action='store_true')
        parser.add_argument('--skip_domainome', action='store_true')
        parser.add_argument('--skip_additive', action='store_true')
        parser.add_argument('--skip_reverse', action='store_true',
                            help='Skip the MT adapter pass and predict purely from the WT adapters. With --protein_gym, benchmarks score wt_lora_pred (the WT additive-approximation prediction) instead of combined_pred.')
        parser.add_argument('--skip_reverse_domainome', action='store_true')
        parser.add_argument('--use_dora', action='store_true')
        
        parser.add_argument('--local_path_to_structures', type=str, default='/home/sareeves/software/esm-msr/data/structures')
        parser.add_argument('--hf_token', type=str, default=None)
        parser.add_argument('--remove_spurs_homologs', action='store_true', help='Remove homologous sequences to SPURS training data from the Tsuboyama splits to test generalization to non-homologous sequences')

        # ProteinGym (all-DMS) benchmark mode
        parser.add_argument('--protein_gym', action='store_true', help='Run esm-msr over all preprocessed ProteinGym DMS benchmarks and exit')
        parser.add_argument('--pgym_dir', type=str, default=None, help='Directory with per-DMS input CSVs + manifest.csv')
        parser.add_argument('--pgym_out', type=str, default=None, help='Output directory for ProteinGym results (per-DMS CSVs + summary.csv)')
        parser.add_argument('--pgym_dms', type=str, default='all', help="Comma-separated DMS ids, or 'all'")
        parser.add_argument('--batch_size', type=int, default=16, help='Inference batch size for ProteinGym scoring')
        parser.add_argument('--batch_map', type=str, default=None,
                            help='Optional JSON file {DMS_id: batch_size} for per-DMS batch sizing (e.g. max-fitting per length bin). DMS not in the map use --batch_size.')
        parser.add_argument('--auto_batch_size', action='store_true',
                            help='On-the-fly per-DMS batch sizing: start at 1, double while a measured memory model predicts the attempt fits (esm_msr/auto_batch.py). Overrides --batch_size/--batch_map.')
        parser.add_argument('--auto_batch_headroom', type=float, default=0.90,
                            help='VRAM fraction a predicted chunk peak may occupy when auto-batching (default 0.90)')
        parser.add_argument('--auto_batch_max', type=int, default=None,
                            help='Optional hard cap on the auto-batched batch size')
        parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp32'], help='Model inference dtype (bf16 = native/autocast, lower VRAM)')

        args, remaining_argv = parser.parse_known_args()
        current_remaining_argv = list(remaining_argv) 

        if current_remaining_argv:
            parser.error(f"unrecognized arguments: {' '.join(current_remaining_argv)}")

        if args.skip_external:
            print('Skipping benchmark datasets!')
        if args.skip_tsuboyama:
            print('Skipping MegaScale validation and testing datasets!')
        if args.skip_functional:
            print('Skipping double mutant DMS assays!')
        if args.skip_domainome:
            print('Skipping domainome VAMP assays!')
        if args.skip_reverse:
            print('Skipping all reverse mutational passes!')
        if args.mask_structure_pos or args.mask_coords_pos:
            print('Forcing structure masking ON for the MT pass (overrides the checkpoint hparams).')
        if not args.split:
            print('Warning! Not using any specific split file!')
        if args.split and 'mega' in args.split and not args.remove_spurs_homologs:
            print('Warning: not removing SPURS homologs')

        token = args.hf_token or get_token()

        if token:
            os.environ["HF_TOKEN"] = token  # auth for hub loads even if login() can't reach the API (offline/cache-only)
            try:
                login(token)
                print('Using token (login ok)')
            except Exception as _e:
                print(f'Using token (login skipped: {type(_e).__name__}); HF_TOKEN exported, using cache/offline')
        else:
            os.environ['INFRA_PROVIDER'] = "1"
            os.chdir(Path(__file__).resolve().parent.parent)
            print(f'Using local model which should be located at {os.path.join(os.getcwd(), "data/weights/esm3_sm_open_v1.pth")}')

        main_(args)