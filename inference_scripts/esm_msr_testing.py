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

# this repository's src/, ahead of any other installed copy of esm_msr
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from huggingface_hub import login, get_token

from esm_msr import stats, utils, models, inference, preprocess_megascale, auto_batch

import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "preprocessed"
MODEL_DIR = REPO_ROOT / "LoRA_models"

def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _mask_strategy_tag(ms) -> str:
    if ms is None:
        return "unmasked"
    if isinstance(ms, dict):
        w = ms.get('wt') or 'unmasked'
        m = ms.get('mt') or 'unmasked'
        return str(w) if w == m else f"wt-{w}_mt-{m}"
    return str(ms)

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

    stats_df.at[row_name, 'ndcg@96'] = safe_ndcg(res_df, pred_col, true_col, top_n=96)
    stats_df.at[row_name, 'ndcg>0'] = safe_ndcg(res_df, pred_col, true_col, threshold=0)
    
    if time_val is not None:
        stats_df.at[row_name, 'time'] = time_val
        
    return stats_df


def run_protein_gym(args, model):
    """Run esm-msr over all (preprocessed) ProteinGym DMS benchmarks, through the loop every model shares
    (pgym_runner.run_protein_gym: per-DMS resumable output CSVs, incremental summary.csv, run.log).

    Each DMS is scored with infer_mutants (skip_additive default; dense by default, or masked via --mask_strategy
    independent/marginal) and correlated with DMS_score on combined_pred, or on wt_lora_pred with --skip_reverse.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pgym_runner
    if not args.pgym_dir:
        raise AssertionError("--pgym_dir is required with --protein_gym")
    pgym_dir = Path(args.pgym_dir).resolve()
    out_dir = Path(args.pgym_out).resolve() if args.pgym_out else (pgym_dir.parent / f"pgym_results_sigma{args.lora_epsilon}")
    # WT-only / additive-approximation mode (--skip_reverse): predict from the
    # WT adapter pass alone. combined_pred / mt_lora_pred are NaN by design in
    # that mode, so the benchmark scores wt_lora_pred instead of combined_pred.
    pred_col = "wt_lora_pred" if getattr(args, 'skip_reverse', False) else "combined_pred"
    log = pgym_runner.make_log(out_dir)

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

    def score_fn(df):
        did = df["code"].iloc[0]   # pgym_preprocess.py sets code = DMS_id
        dms_batch = (batch_map or {}).get(did, args.batch_size)
        n_reports_before = len(sizer.reports) if sizer else 0
        t0 = time.time()
        try:
            df = inference.standardize_input_df(df, quiet=True)
            log(f"[PGYM] {did}: standardize n={len(df)} in {time.time()-t0:.1f}s")
            res = inference.infer_mutants(
                model=model, df=df, batch_size=dms_batch,
                quiet=True, optimize_wt_pass=True,
                skip_reverse=args.skip_reverse, mask_strategy=args.mask_strategy,
                auto_batch=sizer
            )
        finally:
            # Per-DMS L x vocab logits tensors and chunk buffers vary in size; return the caching allocator's hold on
            # freed blocks so VRAM stays flat across the full DMS sweep (otherwise the footprint grows monotonically
            # and the largest DMS can push the card past its limit).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        assert len(res) == len(df), f"len(res)={len(res)} != len(df)={len(df)}"
        assert (res["mut_type_renumbered"].values == df["mut_type_renumbered"].values).all(), "row misalignment"
        b_used = (max(r.max_batch for r in sizer.reports[n_reports_before:]) if sizer else dms_batch)
        return res.reset_index(drop=True), {"b": b_used}

    tag = (f"sigma={args.lora_epsilon} ckpt={args.checkpoint} mt_pass_adapter={args.mt_pass_adapter} batch={args.batch_size}" +
           (f" batch_map={args.batch_map} ({len(batch_map)} entries)" if batch_map else "") +
           (f" auto_batch(headroom={args.auto_batch_headroom}, max={args.auto_batch_max})" if sizer else ""))
    pgym_runner.run_protein_gym(pgym_dir, out_dir, score_fn, pred_col=pred_col, dms=args.pgym_dms, tag=tag, log=log)

def main_(args):

    CHECKPOINT_STR = str(args.checkpoint) if args.checkpoint else "zeroshot"

    print('\n\n\n\n\n')
    print(f"Running Inference for Checkpoint: {CHECKPOINT_STR}")
    print('\n\n\n\n\n')

    os.makedirs('tmp', exist_ok=True)

    if CHECKPOINT_STR != 'zeroshot':
        resolved_ckpt = os.path.abspath(args.checkpoint) if os.path.isfile(args.checkpoint) else str(MODEL_DIR / args.checkpoint)
        hparams_path = os.path.join(os.path.dirname(resolved_ckpt), 'hparams.yaml')
        parsed_config = inference.parse_hparams_to_lora_config(hparams_path)
        adapter_mode = parsed_config.get('adapter_mode', 'dual')
        lora_mode = parsed_config.get('lora_mode', 'ensemble')
        checkpoint_cal = parsed_config.get('calibration')
        if getattr(args, 'mask_strategy_wt', None) is not None or getattr(args, 'mask_strategy_mt', None) is not None:
            m_wt = None if getattr(args, 'mask_strategy_wt', None) in (None, 'none') else args.mask_strategy_wt
            m_mt = None if getattr(args, 'mask_strategy_mt', None) in (None, 'none') else args.mask_strategy_mt
            args.mask_strategy = m_wt if m_wt == m_mt else {'wt': m_wt, 'mt': m_mt}
        elif args.mask_strategy is None and parsed_config.get('mask_strategy') is not None:
            args.mask_strategy = parsed_config['mask_strategy']
            print(f"[MODEL] mask_strategy from hparams: {args.mask_strategy}")
        native_cond_for_mt = bool(parsed_config.get('native_cond_for_mt', False))
        mask_native_cond_struct = bool(parsed_config.get('mask_native_cond_struct', False))
        fused_antisymmetric = bool(parsed_config.get('fused_antisymmetric', False))
        if args.lora_epsilon != 1:
            parsed_config['wt_config']['lora_alpha'] *= args.lora_epsilon
            parsed_config['mt_config']['lora_alpha'] *= args.lora_epsilon

        lora_config = {
            'wt_config': parsed_config['wt_config'],
            'mt_config': parsed_config['mt_config']
        }
    
    else:
        resolved_ckpt = None
        mt_lora_config = {
            "lora_rank": 1, "lora_alpha": 0, "lora_dropout": 0,
            "target_mode": "baseline", "use_dora": False, "seed": args.seed,
            "incl_structure_encoder": False, "last_n_layers": 1,
            "incl_sequence_head": False, "unfreeze_layernorms": False,
        }
        wt_lora_config = {
            "lora_rank": 1, "lora_alpha": 0, "lora_dropout": 0,
            "target_mode": "baseline", "use_dora": False, "seed": args.seed,
            "incl_structure_encoder": False, "last_n_layers": 1,
            "incl_sequence_head": False, "unfreeze_layernorms": False,
        }
        adapter_mode = 'dual'
        lora_mode = 'ensemble'
        checkpoint_cal = None
        native_cond_for_mt = False
        mask_native_cond_struct = False
        fused_antisymmetric = False
        lora_config = {'wt_config': wt_lora_config, 'mt_config': mt_lora_config, 'seed': args.seed}        

    model_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    print(f"[MODEL] inference dtype = {args.dtype} ({model_dtype})")
    cal = inference.resolve_calibration(checkpoint_cal, args.tie_mt_scale, args.per_mutation_bias)
    print(f"[MODEL] calibration: tie_mt_scale={cal['tie_mt_scale']} per_mutation_bias={cal['per_mutation_bias']}")
    model = models.MSRModel(
        lora_config=lora_config, shared_scale_init=1, shared_bias_init=0, adapter_mode=adapter_mode,
        lora_mode=lora_mode, model_dtype=model_dtype, inference_mode=True,
        tie_mt_scale=cal['tie_mt_scale'], per_mutation_bias=cal['per_mutation_bias'],
        native_cond_mt_input=native_cond_for_mt,
        mask_native_cond_struct=mask_native_cond_struct,
        fused_antisymmetric=fused_antisymmetric,
        epistasis_weight=args.epistasis_weight,
    ).to('cuda:0')

    # ---------------------------------------------------------
    # Robust Checkpoint Loading
    # ---------------------------------------------------------
    if args.checkpoint:
        model.load_lora_weights(resolved_ckpt)
    else:
        print('Zero shot mode!')

    if args.mt_pass_adapter == 'wt':
        # Straight swap: the MT pass (mutated sequence in, same LLR readout) runs the WT adapter and its calibration
        # head instead of the MT ones; the MT adapter is left unused. Everything else, combined_pred included, is unchanged.
        if adapter_mode != 'dual':
            raise AssertionError(f"--mt_pass_adapter wt needs a dual-adapter model, got adapter_mode={adapter_mode}")
        model.peft_mt = model.peft_wt
        model.calibration_head_mt = model.calibration_head_wt
        print('[MODEL] MT pass uses the WT adapter and WT calibration head (--mt_pass_adapter wt)')

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
                    optimize_wt_pass=True, skip_reverse=args.skip_reverse
                )
                pred_df['id'] = code + chain + '_' + pred_df['mut_type_renumbered']
                pred_df = pred_df.set_index('id')

                overlap_cols = list(set(data.columns).intersection(set(pred_df.columns)))
                res_partial = data.join(pred_df.drop(overlap_cols, axis=1))

                res_combined.append(res_partial)
                total_time += t_inf

            res_df = pd.concat(res_combined)
            mask_tag = _mask_strategy_tag(args.mask_strategy)

            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{mask_tag}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res_df.to_csv(out_path)

            # Extract Metrics based on New Output Schema
            stats_wt = update_stats(stats_wt, name, res_df, 'ddG', 'wt_lora_pred', 'dddG', 'wt_lora_dddg_pred', total_time)
            
            if not args.skip_reverse:
                stats_mt = update_stats(stats_mt, name, res_df, 'ddG', 'mt_lora_pred', 'dddG', 'mt_lora_dddg_pred', total_time)
                stats_cmb = update_stats(stats_cmb, name, res_df, 'ddG', 'combined_pred', 'dddG', 'combined_dddg_pred', total_time)

            if 'ptmul' not in name:
                assert len(df_true) == len(res_df), f"Lost samples during join for {name}!"

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/external/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{mask_tag}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)
            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')

    # =========================================================================
    # TSUBOYAMA SPLITS
    # =========================================================================
    if args.split is not None and not args.skip_tsuboyama:
        split_file = REPO_ROOT / "data" / f"{args.split}.pkl"
        split_name = args.split

        # The raw table and the AF models ship with the repo, so resolve them against REPO_ROOT as `split_file`
        # above does. They were hardcoded under /home/sareeves, which does not exist on this cluster, so the whole
        # Tsuboyama block could not run here; --local_cluster already switches the domainome path the same way.
        tsu = (REPO_ROOT / 'data' / 'tsuboyama') if args.local_cluster else Path(
            '/home/sareeves/software/esm-msr/data/tsuboyama')
        # censor_range stays at devel-revision's default (True), so the rows the assay only BOUNDED ('<-1' / '>5')
        # are predicted too and land in the predictions CSV, flagged in `cens`. They are excluded from the REPORTED
        # statistics below, because their ddG_ML is that bound rather than a measurement: on the val half they are
        # 516 rows (475 '<-1', 41 '>5') carrying only 26 distinct ddG_ML values, so scoring against them measures
        # agreement with a library constant. Predicting them keeps the option open; reporting them does not.
        ds = preprocess_megascale.MegaScaleDatasetPreprocessor(
            data_file=str(tsu / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'),
            af_model_folder=str(tsu / 'AlphaFold_model_PDBs'))
        splits = ds.create_training_splits(str(split_file), -1)

        if args.remove_spurs_homologs:
            ds.remove_homologs_from_scaffold(scaffold='train')
            ds.remove_homologs_from_scaffold(scaffold='val')

        for scaffold in ['validation', 'testing']:
            res_combined = []

            stats_wt = pd.DataFrame()
            stats_mt = pd.DataFrame()
            stats_cmb = pd.DataFrame()

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
                        skip_additive=False, mask_strategy=args.mask_strategy, optimize_wt_pass=True,
                        skip_reverse=args.skip_reverse
                    )
                    pred_df['id'] = code + ('_' if backbone_mutation is None else '_' + str(backbone_mutation) + '_') + pred_df['mut_type_renumbered']
                    pred_df = pred_df.set_index('id')

                    overlap_cols = list(set(data.columns).intersection(set(pred_df.columns)))
                    res_partial = data.join(pred_df.drop(overlap_cols, axis=1))
                    res_combined.append(res_partial)
                    t_total += t
                
                time_per_code[code] = t_total # Store the accumulated time for this specific code

            # Aggregate DataFrames
            res_df = pd.concat(res_combined)
            mask_tag = _mask_strategy_tag(args.mask_strategy)

            # File Operations
            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{mask_tag}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res_df.to_csv(out_path)

            # Metrics -- on MEASURED rows only. A bounded row's ddG_ML is its library's bound ('<-1' / '>5'), not a
            # measurement, so scoring against it measures agreement with a library constant rather than with the
            # assay. The predictions CSV written above keeps every row, flagged in `cens`, so the bounded ones stay
            # available; they are only kept out of what gets reported.
            scored = res_df[res_df['cens'] == 0] if 'cens' in res_df.columns else res_df
            if len(scored) != len(res_df):
                logging.info(f"reported stats exclude {len(res_df) - len(scored)} bounded rows "
                             f"({int((res_df['cens'] < 0).sum())} '<-1', {int((res_df['cens'] > 0).sum())} '>5'); "
                             f"the predictions CSV keeps all {len(res_df)}")
            for code, group in scored.groupby('code_wt'):
                current_time = time_per_code.get(code, float('nan'))
                stats_wt = update_stats(stats_wt, code, group, 'ddG_ML', 'wt_lora_pred', 'dddG_ML', 'wt_lora_dddg_pred', current_time)
                if not args.skip_reverse:
                    stats_mt = update_stats(stats_mt, code, group, 'ddG_ML', 'mt_lora_pred', 'dddG_ML', 'mt_lora_dddg_pred', current_time)
                    stats_cmb = update_stats(stats_cmb, code, group, 'ddG_ML', 'combined_pred', 'dddG_ML', 'combined_dddg_pred', current_time)

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/{split_name}-{scaffold_}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{mask_tag}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)
            
            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            stats_wt.mean(axis=0).to_csv(f'{stats_base}_WT_LoRA_avg.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')
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
        
        res_combined = []

        for mem_size, prot in zip(mem_sizes, prots): 
            batch_sz = mem_size * 32

            # the repo's own data/preprocessed, not the software/esm-msr copy: that copy's GB1 is missing V39A/V39C/
            # V39D (73 of 76 singles of a 4-site saturation library), which costs 152 of its 2,091 scorable doubles,
            # since a double needs both singles measured. The shared rows are bit-identical.
            df_true = pd.read_csv(REPO_ROOT / 'data' / 'preprocessed' / f'{prot}.csv', low_memory=False)
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
                mask_strategy=args.mask_strategy, optimize_wt_pass=True,
                skip_reverse=args.skip_reverse
            )
            pred_df['id'] = prot_name + '_' + pred_df['mut_type_renumbered']
            pred_df = pred_df.set_index('id')

            overlap_cols = list(set(df_true.columns).intersection(set(pred_df.columns)))
            res = df_true.join(pred_df.drop(overlap_cols, axis=1))

            assert len(df_true) == len(res), f"Merge error on DMS {prot}"
            res_combined.append(res)
            mask_tag = _mask_strategy_tag(args.mask_strategy)

            out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{mask_tag}_predictions.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            res.to_csv(out_path)

            stats_wt = update_stats(stats_wt, prot, res, 'ddG_ML', 'wt_lora_pred', 'dddG_ML', 'wt_lora_dddg_pred', t_inf)
            if not args.skip_reverse:
                stats_mt = update_stats(stats_mt, prot, res, 'ddG_ML', 'mt_lora_pred', 'dddG_ML', 'mt_lora_dddg_pred', t_inf)
                stats_cmb = update_stats(stats_cmb, prot, res, 'ddG_ML', 'combined_pred', 'dddG_ML', 'combined_dddg_pred', t_inf)

            stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/DMS/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if args.skip_reverse else ""}_{mask_tag}')
            os.makedirs(os.path.dirname(stats_base), exist_ok=True)

            stats_wt.to_csv(f'{stats_base}_WT_LoRA.csv', na_rep='', float_format='%.6f')
            
            if not args.skip_reverse:
                stats_mt.to_csv(f'{stats_base}_MT_LoRA.csv', na_rep='', float_format='%.6f')
                stats_cmb.to_csv(f'{stats_base}_Combined.csv', na_rep='', float_format='%.6f')

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
                optimize_wt_pass=True
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

        res_df = pd.concat(res_combined, axis=0)
        mask_tag = _mask_strategy_tag(args.mask_strategy)

        out_path = str(REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if should_skip_reverse_dom else ""}_{mask_tag}_predictions.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        res_df.to_csv(out_path)

        stats_base = str(REPO_ROOT / 'analysis_notebooks' / f'stats/domainome/{CHECKPOINT_STR}_epsilon{args.lora_epsilon}{"_skip_additive" if args.skip_additive else ""}{"_skip_reverse" if should_skip_reverse_dom else ""}_{mask_tag}')
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
        parser.add_argument('--mask_strategy_wt', type=str, choices=['none', 'marginal', 'independent'], default=None)
        parser.add_argument('--mask_strategy_mt', type=str, choices=['none', 'marginal', 'independent'], default=None)
        parser.add_argument('--mask_structure_pos', action='store_true')
        parser.add_argument('--mask_coords_pos', action='store_true')
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
        parser.add_argument('--mt_pass_adapter', type=str, choices=['mt', 'wt'], default='mt',
                            help='Adapter (with its calibration head) that runs the MT pass on the mutated sequence. wt: the WT adapter sees the mutant context; the MT adapter is unused.')
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
        parser.add_argument('--tie_mt_scale', action=argparse.BooleanOptionalAction, default=None,
                            help="Share the WT calibration scale with the MT head. Default: the checkpoint's hparams "
                                 "(OFF when the key is missing, as for the released models).")
        parser.add_argument('--per_mutation_bias', action=argparse.BooleanOptionalAction, default=None,
                            help="Apply the calibration bias once per mutated position. Default: the checkpoint's "
                                 "hparams (ON when the key is missing: the released and dc8 models were trained with "
                                 "one bias per item, so this shifts their multi-mutants by (k - 1) * bias).")
        parser.add_argument('--epistasis_weight', type=float, default=0.5,
                            help="Blend weight lambda in (1 - lambda) * y_WT + lambda * y_MT (default 0.5).")

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
            print('Masking one or more inputs!')
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