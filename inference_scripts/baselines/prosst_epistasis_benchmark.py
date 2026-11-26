"""Benchmark ProSST-2048/4096 under unmasked vs independent-masking dual-view scoring for epistasis.

Evaluates all 4 modes from `prosst_scoring.py` (`wt_unmasked`, `dual_unmasked`, `wt_independent`, `dual_independent`)
without modifying upstream ProSST code, on:
  - MegaScale `tsuboyama_test` (singles, doubles, and exact double-mutant cycle epistasis dddG)
  - All 3 DMS double-mutant assays (DLG4, GRB2, MYO: singles, doubles, and exact dddG)
  - External multi-mutant sets (`ptmul`, `ptmuld`)

Writes per-dataset prediction CSVs and `epistasis_summary.csv` to `--out`.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get('MSR_REPO') or HERE.parents[1]).resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / 'inference_scripts' / 'baselines'))
sys.path.insert(0, str(REPO_ROOT / 'src'))

import paths  # noqa: E402
from benchmark_rows import DMS, dms_rows, external_rows, library_sequence  # noqa: E402
from prosst_scoring import VALID_MODES, score_protein_variants  # noqa: E402


def _extract_chain_pdb(pdb_path: str, chain: str, tmp_dir: Path) -> str:
    lines = []
    chains_seen = set()
    with open(pdb_path) as f:
        for ln in f:
            if ln.startswith('ATOM  '):
                ch = ln[21]
                chains_seen.add(ch)
                if ch == chain:
                    lines.append(ln)
    if len(chains_seen) == 1 and chain in chains_seen:
        return pdb_path
    out = tmp_dir / f'{Path(pdb_path).stem}_{chain}.pdb'
    if not out.exists():
        with open(out, 'w') as f:
            f.writelines(lines)
            f.write('TER\nEND\n')
    return str(out)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float('nan'), float('nan'), n
    xv, yv = x[mask], y[mask]
    if np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return float('nan'), float('nan'), n
    return float(spearmanr(xv, yv).correlation), float(pearsonr(xv, yv)[0]), n


def _compute_cycle_epistasis(df: pd.DataFrame, group_cols: list[str], target_col: str, pred_cols: list[str]) -> pd.DataFrame:
    """Build exact double-mutant cycle table (ddG_AB - ddG_A - ddG_B) for true target and each prediction column."""
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        n_mut = g['mut_type'].astype(str).str.count(':') + 1
        singles = g[n_mut == 1].drop_duplicates('mut_type').set_index('mut_type')
        doubles = g[n_mut == 2]
        if singles.empty or doubles.empty:
            continue
        single_target = singles[target_col].to_dict()
        single_preds = {c: singles[c].to_dict() for c in pred_cols}
        for _, r in doubles.iterrows():
            parts = str(r['mut_type']).split(':')
            if len(parts) != 2 or parts[0] not in single_target or parts[1] not in single_target:
                continue
            y_ab = float(r[target_col])
            y_a, y_b = float(single_target[parts[0]]), float(single_target[parts[1]])
            if not (np.isfinite(y_ab) and np.isfinite(y_a) and np.isfinite(y_b)):
                continue
            rec = {
                'group': key if isinstance(key, str) else '|'.join(map(str, key)),
                'mut_type': r['mut_type'],
                'true_ddG_AB': y_ab,
                'true_dddG': y_ab - (y_a + y_b),
            }
            for c in pred_cols:
                p_ab = float(r[c])
                p_a, p_b = float(single_preds[c][parts[0]]), float(single_preds[c][parts[1]])
                rec[f'{c}_AB'] = p_ab
                rec[f'{c}_cycle_dddG'] = p_ab - (p_a + p_b)
            for m in VALID_MODES:
                ec = f'{m}_epi'
                if ec in r.index:
                    rec[f'{m}_direct_epi'] = float(r[ec])
            rows.append(rec)
    return pd.DataFrame(rows)


def _load_prosst(prosst_dir: str, model_name: str, vocab_size: int, device: str):
    sys.path.insert(0, prosst_dir)
    import prosst.structure.get_sst_seq as gss
    from prosst.structure.get_sst_seq import SSTPredictor
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    def seq_map(func, data, workers=1):
        for x in data:
            yield func(x)
    gss.iter_parallel_map = seq_map

    predictor = SSTPredictor(structure_vocab_size=vocab_size, num_processes=1, num_threads=4, device=device)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return predictor, model, tok


def _score_group(
    df_group: pd.DataFrame,
    seq: str,
    sst_seq: list[int],
    model,
    tok,
    modes: tuple[str, ...],
    batch_size: int,
    max_tokens: int,
    device: str,
) -> tuple[pd.DataFrame, dict]:
    out = df_group.copy()
    muts = out['mut_type'].astype(str).tolist()
    stats = {}
    for m in modes:
        res = score_protein_variants(
            model=model,
            tokenizer=tok,
            seq=seq,
            sst_seq=sst_seq,
            mut_strings=muts,
            mode=m,
            batch_size=batch_size,
            max_tokens=max_tokens,
            device=device,
        )
        out[f'{m}_wt'] = res.wt_pred
        out[f'{m}_mt'] = res.mt_pred
        out[f'{m}_comb'] = res.comb_pred
        out[f'{m}_epi'] = res.epi_pred
        stats[m] = (res.n_wt_passes, res.n_mt_passes, res.elapsed_s)
    return out, stats


def _summarize_dataset(dataset_name: str, df: pd.DataFrame, group_cols: list[str], target_col: str, modes: tuple[str, ...], timing: dict) -> list[dict]:
    summary_rows = []
    n_mut = df['mut_type'].astype(str).str.count(':') + 1
    singles = df[n_mut == 1]
    doubles = df[n_mut == 2]
    multis = df[n_mut >= 2]

    comb_cols = [f'{m}_comb' for m in modes]
    cyc = _compute_cycle_epistasis(df, group_cols, target_col, comb_cols)

    for m in modes:
        col = f'{m}_comb'
        wt_p, mt_p, dt = timing[m]

        # Singles pooled + per-group macro
        s_rho, s_r, s_n = _safe_corr(singles[target_col].to_numpy(float), singles[col].to_numpy(float))
        d_rho, d_r, d_n = _safe_corr(doubles[target_col].to_numpy(float), doubles[col].to_numpy(float))
        m_rho, m_r, m_n = _safe_corr(multis[target_col].to_numpy(float), multis[col].to_numpy(float))

        # Per-group macro Spearman for doubles and epistasis
        d_macro_list, epi_cycle_macro_list, epi_direct_macro_list = [], [], []
        if not doubles.empty:
            for _, g in doubles.groupby(group_cols, sort=False):
                rho_g, _, ng = _safe_corr(g[target_col].to_numpy(float), g[col].to_numpy(float))
                if ng >= 10 and np.isfinite(rho_g):
                    d_macro_list.append(rho_g)

        if not cyc.empty:
            ec_rho, ec_r, ec_n = _safe_corr(cyc['true_dddG'].to_numpy(float), cyc[f'{col}_cycle_dddG'].to_numpy(float))
            ed_rho, ed_r, ed_n = _safe_corr(cyc['true_dddG'].to_numpy(float), cyc[f'{m}_direct_epi'].to_numpy(float))
            for _, g in cyc.groupby('group', sort=False):
                rc, _, ng = _safe_corr(g['true_dddG'].to_numpy(float), g[f'{col}_cycle_dddG'].to_numpy(float))
                rd, _, _ = _safe_corr(g['true_dddG'].to_numpy(float), g[f'{m}_direct_epi'].to_numpy(float))
                if ng >= 10 and np.isfinite(rc):
                    epi_cycle_macro_list.append(rc)
                if ng >= 10 and np.isfinite(rd):
                    epi_direct_macro_list.append(rd)
        else:
            ec_rho = ec_r = ed_rho = ed_r = float('nan')
            ec_n = ed_n = 0

        summary_rows.append({
            'dataset': dataset_name,
            'mode': m,
            'n_singles': s_n,
            'singles_rho': s_rho,
            'n_doubles': d_n,
            'doubles_rho_pooled': d_rho,
            'doubles_rho_macro': float(np.mean(d_macro_list)) if d_macro_list else float('nan'),
            'n_multis': m_n,
            'multis_rho_pooled': m_rho,
            'n_cycles': ec_n,
            'epi_cycle_rho_pooled': ec_rho,
            'epi_cycle_r_pooled': ec_r,
            'epi_cycle_rho_macro': float(np.mean(epi_cycle_macro_list)) if epi_cycle_macro_list else float('nan'),
            'epi_direct_rho_pooled': ed_rho,
            'epi_direct_r_pooled': ed_r,
            'epi_direct_rho_macro': float(np.mean(epi_direct_macro_list)) if epi_direct_macro_list else float('nan'),
            'n_wt_passes': wt_p,
            'n_mt_passes': mt_p,
            'elapsed_s': dt,
        })
    return summary_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='AI4Protein/ProSST-2048')
    ap.add_argument('--vocab', type=int, default=2048)
    ap.add_argument('--prosst_dir', default=os.path.expanduser('~/software/ProSST'))
    ap.add_argument('--split', default='splits_oct06_capped')
    ap.add_argument('--datasets', default='megascale_test,dms,ptmul',
                    help='Comma-separated subset of {megascale_test,dms,ptmul}')
    ap.add_argument('--modes', default=','.join(VALID_MODES))
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_tokens', type=int, default=16384)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = tuple(x.strip() for x in args.modes.split(',') if x.strip())
    datasets = {x.strip() for x in args.datasets.split(',') if x.strip()}

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[ProSST Epistasis] model={args.model} vocab={args.vocab} device={dev} modes={modes}', flush=True)
    t_load = time.perf_counter()
    predictor, model, tok = _load_prosst(args.prosst_dir, args.model, args.vocab, dev)
    print(f'[ProSST Epistasis] Loaded model in {time.perf_counter() - t_load:.1f}s', flush=True)

    sst_cache: dict[str, tuple[str, list[int]]] = {}

    def get_sst(pdb_path: str, chain: str, tmp_dir: Path) -> tuple[str, list[int]]:
        key = f'{pdb_path}::{chain}'
        if key not in sst_cache:
            use_pdb = _extract_chain_pdb(pdb_path, chain, tmp_dir)
            res = predictor.predict_from_pdb([use_pdb])[0]
            sst_cache[key] = (res['aa_seq'], list(res[f'{args.vocab}_sst_seq']))
        return sst_cache[key]

    all_summary: list[dict] = []
    with tempfile.TemporaryDirectory(prefix='prosst_epi_') as tmp:
        tmp_dir = Path(tmp)

        if 'megascale_test' in datasets:
            from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor
            tsu_dir = paths.TSUBOYAMA_DIR
            ds = MegaScaleDatasetPreprocessor(
                data_file=str(tsu_dir / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'),
                af_model_folder=str(tsu_dir / 'AlphaFold_model_PDBs'),
            )
            ds.create_training_splits(str(paths.DATA_DIR / f'{args.split}.pkl'), -1)
            df = ds.split_dfs['test'].copy()
            # Normalize mut_type sorting and ensure ddG_ML numeric
            df['ddG_ML'] = pd.to_numeric(df['ddG_ML'], errors='coerce')
            parts = []
            timing = {m: [0, 0, 0.0] for m in modes}
            n_groups = df.groupby(['code_wt', 'mut_structure']).ngroups
            print(f'[MegaScale test] Scoring {len(df)} rows across {n_groups} (code_wt, mut_structure) groups...', flush=True)
            for k, ((code_wt, mut_struct), g) in enumerate(df.groupby(['code_wt', 'mut_structure'], sort=False), 1):
                seq, pdb_path = library_sequence(g, mut_struct)
                pdb_path = paths.fix_pdb_column(pd.Series([pdb_path])).iloc[0]
                _, st = get_sst(pdb_path, 'A', tmp_dir)
                scored, st_dict = _score_group(g, seq, st, model, tok, modes, args.batch_size, args.max_tokens, dev)
                parts.append(scored)
                for m in modes:
                    timing[m][0] += st_dict[m][0]
                    timing[m][1] += st_dict[m][1]
                    timing[m][2] += st_dict[m][2]
                if k % 10 == 0 or k == n_groups:
                    print(f'  [MegaScale test] {k}/{n_groups} groups done (dual_ind elapsed={timing.get("dual_independent", [0,0,0])[2]:.1f}s)', flush=True)
            full_df = pd.concat(parts, axis=0)
            full_df.to_csv(out_dir / 'megascale_test.csv.gz', index=False, compression='gzip')
            rows = _summarize_dataset('megascale_test', full_df, ['code_wt', 'mut_structure'], 'ddG_ML', modes, {m: tuple(v) for m, v in timing.items()})
            all_summary.extend(rows)

        if 'dms' in datasets:
            for assay in DMS:
                df = dms_rows(assay)
                parts = []
                timing = {m: [0, 0, 0.0] for m in modes}
                print(f'[DMS {assay}] Scoring {len(df)} rows...', flush=True)
                for (pdb_file, code, chain), g in df.groupby(['pdb_file', 'code', 'chain'], sort=False):
                    aa_seq, st = get_sst(str(pdb_file), str(chain), tmp_dir)
                    scored, st_dict = _score_group(g, aa_seq, st, model, tok, modes, args.batch_size, args.max_tokens, dev)
                    parts.append(scored)
                    for m in modes:
                        timing[m][0] += st_dict[m][0]
                        timing[m][1] += st_dict[m][1]
                        timing[m][2] += st_dict[m][2]
                full_df = pd.concat(parts, axis=0)
                full_df.to_csv(out_dir / f'dms_{assay}.csv.gz', index=False, compression='gzip')
                rows = _summarize_dataset(f'dms_{assay}', full_df, ['code'], 'ddG', modes, {m: tuple(v) for m, v in timing.items()})
                all_summary.extend(rows)

        if 'ptmul' in datasets:
            for ext_name in ('ptmul', 'ptmuld'):
                df = external_rows(ext_name)
                parts = []
                timing = {m: [0, 0, 0.0] for m in modes}
                print(f'[EXT {ext_name}] Scoring {len(df)} rows across {df["code"].nunique()} proteins...', flush=True)
                for (pdb_file, code, chain), g in df.groupby(['pdb_file', 'code', 'chain'], sort=False):
                    try:
                        aa_seq, st = get_sst(str(pdb_file), str(chain), tmp_dir)
                        scored, st_dict = _score_group(g, aa_seq, st, model, tok, modes, args.batch_size, args.max_tokens, dev)
                        parts.append(scored)
                        for m in modes:
                            timing[m][0] += st_dict[m][0]
                            timing[m][1] += st_dict[m][1]
                            timing[m][2] += st_dict[m][2]
                    except Exception as e:
                        print(f'  [EXT {ext_name}] SKIP {code}{chain}: {type(e).__name__}: {e}', flush=True)
                if parts:
                    full_df = pd.concat(parts, axis=0)
                    full_df.to_csv(out_dir / f'ext_{ext_name}.csv.gz', index=False, compression='gzip')
                    rows = _summarize_dataset(f'ext_{ext_name}', full_df, ['code', 'chain'], 'ddG', modes, {m: tuple(v) for m, v in timing.items()})
                    all_summary.extend(rows)

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(out_dir / 'epistasis_summary.csv', index=False)
    print('\n=== SUMMARY ===', flush=True)
    cols = ['dataset', 'mode', 'singles_rho', 'doubles_rho_pooled', 'doubles_rho_macro',
            'multis_rho_pooled', 'epi_cycle_rho_pooled', 'epi_cycle_rho_macro', 'epi_direct_rho_pooled',
            'n_wt_passes', 'n_mt_passes', 'elapsed_s']
    print(summary_df[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'), flush=True)


if __name__ == '__main__':
    main()
