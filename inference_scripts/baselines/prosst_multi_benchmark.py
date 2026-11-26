"""Full multi-mutant ProSST benchmark runner across all manuscript benchmark families.

Runs one explicit scoring mode (`dual_unmasked` or `dual_independent`) from `prosst_scoring.py` across every benchmark
that contains multi-mutants (including all component singles on those proteins/assays so double-mutant cycles are exact):

  1. megascale : `splits_oct06_capped` (val, test) and `mega_splits` (val, test)
  2. external  : `ptmul` and `ptmuld`
  3. dms       : `DLG4_HUMAN_Faure_2021_binding_domain`, `GRB2_HUMAN_Faure_2021_binding_domain`, `MYO_HUMAN_Kung_2025_display`
  4. mgnify    : `rocklin_denovo` (all 1,000 scaffolds) and `mgnify` (all scaffolds containing >= 1 multi-mutant, plus their singles)
  5. pgym      : all 69 multi-mutant ProteinGym assays (sorted smallest-first by variant count)

Every output file is written atomically and skipped on rerun, so long jobs resume cleanly if interrupted.
Note: `dual_unmasked` writes `wt_pred` (= exact native `wt_unmasked`), `mt_pred`, `comb_pred`, and `epi_pred`;
      `dual_independent` writes `wt_pred` (= exact `wt_independent`), `mt_pred`, `comb_pred`, and `epi_pred`.
"""
from __future__ import annotations

import argparse
import gc
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
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get('MSR_REPO') or HERE.parents[1]).resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / 'inference_scripts' / 'baselines'))
sys.path.insert(0, str(REPO_ROOT / 'src'))

import mgnify_rows  # noqa: E402
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


def _atomic_to_csv(df: pd.DataFrame, path: Path, compress: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    df.to_csv(tmp, index=False, compression='gzip' if compress else None)
    os.replace(tmp, path)


def _safe_rho(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float('nan')
    if np.std(x[m]) == 0.0 or np.std(y[m]) == 0.0:
        return float('nan')
    return float(spearmanr(x[m], y[m]).correlation)


def _load_prosst(prosst_dir: str, model_name: str, vocab_size: int, device: str):
    sys.path.insert(0, prosst_dir)
    import prosst.structure.get_sst_seq as gss
    from prosst.structure.get_sst_seq import SSTPredictor
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    def seq_map(func, data, workers=1):
        for x in data:
            yield func(x)
    gss.iter_parallel_map = seq_map

    predictor = SSTPredictor(structure_vocab_size=vocab_size, max_batch_nodes=2500, num_processes=1, num_threads=4, device=device)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return predictor, model, tok


def _apply_result(df: pd.DataFrame, res) -> pd.DataFrame:
    out = df.copy()
    out['wt_pred'] = res.wt_pred
    out['mt_pred'] = res.mt_pred
    out['comb_pred'] = res.comb_pred
    out['epi_pred'] = res.epi_pred
    return out


def run_megascale(out_dir: Path, get_sst, model, tok, args, dev: str, tmp_dir: Path) -> None:
    from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor
    tsu_dir = paths.TSUBOYAMA_DIR
    splits = [s.strip() for s in args.megascale_splits.split(',') if s.strip()]
    for split_name in splits:
        split_pkl = paths.DATA_DIR / f'{split_name}.pkl'
        if not split_pkl.exists():
            print(f'[MegaScale] SKIP missing split {split_pkl}', flush=True)
            continue
        targets = [(part, out_dir / 'megascale' / f'{split_name}_{part}.csv.gz') for part in ('val', 'test')]
        if all(p.exists() for _, p in targets):
            print(f'[MegaScale] {split_name} val/test already done; skipping.', flush=True)
            continue
        ds = MegaScaleDatasetPreprocessor(
            data_file=str(tsu_dir / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'),
            af_model_folder=str(tsu_dir / 'AlphaFold_model_PDBs'),
        )
        ds.create_training_splits(str(split_pkl), -1)
        for part, dest in targets:
            if dest.exists():
                print(f'[MegaScale] {split_name}_{part} already exists; skipping.', flush=True)
                continue
            df = ds.split_dfs[part].copy()
            df['mut_structure'] = df['mut_structure'].fillna('-').astype(str)
            df['ddG_ML'] = pd.to_numeric(df['ddG_ML'], errors='coerce')
            t0 = time.perf_counter()
            chunks = []
            tot_wt, tot_mt = 0, 0
            groups = list(df.groupby(['code_wt', 'mut_structure'], sort=False, dropna=False))
            print(f'[MegaScale] {split_name}_{part}: {len(df):,} rows across {len(groups)} libraries ({args.mode})...', flush=True)
            for (code_wt, mut_struct), g in groups:
                seq, pdb_path = library_sequence(g, mut_struct)
                pdb_path = paths.fix_pdb_column(pd.Series([pdb_path])).iloc[0]
                _, st = get_sst(pdb_path, 'A', tmp_dir)
                res = score_protein_variants(
                    model, tok, seq, st, g['mut_type'].astype(str).tolist(),
                    mode=args.mode, batch_size=args.batch_size, max_tokens=args.max_tokens, device=dev,
                )
                chunks.append(_apply_result(g, res))
                tot_wt += res.n_wt_passes
                tot_mt += res.n_mt_passes
            full = pd.concat(chunks, axis=0)
            _atomic_to_csv(full, dest, compress=True)
            n_mut = full['mut_type'].astype(str).str.count(':') + 1
            d_mask = n_mut == 2
            d_wt = _safe_rho(full.loc[d_mask, 'ddG_ML'].to_numpy(float), full.loc[d_mask, 'wt_pred'].to_numpy(float))
            d_cb = _safe_rho(full.loc[d_mask, 'ddG_ML'].to_numpy(float), full.loc[d_mask, 'comb_pred'].to_numpy(float))
            print(f'  DONE {split_name}_{part} in {time.perf_counter()-t0:.1f}s '
                  f'(passes={tot_wt}+{tot_mt}, doubles_rho_pooled wt={d_wt:.4f} comb={d_cb:.4f})', flush=True)


def run_external(out_dir: Path, get_sst, model, tok, args, dev: str, tmp_dir: Path) -> None:
    for ext_name in ('ptmul', 'ptmuld'):
        dest = out_dir / 'external' / f'{ext_name}.csv.gz'
        if dest.exists():
            print(f'[External] {ext_name} already exists; skipping.', flush=True)
            continue
        df = external_rows(ext_name)
        t0 = time.perf_counter()
        chunks, tot_wt, tot_mt = [], 0, 0
        print(f'[External] {ext_name}: {len(df):,} rows across {df["code"].nunique()} proteins ({args.mode})...', flush=True)
        for (pdb_file, code, chain), g in df.groupby(['pdb_file', 'code', 'chain'], sort=False):
            try:
                aa_seq, st = get_sst(str(pdb_file), str(chain), tmp_dir)
                res = score_protein_variants(
                    model, tok, aa_seq, st, g['mut_type'].astype(str).tolist(),
                    mode=args.mode, batch_size=args.batch_size, max_tokens=args.max_tokens, device=dev,
                )
                chunks.append(_apply_result(g, res))
                tot_wt += res.n_wt_passes
                tot_mt += res.n_mt_passes
            except Exception as e:
                print(f'  [External {ext_name}] SKIP {code}{chain}: {type(e).__name__}: {e}', flush=True)
        full = pd.concat(chunks, axis=0)
        _atomic_to_csv(full, dest, compress=True)
        r_wt = _safe_rho(full['ddG'].to_numpy(float), full['wt_pred'].to_numpy(float))
        r_cb = _safe_rho(full['ddG'].to_numpy(float), full['comb_pred'].to_numpy(float))
        print(f'  DONE {ext_name} in {time.perf_counter()-t0:.1f}s (passes={tot_wt}+{tot_mt}, rho wt={r_wt:.4f} comb={r_cb:.4f})', flush=True)


def run_dms(out_dir: Path, get_sst, model, tok, args, dev: str, tmp_dir: Path) -> None:
    for assay in DMS:
        dest = out_dir / 'dms' / f'{assay}.csv.gz'
        if dest.exists():
            print(f'[DMS] {assay} already exists; skipping.', flush=True)
            continue
        df = dms_rows(assay)
        t0 = time.perf_counter()
        chunks, tot_wt, tot_mt = [], 0, 0
        print(f'[DMS] {assay}: {len(df):,} rows ({args.mode})...', flush=True)
        for (pdb_file, code, chain), g in df.groupby(['pdb_file', 'code', 'chain'], sort=False):
            aa_seq, st = get_sst(str(pdb_file), str(chain), tmp_dir)
            res = score_protein_variants(
                model, tok, aa_seq, st, g['mut_type'].astype(str).tolist(),
                mode=args.mode, batch_size=args.batch_size, max_tokens=args.max_tokens, device=dev,
            )
            chunks.append(_apply_result(g, res))
            tot_wt += res.n_wt_passes
            tot_mt += res.n_mt_passes
        full = pd.concat(chunks, axis=0)
        _atomic_to_csv(full, dest, compress=True)
        r_wt = _safe_rho(full['ddG'].to_numpy(float), full['wt_pred'].to_numpy(float))
        r_cb = _safe_rho(full['ddG'].to_numpy(float), full['comb_pred'].to_numpy(float))
        print(f'  DONE {assay} in {time.perf_counter()-t0:.1f}s (passes={tot_wt}+{tot_mt}, rho wt={r_wt:.4f} comb={r_cb:.4f})', flush=True)


def run_mgnify(out_dir: Path, predictor, model, tok, args, dev: str) -> None:
    mgnify_dir = Path(args.mgnify_dir)
    for source in ('rocklin_denovo', 'mgnify'):
        df = mgnify_rows.load(str(mgnify_dir), (source,))
        if source == 'mgnify':
            # Keep ALL rows (multis + component singles) for every scaffold that has >= 1 multi-mutant
            multi_codes = set(df.loc[df['mut_type'].astype(str).str.contains(':'), 'code'])
            df = df[df['code'].isin(multi_codes)].reset_index(drop=True)
        dest_dir = out_dir / 'mgnify' / source
        dest_dir.mkdir(parents=True, exist_ok=True)
        todo = list(mgnify_rows.chunks(df, args.mgnify_chunk))
        print(f'[MGnify {source}] {len(df):,} rows across {df["code"].nunique():,} scaffolds in {len(todo)} chunks ({args.mode})...', flush=True)
        t0 = time.perf_counter()
        for k, part in todo:
            dest = dest_dir / f'chunk_{k:04d}.csv.gz'
            if dest.exists():
                continue
            part = part.reset_index(drop=True)
            scaf = part.drop_duplicates('code')[['code', 'pdb_file', 'overwrite']]
            sst = {}
            pdb_files = list(scaf['pdb_file'])
            torch.cuda.empty_cache()
            for p_start in range(0, len(pdb_files), 64):
                for r in predictor.predict_from_pdb(pdb_files[p_start:p_start + 64]):
                    sst[os.path.splitext(r['name'])[0]] = (r['aa_seq'], list(r[f'{args.vocab}_sst_seq']))
                torch.cuda.empty_cache()
            wt_arr = np.full(len(part), np.nan, dtype=np.float64)
            mt_arr = np.full(len(part), np.nan, dtype=np.float64)
            cb_arr = np.full(len(part), np.nan, dtype=np.float64)
            ep_arr = np.full(len(part), np.nan, dtype=np.float64)
            tot_wt = tot_mt = 0
            for code, g in part.groupby('code', sort=False):
                stem = os.path.splitext(os.path.basename(g['pdb_file'].iloc[0]))[0]
                if stem not in sst:
                    continue
                aa_seq, st = sst[stem]
                ow = g['overwrite'].iloc[0]
                seq = ow if isinstance(ow, str) else aa_seq
                if len(st) != len(seq):
                    continue
                res = score_protein_variants(
                    model, tok, seq, st, g['mut_type'].astype(str).tolist(),
                    mode=args.mode, batch_size=args.batch_size, max_tokens=args.max_tokens, device=dev,
                )
                idx = g.index.to_numpy()
                wt_arr[idx] = res.wt_pred
                mt_arr[idx] = res.mt_pred
                cb_arr[idx] = res.comb_pred
                ep_arr[idx] = res.epi_pred
                tot_wt += res.n_wt_passes
                tot_mt += res.n_mt_passes
            out = part[['id', 'code', 'mut_type', 'source']].copy()
            out['wt_pred'] = wt_arr
            out['mt_pred'] = mt_arr
            out['comb_pred'] = cb_arr
            out['epi_pred'] = ep_arr
            _atomic_to_csv(out, dest, compress=True)
            print(f'  [MGnify {source}] chunk {k+1}/{len(todo)} ({len(part):,} rows, passes={tot_wt}+{tot_mt}, '
                  f'{(time.perf_counter()-t0)/60:.1f} min elapsed)', flush=True)


def run_pgym(out_dir: Path, get_sst, model, tok, args, dev: str, tmp_dir: Path) -> None:
    pg_dir = Path(args.pgym_dir)
    pg_out = out_dir / 'proteingym_multi'
    pg_out.mkdir(parents=True, exist_ok=True)

    # Identify all 69 multi-mutant assays and sort smallest-first
    assays = []
    for p in sorted(pg_dir.glob('*.csv')):
        if p.name == 'summary.csv':
            continue
        hdr = pd.read_csv(p, nrows=2)
        mcol = 'mut_type_renumbered' if 'mut_type_renumbered' in hdr.columns else ('mut_type' if 'mut_type' in hdr.columns else 'mutant')
        s = pd.read_csv(p, usecols=[mcol])[mcol].astype(str)
        n_multi = int(s.str.contains(':').sum())
        if n_multi > 0:
            assays.append((len(s), n_multi, p, mcol))
    assays.sort(key=lambda x: x[0])
    print(f'[PGym multi] Found {len(assays)} multi-mutant assays (total rows={sum(a[0] for a in assays):,}, mode={args.mode})', flush=True)

    summary_rows = []
    t_all = time.perf_counter()
    for i, (n_rows, n_multi, csv_p, mcol) in enumerate(assays, 1):
        dms = csv_p.stem
        dest = pg_out / f'{dms}.csv.gz'
        if dest.exists():
            prev = pd.read_csv(dest)
            y_col = 'DMS_score' if 'DMS_score' in prev.columns else ('ddG' if 'ddG' in prev.columns else None)
            r_wt = _safe_rho(prev[y_col].to_numpy(float), prev['wt_pred'].to_numpy(float)) if y_col else float('nan')
            r_cb = _safe_rho(prev[y_col].to_numpy(float), prev['comb_pred'].to_numpy(float)) if y_col else float('nan')
            summary_rows.append({'dms': dms, 'status': 'ok', 'n_rows': len(prev), 'n_multi': n_multi,
                                 'rho_wt': r_wt, 'rho_comb': r_cb, 'n_wt_passes': np.nan, 'n_mt_passes': np.nan, 'elapsed_s': 0.0})
            continue

        df = pd.read_csv(csv_p)
        pdb_path = paths.fix_pdb_column(df['pdb_file']).iloc[0]
        if not os.path.exists(pdb_path):
            alt = Path.home() / 'ProteinGym' / str(df['pdb_file'].iloc[0])
            if alt.exists():
                pdb_path = str(alt)
        chain = str(df['chain'].iloc[0]) if 'chain' in df.columns else 'A'
        y_col = 'DMS_score' if 'DMS_score' in df.columns else ('ddG' if 'ddG' in df.columns else None)
        try:
            aa_seq, st = get_sst(str(pdb_path), chain, tmp_dir)
            res = score_protein_variants(
                model, tok, aa_seq, st, df[mcol].astype(str).tolist(),
                mode=args.mode, batch_size=args.batch_size, max_tokens=args.max_tokens, device=dev,
                verify_wt=False,
            )
            out_df = _apply_result(df, res)
            _atomic_to_csv(out_df, dest, compress=True)
            r_wt = _safe_rho(out_df[y_col].to_numpy(float), out_df['wt_pred'].to_numpy(float)) if y_col else float('nan')
            r_cb = _safe_rho(out_df[y_col].to_numpy(float), out_df['comb_pred'].to_numpy(float)) if y_col else float('nan')
            summary_rows.append({
                'dms': dms, 'status': 'ok', 'n_rows': len(df), 'n_multi': n_multi,
                'rho_wt': r_wt, 'rho_comb': r_cb,
                'n_wt_passes': res.n_wt_passes, 'n_mt_passes': res.n_mt_passes, 'elapsed_s': res.elapsed_s,
            })
            print(f'  [PGym {i}/{len(assays)}] DONE {dms} (L={len(aa_seq)}, rows={len(df):,}, multi={n_multi:,}, '
                  f'passes={res.n_wt_passes}+{res.n_mt_passes}, {res.elapsed_s:.1f}s) '
                  f'rho wt={r_wt:.4f} comb={r_cb:.4f} (total {(time.perf_counter()-t_all)/60:.1f}m)', flush=True)
        except Exception as e:
            print(f'  [PGym {i}/{len(assays)}] FAIL {dms}: {type(e).__name__}: {str(e)[:200]}', flush=True)
            summary_rows.append({'dms': dms, 'status': f'error:{type(e).__name__}', 'n_rows': len(df), 'n_multi': n_multi,
                                 'rho_wt': np.nan, 'rho_comb': np.nan, 'n_wt_passes': np.nan, 'n_mt_passes': np.nan, 'elapsed_s': np.nan})
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        # Persist running summary after each assay
        _atomic_to_csv(pd.DataFrame(summary_rows), pg_out / 'summary.csv', compress=False)

    sdf = pd.DataFrame(summary_rows)
    _atomic_to_csv(sdf, pg_out / 'summary.csv', compress=False)
    ok = sdf[sdf['status'] == 'ok']
    print(f'[PGym multi] COMPLETE: {len(ok)}/{len(sdf)} ok, mean rho_wt={ok["rho_wt"].mean():.4f}, mean rho_comb={ok["rho_comb"].mean():.4f}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', required=True, choices=VALID_MODES)
    ap.add_argument('--model', default='AI4Protein/ProSST-2048')
    ap.add_argument('--vocab', type=int, default=2048)
    ap.add_argument('--prosst_dir', default=os.path.expanduser('~/software/ProSST'))
    ap.add_argument('--suites', default='megascale,external,dms,mgnify,pgym',
                    help='Comma-separated suites in execution order')
    ap.add_argument('--megascale_splits', default='splits_oct06_capped,mega_splits')
    ap.add_argument('--mgnify_dir', default=str(REPO_ROOT / 'data' / 'mgnify'))
    ap.add_argument('--mgnify_chunk', type=int, default=20000)
    ap.add_argument('--pgym_dir', default=str(paths.PGYM_INPUTS))
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--max_tokens', type=int, default=32768)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = [s.strip() for s in args.suites.split(',') if s.strip()]

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[ProSST Multi Benchmark] mode={args.mode} model={args.model} suites={suites} dev={dev} '
          f'bs={args.batch_size} max_tok={args.max_tokens}', flush=True)
    t_load = time.perf_counter()
    predictor, model, tok = _load_prosst(args.prosst_dir, args.model, args.vocab, dev)
    print(f'[ProSST Multi Benchmark] Loaded in {time.perf_counter()-t_load:.1f}s', flush=True)

    sst_cache: dict[str, tuple[str, list[int]]] = {}

    def get_sst(pdb_path: str, chain: str, tmp_dir: Path) -> tuple[str, list[int]]:
        key = f'{pdb_path}::{chain}'
        if key not in sst_cache:
            use_pdb = _extract_chain_pdb(pdb_path, chain, tmp_dir)
            res = predictor.predict_from_pdb([use_pdb])[0]
            sst_cache[key] = (res['aa_seq'], list(res[f'{args.vocab}_sst_seq']))
        return sst_cache[key]

    with tempfile.TemporaryDirectory(prefix='prosst_multi_') as tmp:
        tmp_dir = Path(tmp)
        for suite in suites:
            if suite == 'megascale':
                run_megascale(out_dir, get_sst, model, tok, args, dev, tmp_dir)
            elif suite == 'external':
                run_external(out_dir, get_sst, model, tok, args, dev, tmp_dir)
            elif suite == 'dms':
                run_dms(out_dir, get_sst, model, tok, args, dev, tmp_dir)
            elif suite == 'mgnify':
                run_mgnify(out_dir, predictor, model, tok, args, dev)
            elif suite == 'pgym':
                run_pgym(out_dir, get_sst, model, tok, args, dev, tmp_dir)
            else:
                raise ValueError(f'Unknown suite: {suite!r}')

    print(f'=== ALL SUITES COMPLETE ({args.mode}) -> {out_dir} ===', flush=True)


if __name__ == '__main__':
    main()
