"""ProSST-2048 (AI4Protein/ProSST-2048) fixed-slot benchmark runner with per-protein resource tracking.

Scores the exact shared benchmark rows (benchmark_rows.py + pgym_runner.py) across:
  - MegaScale test (tsuboyama_test from splits_oct06_capped.pkl)
  - All 12 External sets (EXTERNAL)
  - Domainome (522 domains)
  - All 217 ProteinGym assays (pgym_inputs)

Writes standard resource_tracker outputs under <out>/resources/<run_id>/resources_{run.json,scopes.jsonl,scopes.csv}.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO_ROOT / 'src'))

import paths  # noqa: E402
import resource_tracker  # noqa: E402
from benchmark_rows import EXTERNAL, external_rows, domainome_rows, library_sequence  # noqa: E402


def _extract_chain_pdb(pdb_path: str, chain: str, tmp_dir: Path) -> str:
    """Write a clean single-chain PDB containing only ATOM lines of `chain` (avoids multi-chain SST mismatches)."""
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
    with open(out, 'w') as f:
        f.writelines(lines)
        f.write('TER\nEND\n')
    return str(out)


def _structure_tokens(sst_seq):
    return torch.tensor([[1, *[i + 3 for i in sst_seq], 2]], dtype=torch.long)


class ProSSTRunner:
    def __init__(self, args, tracker):
        self.args = args
        self.tracker = tracker
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        sys.path.insert(0, args.prosst_dir)
        import prosst.structure.get_sst_seq as gss
        from prosst.structure.get_sst_seq import SSTPredictor
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        def seq_map(func, data, workers=1):
            for x in data:
                yield func(x)
        gss.iter_parallel_map = seq_map

        with tracker.stage('model_load'):
            self.predictor = SSTPredictor(structure_vocab_size=args.vocab, num_processes=1, num_threads=4, device=self.dev)
            self.model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True).to(self.dev).eval()
            self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            self.vocab = self.tok.get_vocab()

    def score_protein(self, name: str, pdb_path: str, chain: str, mut_strings: list[str],
                      overwrite_seq: str | None = None, tmp_dir: Path | None = None) -> np.ndarray:
        scores = np.full(len(mut_strings), np.nan, dtype=np.float64)
        use_pdb = _extract_chain_pdb(pdb_path, chain, tmp_dir) if tmp_dir is not None else pdb_path
        with self.tracker.scope('protein', name, n_rows=len(mut_strings)) as sc:
            with sc.phase('sst'):
                res = self.predictor.predict_from_pdb([use_pdb])[0]
                aa_seq, st = res['aa_seq'], res[f'{self.args.vocab}_sst_seq']
            seq = overwrite_seq if isinstance(overwrite_seq, str) else aa_seq
            sc.note(seq_len=len(seq))
            if len(st) != len(seq):
                raise ValueError(f'SST length {len(st)} != seq length {len(seq)} for {name}')
            enc = self.tok([seq], return_tensors='pt')
            ss_ids = _structure_tokens(st).to(self.dev)
            with sc.phase('forward'), torch.no_grad():
                lg = self.model(
                    input_ids=enc['input_ids'].to(self.dev),
                    attention_mask=enc['attention_mask'].to(self.dev),
                    ss_input_ids=ss_ids,
                ).logits
                lp = torch.log_softmax(lg[:, 1:-1, :], dim=-1)[0].float().cpu().numpy()
            with sc.phase('gather'):
                for i, m in enumerate(mut_strings):
                    tot = 0.0
                    ok = True
                    for sub in str(m).split(':'):
                        wt, pos, mt = sub[0], int(sub[1:-1]) - 1, sub[-1]
                        if not (0 <= pos < len(seq)) or wt not in self.vocab or mt not in self.vocab:
                            ok = False
                            break
                        tot += float(lp[pos, self.vocab[mt]] - lp[pos, self.vocab[wt]])
                    if ok:
                        scores[i] = tot
        torch.cuda.empty_cache()
        return scores

    def score_table(self, df: pd.DataFrame, label: str, tmp_dir: Path) -> pd.DataFrame:
        out = df.copy()
        preds = np.full(len(df), np.nan, dtype=np.float64)
        pos_map = {idx: k for k, idx in enumerate(df.index)}
        with self.tracker.scope('dataset', label, n_rows=len(df)):
            for (pdb_file, code, chain), g in df.groupby(['pdb_file', 'code', 'chain'], sort=False):
                ow = g['overwrite'].iloc[0] if 'overwrite' in g.columns else None
                try:
                    sc_vals = self.score_protein(
                        f'{code}{chain}', str(pdb_file), str(chain), g['mut_type'].tolist(),
                        overwrite_seq=ow if isinstance(ow, str) else None, tmp_dir=tmp_dir,
                    )
                    for r_idx, val in zip(g.index, sc_vals):
                        preds[pos_map[r_idx]] = val
                except Exception as e:
                    print(f'[ProSST] {label}/{code}{chain}: FAIL {type(e).__name__}: {str(e)[:180]}', flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
        out['prosst'] = preds
        return out

    def score_megascale_test(self, split_name: str, out_dir: Path, tmp_dir: Path) -> None:
        from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor
        tsu_dir = paths.TSUBOYAMA_DIR
        ds = MegaScaleDatasetPreprocessor(
            data_file=str(tsu_dir / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'),
            af_model_folder=str(tsu_dir / 'AlphaFold_model_PDBs'),
        )
        ds.create_training_splits(str(paths.DATA_DIR / f'{split_name}.pkl'), -1)
        df = ds.split_dfs['test'].copy()
        parts = []
        with self.tracker.scope('dataset', 'tsuboyama_test', n_rows=len(df)):
            for code, df_true in df.groupby('code_wt', sort=False):
                df_true = df_true.copy()
                for mut_structure, data in df_true.groupby('mut_structure', sort=False):
                    seq, pdb_path = library_sequence(data, mut_structure)
                    pdb_path = paths.fix_pdb_column(pd.Series([pdb_path])).iloc[0]
                    name = code if mut_structure == '-' else f'{code}_{mut_structure}'
                    try:
                        vals = self.score_protein(name, pdb_path, 'A', data['mut_type'].tolist(), overwrite_seq=seq, tmp_dir=tmp_dir)
                        df_true.loc[data.index, 'prosst'] = vals
                    except Exception as e:
                        print(f'[ProSST] tsuboyama_test/{name}: FAIL {type(e).__name__}: {str(e)[:180]}', flush=True)
                parts.append(df_true)
        pd.concat(parts, axis=0).to_csv(out_dir / 'tsuboyama_test.csv', index=False)

    def score_proteingym(self, pgym_dir: Path, out_dir: Path, tmp_dir: Path) -> None:
        pg_out = out_dir / 'proteingym'
        pg_out.mkdir(parents=True, exist_ok=True)
        csvs = sorted(p for p in pgym_dir.glob('*.csv') if p.name != 'summary.csv')
        summary_rows = []
        with self.tracker.scope('dataset', 'proteingym', n_dms=len(csvs)):
            for csv_p in csvs:
                dms = csv_p.stem
                df = pd.read_csv(csv_p)
                pdb_path = paths.fix_pdb_column(df['pdb_file']).iloc[0]
                if not os.path.exists(pdb_path):
                    alt = Path.home() / 'ProteinGym' / str(df['pdb_file'].iloc[0])
                    if alt.exists():
                        pdb_path = str(alt)
                chain = str(df['chain'].iloc[0]) if 'chain' in df.columns else 'A'
                mut_col = 'mut_type_renumbered' if 'mut_type_renumbered' in df.columns else 'mut_type'
                try:
                    vals = self.score_protein(dms, pdb_path, chain, df[mut_col].tolist(), tmp_dir=tmp_dir)
                    df['prosst'] = vals
                    df.to_csv(pg_out / f'{dms}.csv', index=False)
                    summary_rows.append({'dms': dms, 'status': 'ok', 'n_rows': len(df)})
                except Exception as e:
                    print(f'[ProSST] proteingym/{dms}: FAIL {type(e).__name__}: {str(e)[:180]}', flush=True)
                    summary_rows.append({'dms': dms, 'status': f'error:{type(e).__name__}', 'n_rows': len(df)})
                    gc.collect()
                    torch.cuda.empty_cache()
        pd.DataFrame(summary_rows).to_csv(pg_out / 'summary.csv', index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='splits_oct06_capped')
    ap.add_argument('--pgym_dir', default=str(paths.PGYM_INPUTS))
    ap.add_argument('--prosst_dir', default=os.path.expanduser('~/software/ProSST'))
    ap.add_argument('--model', default='AI4Protein/ProSST-2048')
    ap.add_argument('--vocab', type=int, default=2048)
    resource_tracker.add_tracking_args(ap)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = resource_tracker.Tracker.from_args(
        args, label='prosst', default_dir=out_dir / 'resources',
        meta={'model': args.model, 'vocab': args.vocab, 'split': args.split},
    )
    try:
        runner = ProSSTRunner(args, tracker)
        with tempfile.TemporaryDirectory(prefix='prosst_chains_') as tmp:
            tmp_dir = Path(tmp)
            runner.score_megascale_test(args.split, out_dir, tmp_dir)
            for name in EXTERNAL:
                runner.score_table(external_rows(name), f'external_{name}', tmp_dir).to_csv(
                    out_dir / f'external_{name}.csv', index=False
                )
            runner.score_table(domainome_rows(), 'domainome', tmp_dir).to_csv(out_dir / 'domainome.csv', index=False)
            runner.score_proteingym(Path(args.pgym_dir), out_dir, tmp_dir)
    finally:
        tracker.close()
    print('Done ProSST tracked benchmark:', out_dir)


if __name__ == '__main__':
    main()
