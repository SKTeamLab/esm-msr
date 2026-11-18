"""ProSST (Li et al., NeurIPS 2024; AI4Protein/ProSST-2048) zero-shot on the MGnify scaffolds and Rocklin designs (mgnify_rows.py inputs).

The ProteinGym protocol of the authors (zero_shot/proteingym_benchmark.py): one forward pass of the wild-type sequence with its quantized structure
(SSTPredictor, 2048 structure tokens), score of a variant = sum over its mutations of log p(mutant aa) - log p(wild-type aa) at the position
(positive = stabilizing, like every other readout here).  A backbone-mutated scaffold reads the library's wild-type sequence on the structure of its
backbone (the sequence the measurements are stated against).  Output <out>/mgnify/<source>/chunk_NNNN.csv.gz (id, code, mut_type, source, prosst),
about --mgnify_chunk rows each; a rerun skips finished chunks.

  python prosst_inference.py --mgnify_dir data/mgnify --out outputs/prosst [--prosst_dir ~/software/ProSST] [--model AI4Protein/ProSST-2048]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mgnify_rows  # noqa: E402


def structure_tokens(sst_seq):
    """ProSST's input: 1, the 2048-vocab labels shifted by 3, 2 (zero_shot/proteingym_benchmark.py)."""
    return torch.tensor([[1, *[i + 3 for i in sst_seq], 2]], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mgnify_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mgnify_sources', default='rocklin_denovo,mgnify')
    ap.add_argument('--mgnify_chunk', type=int, default=20000)
    ap.add_argument('--max_proteins', type=int, default=0, help='smoke test: only the first N scaffolds of every source')
    ap.add_argument('--prosst_dir', default=os.environ.get('PROSST_DIR', os.path.expanduser('~/software/ProSST')))
    ap.add_argument('--model', default='AI4Protein/ProSST-2048')
    ap.add_argument('--vocab', type=int, default=2048)
    ap.add_argument('--num_processes', type=int, default=4, help='graph building workers of the structure quantizer')
    a = ap.parse_args()
    sys.path.insert(0, a.prosst_dir)
    from prosst.structure.get_sst_seq import SSTPredictor
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = SSTPredictor(structure_vocab_size=a.vocab, num_processes=a.num_processes, num_threads=a.num_processes, device=dev)
    model = AutoModelForMaskedLM.from_pretrained(a.model, trust_remote_code=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    vocab = tok.get_vocab()
    skipped = []
    for source in a.mgnify_sources.split(','):
        df = mgnify_rows.load(a.mgnify_dir, (source,))
        if a.max_proteins:
            df = df[df['code'].isin(list(dict.fromkeys(df['code']))[:a.max_proteins])]
        dest = Path(a.out) / 'mgnify' / source
        dest.mkdir(parents=True, exist_ok=True)
        todo = list(mgnify_rows.chunks(df, a.mgnify_chunk))
        t0 = time.time()
        for k, part in todo:
            path = dest / f'chunk_{k:04d}.csv.gz'
            if path.exists():
                continue
            part = part.reset_index(drop=True)
            scaf = part.drop_duplicates('code')[['code', 'pdb_file', 'overwrite']]
            sst = {}
            for r in predictor.predict_from_pdb(list(scaf['pdb_file'])):                # one batched pass over the chunk's structures
                sst[os.path.splitext(r['name'])[0]] = (r['aa_seq'], r[f'{a.vocab}_sst_seq'])
            score = np.full(len(part), np.nan)
            for code, g in part.groupby('code', sort=False):
                stem = os.path.splitext(os.path.basename(g['pdb_file'].iloc[0]))[0]
                if stem not in sst:
                    skipped.append(code)
                    continue
                aa_seq, st = sst[stem]
                ow = g['overwrite'].iloc[0]
                seq = ow if isinstance(ow, str) else aa_seq
                if len(st) != len(seq):
                    skipped.append(code)
                    continue
                enc = tok([seq], return_tensors='pt')
                with torch.no_grad():
                    lg = model(input_ids=enc['input_ids'].to(dev), attention_mask=enc['attention_mask'].to(dev), ss_input_ids=structure_tokens(st).to(dev)).logits
                lp = torch.log_softmax(lg[:, 1:-1, :], dim=-1)[0].float().cpu()
                for i, m in zip(g.index, g['mut_type']):
                    tot = 0.0
                    for sub in m.split(':'):
                        wt, pos, mt = sub[0], int(sub[1:-1]) - 1, sub[-1]
                        assert seq[pos] == wt, f'{code}: {sub} does not match the sequence'
                        tot += float(lp[pos, vocab[mt]] - lp[pos, vocab[wt]])
                    score[i] = tot
            out = part[['id', 'code', 'mut_type', 'source']].copy()
            out['prosst'] = score
            out.to_csv(str(path) + '.tmp', index=False, compression='gzip')
            os.replace(str(path) + '.tmp', path)
            print(f'prosst {source}: chunk {k + 1}/{len(todo)} ({len(part)} rows) done, {(time.time() - t0) / 60:.1f} min elapsed', flush=True)
    if skipped:
        sys.exit(f'{len(skipped)} scaffold(s) skipped, their predictions are NaN: {skipped[:8]}')


if __name__ == '__main__':
    main()
