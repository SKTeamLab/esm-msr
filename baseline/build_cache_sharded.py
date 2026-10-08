"""
Builds the missing library caches of a split with the released dataset class, one shard of the missing list per process, each file written
atomically (temp file + rename), so shards can run side by side and a kill never leaves a truncated cache.

Usage: PYTHONPATH=src python baseline/build_cache_sharded.py --split SPLIT --cache DIR --shard I --of N [--threads 2] [--scaffolds train val test]
Libraries are ordered by size and dealt round-robin to the shards.
"""
import argparse, os, sys, time, pickle, logging
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--split', required=True); ap.add_argument('--cache', required=True)
ap.add_argument('--shard', type=int, default=0); ap.add_argument('--of', type=int, default=1)
ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--scaffolds', nargs='+', default=['train', 'val', 'test'])
pa = ap.parse_args()
torch.set_num_threads(pa.threads)

from esm_msr import data as D
from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

def atomic_save(self):
    tmp = self.cache_path + f'.tmp{os.getpid()}'
    with open(tmp, 'wb') as f:
        pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, self.cache_path)
D.ProteinStructureMutationEpistasisDataset._save_data_to_cache = atomic_save

root = '/home/sareeves/software/esm-msr/data/tsuboyama/'
pp = MegaScaleDatasetPreprocessor(root + 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv', root + 'AlphaFold_model_PDBs', spurs_override=False)
logging.disable(logging.NOTSET)
pp.create_training_splits(pa.split, -1)
tok, enc = EsmSequenceTokenizer('cpu'), ESM3_structure_encoder_v0('cpu')
todo = []
for sc in pa.scaffolds:
    df = pp.split_dfs[sc]
    for code in df['code'].unique():
        if not os.path.exists(os.path.join(pa.cache, f'{code}_ddG_ML_MAX_Smasked0.pkl')):
            todo.append((int((df['code'] == code).sum()), code, sc))
todo.sort(reverse=True)
mine = todo[pa.shard::pa.of]
print(f'shard {pa.shard}/{pa.of}: {len(mine)} of {len(todo)} missing libraries', flush=True)
for n, code, sc in mine:
    t = time.time()
    df = pp.split_dfs[sc]; df = df.loc[df['code'] == code]
    D.ProteinStructureMutationEpistasisDataset(
        dms_df=df, tokenizer=tok, structure_encoder=enc, dms_name=code, path=pa.cache, score_name='ddG_ML', generate=False,
        mut_structs_root='/home/sareeves/software/esm-msr/data/tsuboyama/FINAL_results/', incl_destab_bb=True,
        incl_singles=True, incl_doubles=True, incl_reversions=True, incl_mut_ctx=True)
    print(f'built {code} ({sc}, {n} rows) in {time.time() - t:.0f}s', flush=True)
print('SHARD_DONE', flush=True)
