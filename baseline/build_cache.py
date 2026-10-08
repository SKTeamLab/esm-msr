"""
Builds the dataset caches the released code needs for a split, with the released code (MutationStabilityDataset generates a library's items
when its cache file is missing), into a cache directory that already holds symlinks to the old cache.

Usage: PYTHONPATH=src python baseline/build_cache.py --split data/splits_oct06_capped.pkl --cache DIR [--threads 6]
The recipe flags are baseline/recipe_args.txt, so the caches are exactly those the baseline run will look for.
"""
import argparse, os, shlex, sys, threading, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--split', required=True)
ap.add_argument('--cache', required=True)
ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--rss_cap_gb', type=float, default=26.0)
ap.add_argument('--benchmarks', default='/home/sareeves/playground/esm-msr-devel/repo/data/preprocessed')
pa = ap.parse_args()
torch.set_num_threads(pa.threads)

def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2 ** 30
def watchdog():
    while True:
        if rss() > pa.rss_cap_gb:
            print(f'WATCHDOG: RSS {rss():.1f} GB over the cap, exiting', flush=True); os._exit(3)
        time.sleep(2)
threading.Thread(target=watchdog, daemon=True).start()

HERE = os.path.dirname(os.path.abspath(__file__))
D = '/home/sareeves/software/esm-msr/data/tsuboyama/'
recipe = shlex.split(open(os.path.join(HERE, 'recipe_args.txt')).read())
sys.argv = ['build', '--experiment_name', 'build', '--version', '0', '--raw_data_file', D + 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv',
            '--af_model_folder', D + 'AlphaFold_model_PDBs', '--split_file', pa.split, '--cache_path', pa.cache, '--benchmark_data_path', pa.benchmarks,
            '--num_workers', '0', '--log_dir', '/tmp/baseline_build_logs', '--checkpoint_path', '/tmp/baseline_build_ckpt', '--seed', '1'] + recipe
from esm_msr.config import parse_arguments
from esm_msr.preprocess_megascale import setup_dataloaders
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

args = parse_arguments()
tok, enc = EsmSequenceTokenizer('cpu'), ESM3_structure_encoder_v0('cpu')
t = time.time()
tr, va, be, tn, vn, bn = setup_dataloaders(args, tok, enc, add_benchmarks_to_val=True)
print(f'BUILD_DONE {time.time() - t:.0f}s: train loaders {len(tr)}, val loaders {len(va)} ({vn}), benchmarks {bn}', flush=True)
