"""
Builds (and only builds) the dataset caches a run would need, with the structure encoder on a chosen device, then exits.

Usage: PYTHONPATH=src python scripts/build_cache.py [--device cpu|cuda] [--threads 4] [-- extra training flags, e.g. --premask_mt_structure]
Takes the flags of scripts/run_arm.sh (split, cache path, ...) so that the cache it writes is exactly the one that run would look for. Used for the
caches that need many structure re-encodes (--premask_mt_structure), which are best done ahead of the run, on the GPU when it is free.
"""
import argparse, os, re, shlex, sys, threading, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cpu')
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--rss_cap_gb', type=float, default=24.0)
pa, extra = ap.parse_known_args()
torch.set_num_threads(pa.threads)

def _rss_gb():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2 ** 30
def _watchdog():
    while True:
        if _rss_gb() > pa.rss_cap_gb:
            print(f'WATCHDOG: RSS {_rss_gb():.1f} GB over the cap, exiting', flush=True); os._exit(3)
        time.sleep(2)
threading.Thread(target=_watchdog, daemon=True).start()

HERE = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(HERE)
txt = open(os.path.join(HERE, 'run_arm.sh')).read().split('training.py', 1)[1].replace('\\\n', ' ')
line = [l for l in txt.split('\n') if l.strip()][0]
line = (line.replace('$WT', WT).replace('${SPLIT:-' + WT + '/data/splits_oct06_capped.pkl}', WT + '/data/splits_oct06_capped.pkl')
        .replace('${COMET_PROJECT:-esm-msr-agent-oct06}', 'build').replace('$NAME', 'build').replace('$EPOCHS', '1').replace('$SEED', '1'))
line = re.sub(r'--comet_api_key.*$', '', line)
line = line.replace('--cache_path cache_v7', '--cache_path /home/sareeves/playground/esm-msr-devel/cache_v7')
sys.argv = ['build'] + shlex.split(line) + ['--include_out_of_range', '--num_workers', '0', '--log_dir', '/tmp/build_logs', '--checkpoint_path', '/tmp/build_ckpt'] + [a for a in extra if a != '--']

from esm_msr import training
from esm_msr.config import parse_arguments
args = parse_arguments()
tok = training.EsmSequenceTokenizer('cpu')
enc = training.ESM3_structure_encoder_v0(pa.device)
t = time.time()
tr, va, be, tn, vn, bn = training.setup_dataloaders(args, tok, enc, add_benchmarks_to_val=True)
n_tr = sum(len(l.dataset) for l in tr if hasattr(l, 'dataset'))
n_va = sum(len(l.dataset) for l in va if hasattr(l, 'dataset'))
print(f'BUILD_DONE {time.time() - t:.0f}s: {len(tr)} train loaders ({n_tr} items), {len(va)} validation loaders ({n_va} items); premask={getattr(args, "premask_mt_structure", False)}', flush=True)
