"""End-to-end smoke of ``training.main()`` on CPU, on a handful of truncated real libraries from a cache.

Builds a tiny cache (a few libraries cut to a few singles and doubles, plus a manifest) in a temp directory and runs the real entry point
for each mode: model construction, the cache loader, one epoch of the streaming loss with the real backbone, a real validation (all four
passes, the epistasis metrics, the dump), checkpointing, the plateau hooks. It cannot say whether a recipe is any good; it says that the
code path runs and every loss is finite. CUDA is hidden, so it cannot contend with a live job (it does compete for CPU: run it niced).

    python analysis_scripts/smoke_cpu.py --cache_path CACHE --cache_version c6 [--modes dual_joint fused ...] [--wt_ckpt WT.ckpt]
"""
import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

NATIVE = re.compile(r'_[A-Z]\d+[A-Z]$')
MODES = {
    'dual_joint': [],
    'frozen_wt': ['--frozen_wt_mt_only'],
    'fused': ['--adapter_mode', 'fused'],
    'mask_mt_independent': ['--frozen_wt_mt_only', '--mask_strategy_mt', 'independent'],
    'mask_wt_marginal_mt_independent': ['--mask_strategy_wt', 'marginal', '--mask_strategy_mt', 'independent'],
    'native_cond_for_mt': ['--frozen_wt_mt_only', '--native_cond_for_mt'],
    'huber': ['--frozen_wt_mt_only', '--huber_delta', '1.0'],
    'wt_only': ['--freeze_mt_adapter', '--val_wt_only', '--subsets', 'single', '--lambda_rank_mt', '0', '--lambda_reg_mt', '0'],
}


def _truncate(items, n_single=12, n_double=12):
    out, c = [], {'single': 0, 'double': 0}
    for it in items:
        t = it.get('subset_type')
        if t in c and c[t] < (n_single if t == 'single' else n_double):
            out.append(it)
            c[t] += 1
    return out


def make_cache(src, version, dst, n_train=2, n_val=1):
    with open(os.path.join(src, f'manifest_{version}.json')) as f:
        man = json.load(f)
    def usable(code):
        with open(os.path.join(src, f'{code}_ddG_ML_{version}.pkl'), 'rb') as f:
            items = pickle.load(f)['items']
        t = [i['subset_type'] for i in items]
        return t.count('single') >= 8 and t.count('double') >= 8
    train = [c for c in man['train'] if not NATIVE.search(c)]
    train = [c for c in train if usable(c)][:n_train]
    native = [c for c in man['train'] if NATIVE.search(c)][:1]
    val = [c for c in man['val'] if not NATIVE.search(c) and usable(c)][:n_val]
    for code in train + native + val:
        with open(os.path.join(src, f'{code}_ddG_ML_{version}.pkl'), 'rb') as f:
            blob = pickle.load(f)
        blob['items'] = _truncate(blob['items'])
        with open(os.path.join(dst, f'{code}_ddG_ML_{version}.pkl'), 'wb') as f:
            pickle.dump(blob, f)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from esm_msr.preprocess_megascale import write_manifest
    write_manifest(dst, version, {'train': train + native, 'val': val}, [], man.get('split_file'), note='smoke')
    return train, native, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_path', required=True)
    ap.add_argument('--cache_version', default='c6')
    ap.add_argument('--modes', nargs='*', default=list(MODES))
    ap.add_argument('--wt_ckpt', default=None, help="a WT checkpoint (rank 2 / alpha 4) for the frozen-WT modes")
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    root = tempfile.mkdtemp(prefix='smoke_')
    cache = os.path.join(root, 'cache')
    os.makedirs(cache)
    train, native, val = make_cache(a.cache_path, a.cache_version, cache)
    print(f"smoke cache {cache}: train {train} native {native} val {val}", flush=True)
    failed = []
    for mode in a.modes:
        extra = list(MODES[mode])
        if '--frozen_wt_mt_only' in extra:
            if not a.wt_ckpt:
                print(f"[{mode}] skipped: needs --wt_ckpt")
                continue
            extra += ['--init_wt_from_checkpoint', a.wt_ckpt]
        cmd = [sys.executable, '-m', 'esm_msr.training', '--experiment_name', f'smoke_{mode}', '--version', '0', '--log_dir', os.path.join(root, 'logs'),
               '--cache_path', cache, '--cache_version', a.cache_version, '--num_epochs', '1', '--batch_size', '8', '--micro_batch_size', '4',
               '--num_workers', '0', '--no-comet', '--rank_list_size', '4', '--lr_warmup_steps', '2'] + extra
        env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'), HF_HUB_OFFLINE='1', ESM_MSR_PRECISION='32')   # CPU bf16 autocast fails inside ESM3's geometric attention; production is bf16-mixed on GPU
        print(f"[{mode}] {' '.join(extra) or '(defaults)'}", flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log = r.stdout + r.stderr
        csv = os.path.join(root, 'logs', f'smoke_{mode}', '0', 'metrics.csv')
        ok = r.returncode == 0 and os.path.exists(csv) and 'Traceback' not in log and 'Failed to initialize' not in log
        # Training raises on a non-finite loss (manual_backward is guarded), so a clean finish means every loss was finite. The run is too short for
        # Lightning's every-25-steps train logging; what is checked is that validation ran twice (zero-shot and after the epoch) and a checkpoint was written.
        if ok:
            import numpy as np
            import pandas as pd
            df = pd.read_csv(csv)
            rho = df['val_rho_combined_avg'].dropna().tolist() if 'val_rho_combined_avg' in df.columns else []
            ok = len(rho) >= 2 and bool(np.isfinite(rho).all()) and os.path.exists(os.path.join(root, 'logs', f'smoke_{mode}', '0', 'last.ckpt'))
            print(f"[{mode}] validations {len(rho)}, val_rho_combined_avg {['%.4f' % r for r in rho]}", flush=True)
        if not ok:
            failed.append(mode)
            print(f"[{mode}] FAILED rc={r.returncode}\n" + '\n'.join(log.splitlines()[-25:]), flush=True)
        else:
            print(f"[{mode}] ok", flush=True)
    print(f"failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
