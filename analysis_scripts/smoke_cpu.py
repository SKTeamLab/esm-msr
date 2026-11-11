"""End-to-end smoke of ``training.main()`` on CPU, on a handful of truncated real libraries from a cache.

Builds a tiny cache (a few libraries cut to a few singles and doubles, plus a manifest) in a temp directory and runs the real entry point
for each mode: model construction, the cache loader, one epoch of block training with the real backbone, a real validation (both
passes, the epistasis metrics, the dump), checkpointing, the cosine schedule. It cannot say whether a recipe is any good; it says that the
code path runs and every loss is finite. CUDA is hidden, so it cannot contend with a live job (it does compete for CPU: run it niced).

    python analysis_scripts/smoke_cpu.py --cache_path CACHE --cache_version c6 [--modes blocks blocks_domain_rank ...]
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
# summary columns a mode must have written (and the ones no mode may write any more)
EXPECT = {}
CULLED = ('val_cycle_gap', 'val_rho_ungrouped', 'val_diag_', 'val_rho_singles_combined', 'val_rho_doubles_wt')
# The one truncated validation library is too small for the epistasis metrics, so the modes watch rho_combined (the main, masked regime). A CPU run cannot hold the unmasked block shape
# (host memory), so the unmasked recipe is smoke-tested on the cluster, not here.
MODES = {
    'blocks': ['--block_positions', '4'],
    # the T objective (Tobit likelihood, posterior teacher-forcing label, no rank loss) read through the Tobit expected reading: columns val_rmse_* on the observed scale
    'blocks_tobit_posterior': ['--block_positions', '4', '--tf_label', 'posterior', '--lambda_rank_wt', '0', '--lambda_rank_mt', '0'],
    # the recipe's moving parts on a masked read: domain steps, list rows, pair weight, rank 0.5 : reg 15, the rank-tie threshold, the wide floor
    'blocks_domain_rank': ['--block_positions', '4', '--block_domain_steps', '--block_pair_weight', '0.38', '--block_list_rows', '4', '--tf_label', 'posterior',
                           '--lambda_rank_wt', '0.5', '--lambda_rank_mt', '0.5', '--lambda_reg_wt', '15', '--lambda_reg_mt', '15', '--rank_censor_threshold', '-1.0',
                           '--tobit_mu_f', '-0.14', '--tobit_sigma_f', '0.5'],
    'blocks_cond_chunked': ['--block_positions', '4', '--block_row_chunk', '10', '--lambda_cond_mt', '7.5'],
    'exclude_mutbg': ['--block_positions', '4', '--exclude_mutbg'],
    'comet_offline': ['--block_positions', '4', '--comet', '--comet_offline'],      # the key lookup and the logger path, without uploading anything
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
        cmd = [sys.executable, '-m', 'esm_msr.training', '--experiment_name', f'smoke_{mode}', '--version', '0', '--log_dir', os.path.join(root, 'logs'),
               '--cache_path', cache, '--cache_version', a.cache_version, '--num_epochs', '1',
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
            missing = [c for c in EXPECT.get(mode, []) if c not in df.columns or not df[c].notna().any()]
            culled = [c for c in df.columns if any(c.startswith(g) for g in CULLED)]
            ok = ok and not missing and not culled
            print(f"[{mode}] missing {missing or 'none'}, culled columns still logged {culled[:3] or 'none'}, validation columns {len([c for c in df.columns if c.startswith('val_')])}", flush=True)
            print(f"[{mode}] validations {len(rho)}, val_rho_combined_avg {['%.4f' % r for r in rho]}", flush=True)
        if not ok:
            failed.append(mode)
            print(f"[{mode}] FAILED rc={r.returncode}\n" + '\n'.join(log.splitlines()[-25:]), flush=True)
        else:
            print(f"[{mode}] ok", flush=True)
    print(f"failed: {failed or 'none'}")
    if not a.keep and not failed:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
