"""Two short GPU probes of ``--joint_doubles`` on the real model, one training run per mode on a truncated cache.

    python analysis_scripts/joint_doubles_probe.py memory --cache_path CACHE [--micro 32] [--modes off alternate both] [--extra FLAGS...]
    python analysis_scripts/joint_doubles_probe.py grad   --cache_path CACHE [--micro 16]

memory: the peak CUDA memory allocated over the run (``torch.cuda.max_memory_allocated``). Use a micro-batch big enough that training, not
        the one-off load of the base model (about 10.7 GB), sets the peak; a batch is one library and a short batch is dropped, so the
        probe keeps 48 singles and 48 doubles per library.
grad:   the gradient norm of each adapter at every step with the WT losses at 0 and --tf_mix 0.5, so a WT gradient can only come from the joint doubles:
        ``off`` must show exactly 0 for ``lora_wt``, ``both`` and ``alternate`` a positive one (the check that autocast's weight cache is
        not handing the adapter a graph-less copy after a ``no_grad`` read).
"""
import argparse
import os
import subprocess
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')          # smoke_cpu hides CUDA unless this is already set
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import smoke_cpu  # noqa: E402

LOG_EVERY_STEP = ("import lightning.pytorch as pl\n_i = pl.Trainer.__init__\n"
                  "def _init(self, *a, **k):\n    k['log_every_n_steps'] = 1\n    _i(self, *a, **k)\n"
                  "pl.Trainer.__init__ = _init\n")


def run(root, cache, version, mode, micro, extra, prelude, epilogue):
    argv = ['training.py', '--experiment_name', f'probe_{mode}', '--version', '0', '--log_dir', os.path.join(root, 'logs'), '--cache_path', cache,
            '--cache_version', version, '--num_epochs', '1', '--batch_size', str(micro), '--micro_batch_size', str(micro), '--num_workers', '0',
            '--no-comet', '--rank_list_size', '4', '--lr_warmup_steps', '2', '--skip_val', '--joint_doubles', mode] + extra
    # --skip_val leaves the checkpoint callback without its monitor, which raises after the last step: the run is over by then, so it is caught.
    code = (f"import runpy, sys, torch\n{prelude}sys.argv = {argv!r}\ntry:\n    runpy.run_module('esm_msr.training', run_name='__main__')\n"
            f"except BaseException as e:\n    print('EXC', type(e).__name__)\n{epilogue}")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, 'src'), HF_HUB_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    return subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['memory', 'grad'])
    ap.add_argument('--cache_path', required=True)
    ap.add_argument('--cache_version', default='c6')
    ap.add_argument('--micro', type=int, default=None)
    ap.add_argument('--modes', nargs='*', default=['off', 'alternate', 'both'])
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help="more training flags, e.g. --tf_mix 0.5 --mask_strategy_mt independent (last)")
    a = ap.parse_args()
    micro = a.micro or (32 if a.what == 'memory' else 16)
    root = tempfile.mkdtemp(prefix='joint_probe_')
    cache = os.path.join(root, 'cache')
    os.makedirs(cache)
    smoke_cpu._truncate.__defaults__ = (48, 48)
    smoke_cpu.make_cache(a.cache_path, a.cache_version, cache, n_train=3, n_val=1)
    for mode in a.modes:
        if a.what == 'memory':
            r = run(root, cache, a.cache_version, mode, micro, a.extra, '',
                    "print('PEAK_GB %.3f' % (torch.cuda.max_memory_allocated() / 2**30))\n")
            peak = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith('PEAK_GB')]
            print(f"{mode:10s} micro={micro} {peak[-1] if peak else 'NO PEAK LINE'}", flush=True)
            if not peak:
                print('\n'.join((r.stdout + r.stderr).splitlines()[-20:]), flush=True)
        else:
            import pandas as pd
            run(root, cache, a.cache_version, mode, micro, ['--lambda_reg_wt', '0', '--lambda_rank_wt', '0', '--lambda_rank_mt', '0', '--tf_mix', '0.5'] + a.extra, LOG_EVERY_STEP, '')
            csv = os.path.join(root, 'logs', f'probe_{mode}', '0', 'metrics.csv')
            if not os.path.exists(csv):
                print(mode, 'no metrics.csv', flush=True)
                continue
            df = pd.read_csv(csv)
            cols = [c for c in df.columns if c.startswith('norm_grad/lora') or c in ('train/L_reg_combined', 'train/L_reg_joint')]
            print(f"== {mode}\n{df[cols].dropna(how='all').head(8).to_string()}", flush=True)
    print(f"(scratch: {root})")


if __name__ == '__main__':
    main()
