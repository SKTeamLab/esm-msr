#!/usr/bin/env python3
"""ESM-MSR inference on the 12 external thermostability sets with ColabFold AF2 vs. crystal structures.

Generalizes the Oct 8 `msr_external_af2.py` harness for any ESM-MSR checkpoint (including the
authoritative `FINAL_p0` triplicate checkpoints) while mirroring `inference_scripts/esm_msr_testing.py`
model construction, calibration (`tie_mt_scale=False, per_mutation_bias=True`), and auto-batching.

Key safeguards:
  - Strict per-group indexing guard: asserts `get_seq(af2_pdb, 'A') == get_seq(crystal_pdb, chain)`
    and identical `mut_type_renumbered` arrays across all rows before running inference.
  - Explicit `--plddt_mode {zero,one}` control: checkpoints trained before commit `95f6b42`
    (including `FINAL_p0_s{1,2,3}` from Nov 11-13) were trained and benchmarked in `final_bench`
    with `per_res_plddt=None` (ESM3.forward all-zero default), whereas newer checkpoints trained
    after `95f6b42` (`FINAL_p0_plddt1_*`) use `per_res_plddt=1.0` for finite coordinates.
  - Atomic `.tmp` -> rename CSV writes so resumed runs skip only complete datasets cleanly.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock

# Optional structure-repair deps not used on this inference path
for _m in (
    'pdbfixer', 'openmm', 'openmm.app', 'openmm.unit',
    'simtk', 'simtk.openmm', 'simtk.openmm.app', 'simtk.unit',
):
    sys.modules.setdefault(_m, MagicMock())

warnings.filterwarnings('ignore')

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

EXTERNAL = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo_dir', default='/home/sreeves/esm-msr-devel', help='ESM-MSR repository root on cluster')
    ap.add_argument('--checkpoint', required=True, help='Path to .ckpt file (absolute or relative to LoRA_models/)')
    ap.add_argument('--out_dir', required=True, help='Output root directory (creates <out_dir>/<mode>/{predictions,stats})')
    ap.add_argument('--structures', default='af2,crystal', help='Comma-separated list of modes: af2,crystal')
    ap.add_argument('--datasets', default='external', help='Comma-separated datasets or "external" for all 12')
    ap.add_argument('--af2_dir', default='/home/sreeves/stability-baselines/structures/af2_external')
    ap.add_argument('--structures_dir', default=None, help='Crystal PDB dir (default: <repo_dir>/data/structures)')
    ap.add_argument('--sigma', type=float, default=1.0, help='LoRA alpha scaling factor (default 1.0)')
    ap.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    ap.add_argument('--plddt_mode', choices=('zero', 'one'), default='zero',
                    help='"zero" matches pre-95f6b42 FINAL_p0_s{1,2,3} training & final_bench; "one" uses per_res_plddt=1.0')
    ap.add_argument('--batch_mode', choices=('auto', 'fixed'), default='auto')
    ap.add_argument('--batch_size', type=int, default=1, help='Fixed batch size or starting batch size')
    ap.add_argument('--auto_batch_headroom', type=float, default=0.80)
    ap.add_argument('--limit_groups', type=int, default=None, help='Optional cap on groups per dataset for smoke tests')
    ap.add_argument('--dry_run', action='store_true', help='CPU-only: verify checkpoint, hparams, and AF2/crystal indexing')
    return ap.parse_args()


def _install_plddt_zero_hook(model) -> None:
    """Restore exact pre-95f6b42 ESM3 forward call (`per_res_plddt` omitted -> ESM3 all-zero default)."""
    def _get_esm3_outputs_zero(
        sequence_tokens,
        structure_coords=None,
        structure_tokens=None,
        per_res_plddt=None,
        active_model=None,
    ):
        def _prep(tensor, expected_dims):
            if tensor is None or not torch.is_tensor(tensor):
                return None
            if tensor.dim() == expected_dims - 1:
                tensor = tensor.unsqueeze(0)
            while tensor.dim() > expected_dims and tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            return tensor

        target_model = active_model if active_model is not None else model.model
        return target_model.model(
            sequence_tokens=_prep(sequence_tokens, 2),
            structure_coords=_prep(structure_coords, 4),
            structure_tokens=_prep(structure_tokens, 2),
        )

    model._get_esm3_outputs = _get_esm3_outputs_zero


def _atomic_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)


def main() -> None:
    a = parse_args()
    repo_dir = Path(a.repo_dir).resolve()
    sys.path.insert(0, str(repo_dir / 'src'))
    sys.path.insert(0, str(repo_dir / 'inference_scripts'))

    from esm_msr import auto_batch, inference, models, utils  # noqa: E402
    from esm_msr.preprocess import get_seq  # noqa: E402
    import esm_msr_testing as T  # noqa: E402

    data_dir = repo_dir / 'data' / 'preprocessed'
    structures_dir = Path(a.structures_dir) if a.structures_dir else (repo_dir / 'data' / 'structures')
    af2_dir = Path(a.af2_dir)
    out_root = Path(a.out_dir)

    resolved_ckpt = Path(a.checkpoint).resolve() if os.path.isfile(a.checkpoint) else (repo_dir / 'LoRA_models' / a.checkpoint)
    if not resolved_ckpt.is_file():
        sys.exit(f'ERROR: checkpoint not found: {resolved_ckpt}')
    hparams_path = resolved_ckpt.parent / 'hparams.yaml'
    if not hparams_path.is_file():
        sys.exit(f'ERROR: hparams.yaml not found next to checkpoint: {hparams_path}')

    parsed_config = inference.parse_hparams_to_lora_config(str(hparams_path), sigma=a.sigma)
    adapter_mode = parsed_config.get('adapter_mode', 'dual')
    lora_mode = parsed_config.get('lora_mode', 'ensemble')
    cal = inference.resolve_calibration(parsed_config.get('calibration'), None, None)
    mask_strategy = parsed_config.get('mask_strategy', None)
    native_cond_for_mt = bool(parsed_config.get('native_cond_for_mt', False))
    mask_native_cond_struct = bool(parsed_config.get('mask_native_cond_struct', False))
    fused_antisymmetric = bool(parsed_config.get('fused_antisymmetric', False))
    lora_config = {'wt_config': parsed_config['wt_config'], 'mt_config': parsed_config['mt_config']}

    print(
        f'[CONFIG] ckpt={resolved_ckpt}\n'
        f'         adapter_mode={adapter_mode} lora_mode={lora_mode} sigma={a.sigma} dtype={a.dtype}\n'
        f'         calibration={cal} mask_strategy={mask_strategy} plddt_mode={a.plddt_mode} batch_mode={a.batch_mode}',
        flush=True,
    )

    # Map benchmark group (<code+chain>) -> ColabFold relaxed rank-1 PDB
    man = pd.read_csv(af2_dir / 'manifest.csv')
    g2af: dict[str, str] = {}
    for r in man.itertuples():
        for g in str(r.groups).split(';'):
            g2af[g] = str(r.af2_id)

    def af2_pdb(group: str) -> str | None:
        af = g2af.get(group)
        if not af:
            return None
        hits = sorted(glob.glob(str(af2_dir / 'predictions' / f'{af}_relaxed_rank_001_*.pdb')))
        return hits[0] if hits else None

    def load_dataset(name: str) -> pd.DataFrame:
        df = pd.read_csv(data_dir / f'{name}_mapped.csv')
        df['pdb_file'] = df['pdb_file'].str.replace(
            '/home/sareeves/software/esm-msr/data/structures', str(structures_dir), regex=False
        )
        if name in {'s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 's571', 's783', 's2648', 's8754'}:
            df = df.reset_index()
            df['position_pdb'] = df['position']
            df['position'] = df['seq_pos']
            df['mut_type'] = df['wild_type'] + df['position'].astype(int).astype(str) + df['mutation']
            df['id'] = df['code'] + df['chain'] + '_' + df['mut_type']
            df = df.set_index('id')
            if 'dTm' in df.columns:
                df['ddG'] = df['dTm']
        else:
            df = df.reset_index()
            df = utils.sort_mutations_by_position(df, 'mut_info_seq_pos', 'mut_type')
            df['id'] = df['code'] + df['chain'] + '_' + df['mut_type']
            df = df.set_index('id')
            df = utils.parse_multimutant_column(df, 'mut_type', max_mutations=10)
        return df

    names = EXTERNAL if a.datasets == 'external' else [s.strip() for s in a.datasets.split(',') if s.strip()]
    modes = [s.strip() for s in a.structures.split(',') if s.strip()]

    if 'af2' in modes:
        need = set()
        for n in names:
            d = load_dataset(n)
            for (_pdb, code, chain), _ in d.groupby(['pdb_file', 'code', 'chain']):
                need.add(f'{code}{chain}')
        miss = sorted(g for g in need if af2_pdb(g) is None)
        print(f'[AF2] structures available: {len(need) - len(miss)}/{len(need)} groups', flush=True)
        if miss:
            sys.exit(f'ERROR: missing AF2 structures for {len(miss)} groups: {miss[:20]}')

    model = None
    sizer = None
    if not a.dry_run:
        t0 = time.time()
        model_dtype = torch.bfloat16 if a.dtype == 'bf16' else torch.float32
        model = models.MSRModel(
            lora_config=lora_config,
            shared_scale_init=1,
            shared_bias_init=0,
            adapter_mode=adapter_mode,
            lora_mode=lora_mode,
            model_dtype=model_dtype,
            inference_mode=True,
            tie_mt_scale=cal['tie_mt_scale'],
            per_mutation_bias=cal['per_mutation_bias'],
            native_cond_mt_input=native_cond_for_mt,
            mask_native_cond_struct=mask_native_cond_struct,
            fused_antisymmetric=fused_antisymmetric,
        ).to('cuda:0')
        model.load_lora_weights(str(resolved_ckpt), load_wt_only=False)
        if a.plddt_mode == 'zero':
            _install_plddt_zero_hook(model)
            print('[MODEL] installed pre-95f6b42 per_res_plddt=None hook (matches FINAL_p0 training & final_bench)', flush=True)
        else:
            print('[MODEL] using default per_res_plddt=1.0 for finite coordinates', flush=True)
        model.eval()
        if a.batch_mode == 'auto':
            sizer = auto_batch.AutoBatchSizer(
                device=next(model.parameters()).device,
                headroom=a.auto_batch_headroom,
                log=lambda msg: print(f'  {msg}', flush=True),
            )
        print(f'[MODEL] loaded in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)}', flush=True)

    meta = dict(
        checkpoint=str(resolved_ckpt),
        hparams=str(hparams_path),
        sigma=a.sigma,
        dtype=a.dtype,
        plddt_mode=a.plddt_mode,
        mask_strategy=mask_strategy,
        calibration=cal,
        adapter_mode=adapter_mode,
        lora_mode=lora_mode,
        batch_mode=a.batch_mode,
        af2_dir=str(af2_dir),
        structures_dir=str(structures_dir),
        host=socket.gethostname(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
    )
    if not a.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / 'run_meta.json').write_text(json.dumps(meta, indent=2) + '\n')

    def run_mode(mode: str) -> None:
        out = out_root / mode
        if not a.dry_run:
            (out / 'predictions').mkdir(parents=True, exist_ok=True)
            (out / 'stats').mkdir(parents=True, exist_ok=True)
            tf = out / 'timing.csv'
            if not tf.exists():
                tf.write_text('dataset,n_rows,n_groups,wall_s,ms_per_row,host,gpu,slurm_job\n')
        stats_wt = pd.DataFrame()
        stats_mt = pd.DataFrame()
        stats_cmb = pd.DataFrame()
        skipped = []

        for name in names:
            res_f = out / 'predictions' / f'{name}.csv'
            if res_f.exists() and not a.dry_run:
                print(f'[{mode}:{name}] exists, loading for stats and skipping inference', flush=True)
                res_df = pd.read_csv(res_f, index_col='id')
                stats_wt = T.update_stats(stats_wt, name, res_df, 'ddG', 'wt_lora_pred', 'dddG', 'wt_lora_dddg_pred')
                stats_mt = T.update_stats(stats_mt, name, res_df, 'ddG', 'mt_lora_pred', 'dddG', 'mt_lora_dddg_pred')
                stats_cmb = T.update_stats(stats_cmb, name, res_df, 'ddG', 'combined_pred', 'dddG', 'combined_dddg_pred')
                continue

            df_true = load_dataset(name)
            res_combined = []
            total_time = 0.0
            t_ds = time.time()
            ngr = 0
            groups = list(df_true.groupby(['pdb_file', 'code', 'chain']))
            if a.limit_groups:
                groups = groups[:a.limit_groups]

            for (pdb, code, chain), data in tqdm(groups, desc=f'{mode}:{name}', mininterval=30):
                unique_data = data[~data.index.duplicated(keep='first')]
                std_crystal = inference.standardize_input_df(unique_data, quiet=True)
                if mode == 'af2':
                    p = af2_pdb(f'{code}{chain}')
                    if p is None:
                        skipped.append((name, f'{code}{chain}', 'no AF2 structure'))
                        continue
                    ud = unique_data.copy()
                    ud['pdb_file'] = p
                    ud['chain'] = 'A'
                    try:
                        input_data = inference.standardize_input_df(ud, quiet=True)
                        same_seq = get_seq(p, 'A') == get_seq(pdb, chain)
                        same_idx = (input_data['mut_type_renumbered'].values == std_crystal['mut_type_renumbered'].values).all()
                    except Exception as e:
                        skipped.append((name, f'{code}{chain}', f'standardize failed: {str(e)[:100]}'))
                        continue
                    if not (same_seq and same_idx):
                        skipped.append((name, f'{code}{chain}', f'INDEXING MISMATCH same_seq={same_seq} same_idx={same_idx}'))
                        continue
                else:
                    input_data = std_crystal

                ngr += 1
                if a.dry_run:
                    continue

                pred_df, t_inf = T.timed_call(
                    inference.infer_mutants,
                    model=model,
                    df=input_data,
                    batch_size=a.batch_size,
                    quiet=True,
                    mask_strategy=mask_strategy,
                    optimize_wt_pass=True,
                    skip_reverse=False,
                    auto_batch=sizer,
                )
                pred_df['id'] = code + chain + '_' + pred_df['mut_type_renumbered']
                pred_df = pred_df.set_index('id')
                overlap = list(set(data.columns).intersection(set(pred_df.columns)))
                res_combined.append(data.join(pred_df.drop(overlap, axis=1)))
                total_time += t_inf
                torch.cuda.empty_cache()

            if a.dry_run:
                print(f'[{mode}:{name}] dry-run groups OK: {ngr}/{len(groups)}', flush=True)
                continue
            if not res_combined:
                raise RuntimeError(f'[{mode}:{name}] produced 0 rows!')

            res_df = pd.concat(res_combined)
            if len(res_df) != len(df_true):
                raise RuntimeError(f'[{mode}:{name}] row count mismatch: {len(res_df)} vs {len(df_true)}')
            _atomic_to_csv(res_df, res_f)

            wall = time.time() - t_ds
            with open(tf, 'a') as fh:
                fh.write(
                    f'{name},{len(res_df)},{ngr},{wall:.1f},{1000 * wall / len(res_df):.2f},'
                    f'{socket.gethostname()},{torch.cuda.get_device_name(0)},{os.environ.get("SLURM_JOB_ID")}\n'
                )
            stats_wt = T.update_stats(stats_wt, name, res_df, 'ddG', 'wt_lora_pred', 'dddG', 'wt_lora_dddg_pred', total_time)
            stats_mt = T.update_stats(stats_mt, name, res_df, 'ddG', 'mt_lora_pred', 'dddG', 'mt_lora_dddg_pred', total_time)
            stats_cmb = T.update_stats(stats_cmb, name, res_df, 'ddG', 'combined_pred', 'dddG', 'combined_dddg_pred', total_time)
            _atomic_to_csv(stats_wt, out / 'stats' / 'WT_LoRA.csv', na_rep='', float_format='%.6f')
            _atomic_to_csv(stats_mt, out / 'stats' / 'MT_LoRA.csv', na_rep='', float_format='%.6f')
            _atomic_to_csv(stats_cmb, out / 'stats' / 'Combined.csv', na_rep='', float_format='%.6f')
            wt_rho = stats_wt.at[name, 'spearman_all']
            cb_rho = stats_cmb.at[name, 'spearman_all']
            print(
                f'[{mode}:{name}] done: {len(res_df)} rows, {ngr} groups, {wall:.0f}s; '
                f'wt_rho={wt_rho:.4f} comb_rho={cb_rho:.4f}',
                flush=True,
            )
            torch.cuda.empty_cache()

        if skipped:
            if not a.dry_run:
                _atomic_to_csv(pd.DataFrame(skipped, columns=['dataset', 'group', 'reason']), out / 'skipped_groups.csv', index=False)
            sys.exit(f'ERROR: [{mode}] skipped {len(skipped)} groups: {skipped[:5]}')

    for m in modes:
        run_mode(m)
    print('[DONE] finished cleanly', flush=True)


if __name__ == '__main__':
    main()
