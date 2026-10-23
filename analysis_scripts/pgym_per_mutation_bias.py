"""Correct the stored ESM-MSR ProteinGym predictions for the per-mutation calibration bias (decision of 2026-10-23).

WHAT CHANGED. A calibration head outputs  y = scale * raw + bias        (one bias per item)      -- how the released and the dc8
checkpoints were trained and, until 2026-10-23, scored -- and now      y = scale * raw + bias * n_mut   (bias per mutated position).
`raw` (the summed log-likelihood ratio, which carries the LoRA and its epsilon) and `scale` are untouched, so for an item with
k = n_mut mutated positions each head's calibrated prediction moves by exactly
        d_head = bias_head * (k - 1)        (singles do not move)
and every other stored column is a fixed linear function of the heads (esm_msr.models.combine_predictions, mode 'mean', and
esm_msr.inference.infer_mutants with skip_additive):
        wt_lora_pred          += bias_wt * (k - 1)
        mt_lora_pred          += bias_mt * (k - 1)
        combined_pred         += 0.5 * (d_wt + d_mt)
        combined_dddg_pred    += 0.5 * (d_mt - d_wt)         (stored as 0.5*mt - 0.5*wt when additive scoring is skipped)
        epistasis_pred        += 0.5 * (d_mt - d_wt)         (stored as combined - wt; released full run only)
A WT-only run (--skip_reverse) stores wt_lora_pred only. The MT-pass ablation (`*_mtpass_wt`) runs the MT pass through the WT adapter
AND the WT calibration head (esm_msr_testing --mt_pass_adapter wt), so there d_mt = d_wt = bias_wt * (k - 1).
The biases are read from each checkpoint's state dict (`*calibration_head_wt.bias`, `*calibration_head_mt.bias`); epsilon only scales
the adapter that produces `raw`, so the same shift holds at every epsilon.

SUBCOMMANDS (all take --root, default the cluster's outputs/pgym):
  check       read-only. Biases, the identities above on the OLD files, mutation counts from two columns, and a reproduction of every
              stored per-DMS Spearman from the old predictions. Must pass before anything is written.
  backup      copy every run directory to <root>_pre_per_mutation_bias, verifying a sha256 per file.
  apply       needs a verified backup. Edits only the shifted rows' text of the shifted columns, keeps every other byte, resumes safely
              (per-file hashes in PER_MUTATION_BIAS_CORRECTION.json), refuses to apply twice, updates summary.csv (keeping the old
              value as spearman_combined_per_item_bias).
  aggregates  rebuilds the ESM-MSR columns of pgym_dc8_per_dms.csv and pgym_all_models_per_dms.csv from the corrected summaries,
              after reproducing the existing values from the backed-up summaries.
  compare     checks a real rerun (outputs/pgym_verify_pmb) against OLD + shift.

Run on the cluster under msr_venv.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

ROOT = '/home/sreeves/esm-msr-devel/outputs/pgym'
MODELS = '/home/sreeves/esm-msr-devel/LoRA_models'
REL = 'esm-msr-baseline-w2a4-m16a16-capped'
DEV = [f'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{i}' for i in (1, 2, 3)]
CKPT = {REL: 'epoch=07-val_rho_combined_avg=0.839.ckpt',
        DEV[0]: 'epoch=04-val_rho_combined_avg=0.840.ckpt',
        DEV[1]: 'epoch=07-val_rho_combined_avg=0.839.ckpt',
        DEV[2]: 'epoch=05-val_rho_combined_avg=0.841.ckpt'}
MANIFEST = 'PER_MUTATION_BIAS_CORRECTION.json'
TOL_IDENTITY = 1e-5          # the stored heads are float32 values printed as float64
TOL_SPEARMAN = 1e-9


def safe_spearman(preds, targets):
    """inference_scripts/pgym_runner.safe_spearman, verbatim in behaviour (the rule that produced summary.csv)."""
    preds, targets = np.asarray(preds, dtype=float), np.asarray(targets, dtype=float)
    ok = ~np.isnan(preds)
    preds, targets = preds[ok], targets[ok]
    if len(preds) < 2 or len(np.unique(targets)) < 2:
        return np.nan
    if len(np.unique(preds)) < 2:
        return np.nan
    rho = spearmanr(preds, targets)[0]
    return np.nan if np.isnan(rho) else float(rho)


def kind_of(sub):
    if sub.endswith('_wt_only'):
        return 'wt_only'
    if sub.endswith('_mtpass_wt'):
        return 'mtpass_wt'
    return 'full'


def pred_col(kind):
    return 'wt_lora_pred' if kind == 'wt_only' else 'combined_pred'


def biases(ckpt_dir):
    """(bias_wt, bias_mt) as float32 values from the checkpoint, plus the combined mode from its hparams."""
    sd = torch.load(f'{MODELS}/{ckpt_dir}/{CKPT[ckpt_dir]}', map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    out = {}
    for h in ('wt', 'mt'):
        ks = [k for k in sd if k.endswith(f'calibration_head_{h}.bias')]
        assert len(ks) == 1, (ckpt_dir, h, ks)
        out[h] = float(sd[ks[0]].float())
    hp = yaml.safe_load(open(f'{MODELS}/{ckpt_dir}/hparams.yaml'))
    mode = hp.get('combined_mode') or 'mean'
    assert mode == 'mean', f'{ckpt_dir}: combined_mode {mode!r}; the formulas here are for mean'
    assert not hp.get('per_mutation_bias'), f'{ckpt_dir} was TRAINED per mutation; its stored predictions would not need this correction'
    return out['wt'], out['mt']


def deltas(kind, k, bw, bm):
    """Shift of every stored column for items with k mutated positions (k: array)."""
    km1 = (np.asarray(k, dtype=float) - 1.0)
    if kind == 'wt_only':
        return {'wt_lora_pred': bw * km1}
    dw = bw * km1
    dm = (bw if kind == 'mtpass_wt' else bm) * km1
    return {'wt_lora_pred': dw, 'mt_lora_pred': dm, 'combined_pred': 0.5 * (dw + dm),
            'combined_dddg_pred': 0.5 * (dm - dw), 'epistasis_pred': 0.5 * (dm - dw)}


def runs(root):
    out = []
    for ck in [REL] + DEV:
        for sub in sorted(os.listdir(f'{root}/{ck}')):
            if os.path.isdir(f'{root}/{ck}/{sub}') and os.path.exists(f'{root}/{ck}/{sub}/summary.csv'):
                out.append((ck, sub, kind_of(sub)))
    return out


def files_of(d):
    return sorted(f for f in os.listdir(d) if f.endswith('.csv') and f != 'summary.csv')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def n_mut(df):
    """Mutated positions per row, from `mutant` (ProteinGym numbering) and cross-checked against `mut_type_renumbered`."""
    k1 = df['mutant'].str.count(':').to_numpy() + 1
    k2 = df['mut_type_renumbered'].str.count(':').to_numpy() + 1
    assert (k1 == k2).all(), 'mutation counts from `mutant` and `mut_type_renumbered` disagree'
    return k1


def read_str(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def as_float(s):
    return pd.to_numeric(s.replace('', np.nan), errors='raise').to_numpy(dtype=float)


# --------------------------------------------------------------------------------------------------------------------- check
def cmd_check(a):
    print(f'root {a.root}')
    bias = {ck: biases(ck) for ck in [REL] + DEV}
    for ck, (bw, bm) in bias.items():
        print(f'  {ck}: bias_wt {bw:+.9f}  bias_mt {bm:+.9f}  -> combined shift per extra mutation {0.5*(bw+bm):+.9f}')
    worst = {'combined': 0.0, 'dddg': 0.0, 'epistasis': 0.0}
    spear_bad, n_checked, hist = [], 0, {}
    for ck, sub, kind in runs(a.root):
        d = f'{a.root}/{ck}/{sub}'
        summ = pd.read_csv(f'{d}/summary.csv').drop_duplicates('DMS_id', keep='last').set_index('DMS_id')
        n_rows = n_shift = 0
        for f in files_of(d):
            df = read_str(f'{d}/{f}')
            k = n_mut(df)
            n_rows += len(df)
            n_shift += int((k > 1).sum())
            for v, c in zip(*np.unique(k, return_counts=True)):
                hist[int(v)] = hist.get(int(v), 0) + int(c)
            need = ['wt_lora_pred'] if kind == 'wt_only' else ['wt_lora_pred', 'mt_lora_pred', 'combined_pred', 'combined_dddg_pred']
            assert all(c in df.columns for c in need), (d, f, list(df.columns))
            wt = as_float(df['wt_lora_pred'])
            if kind != 'wt_only':
                mt, cb, dd = (as_float(df[c]) for c in ('mt_lora_pred', 'combined_pred', 'combined_dddg_pred'))
                worst['combined'] = max(worst['combined'], float(np.nanmax(np.abs(cb - 0.5 * (wt + mt)))))
                worst['dddg'] = max(worst['dddg'], float(np.nanmax(np.abs(dd - 0.5 * (mt - wt)))))
                if 'epistasis_pred' in df.columns:
                    worst['epistasis'] = max(worst['epistasis'], float(np.nanmax(np.abs(as_float(df['epistasis_pred']) - (cb - wt)))))
            did = f[:-4]
            if did in summ.index and summ.loc[did, 'status'] == 'done':
                rho = safe_spearman(as_float(df[pred_col(kind)]), as_float(df['DMS_score']))
                old = summ.loc[did, 'spearman_combined']
                n_checked += 1
                if not (np.isnan(rho) and np.isnan(old)) and not abs(rho - old) <= TOL_SPEARMAN:
                    spear_bad.append((ck, sub, did, rho, float(old)))
        print(f'  {ck}/{sub:22s} [{kind:9s}] files {len(files_of(d)):3d}  rows {n_rows:8d}  to shift (k>1) {n_shift:8d}')
    print('worst identity deviations on the OLD files:', {k: f'{v:.2e}' for k, v in worst.items()}, f'(tolerance {TOL_IDENTITY:.0e})')
    print('mutation-count histogram over all stored rows:', dict(sorted(hist.items())[:6]), '... max k', max(hist))
    print(f'stored Spearmans reproduced from the old predictions: {n_checked - len(spear_bad)} of {n_checked}')
    for r in spear_bad[:10]:
        print('   MISMATCH', r)
    ok = all(v <= TOL_IDENTITY for v in worst.values()) and not spear_bad
    print('CHECK', 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


# --------------------------------------------------------------------------------------------------------------------- backup
def cmd_backup(a):
    dst_root = a.root + '_pre_per_mutation_bias'
    n = 0
    for ck, sub, kind in runs(a.root):
        s, d = f'{a.root}/{ck}/{sub}', f'{dst_root}/{ck}/{sub}'
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(s):
            if not os.path.isfile(f'{s}/{f}'):
                continue
            if os.path.exists(f'{d}/{f}') and sha256(f'{d}/{f}') == sha256(f'{s}/{f}'):
                continue
            shutil.copy2(f'{s}/{f}', f'{d}/{f}')
            assert sha256(f'{d}/{f}') == sha256(f'{s}/{f}'), f'backup of {s}/{f} does not verify'
            n += 1
    for f in ('pgym_dc8_per_dms.csv', 'pgym_all_models_per_dms.csv'):
        if not os.path.exists(f'{dst_root}/{f}'):                   # never replace an existing backup of the originals
            shutil.copy2(f'{a.root}/{f}', f'{dst_root}/{f}')
            assert sha256(f'{dst_root}/{f}') == sha256(f'{a.root}/{f}'), f'backup of {f} does not verify'
    open(f'{dst_root}/README.txt', 'w').write(
        f'Copy of {a.root} taken {time.strftime("%F %T")} BEFORE the per-mutation-bias correction '
        f'(analysis_scripts/pgym_per_mutation_bias.py). Every file here was verified by sha256 against its source at copy time. '
        f'These values use ONE calibration bias per item.\n')
    print(f'backup complete: {n} files copied to {dst_root}')


# --------------------------------------------------------------------------------------------------------------------- apply
def cmd_apply(a):
    back = a.root + '_pre_per_mutation_bias'
    assert os.path.isdir(back), 'no backup directory: run `backup` first'
    for ck in [REL] + DEV:
        bw, bm = biases(ck)
        for c2, sub, kind in [r for r in runs(a.root) if r[0] == ck]:
            if kind == 'mtpass_wt' and not a.include_ablation:
                print(f'skip {ck}/{sub} (MT-pass ablation; pass --include-ablation to correct it)')
                continue
            d, bd = f'{a.root}/{ck}/{sub}', f'{back}/{ck}/{sub}'
            mpath = f'{d}/{MANIFEST}'
            man = json.load(open(mpath)) if os.path.exists(mpath) else {'files': {}}
            if man.get('complete'):
                print(f'skip {ck}/{sub}: already corrected ({man["finished"]})')
                continue
            man.update(formula=__doc__.split('SUBCOMMANDS')[0].strip(), bias_wt=bw, bias_mt=bm, kind=kind, checkpoint=f'{ck}/{CKPT[ck]}',
                       started=man.get('started', time.strftime('%F %T')))
            n_ch = n_rows = 0
            for f in files_of(d):
                cur, old = f'{d}/{f}', f'{bd}/{f}'
                h_cur, h_old = sha256(cur), sha256(old)          # the backup must exist and be the pre-correction file
                rec = man['files'].get(f)
                if rec and rec['new_sha256'] == h_cur:
                    continue                                       # already done (resumed run)
                assert h_cur == h_old, f'{cur} is neither the backed-up original nor a recorded corrected file; refusing to touch it'
                df = read_str(cur)
                k = n_mut(df)
                dl = deltas(kind, k, bw, bm)
                new = df.copy()
                shifted = k > 1
                maxshift = 0.0
                for c, dv in dl.items():
                    if c not in df.columns:
                        continue
                    vals = as_float(df[c])
                    nv = vals + dv
                    s = df[c].to_numpy(dtype=object).copy()
                    for i in np.nonzero(shifted & ~np.isnan(vals))[0]:
                        s[i] = repr(float(nv[i]))
                    new[c] = s
                    maxshift = max(maxshift, float(np.max(np.abs(dv))) if len(dv) else 0.0)
                cols_shift = [c for c in dl if c in df.columns]
                other = [c for c in df.columns if c not in cols_shift]
                n_rows += len(df)
                if shifted.any():
                    tmp = cur + '.tmp'
                    new.to_csv(tmp, index=False)
                    chk = read_str(tmp)                       # what is now on disk, read back as text
                    assert chk.shape == df.shape and (chk.columns == df.columns).all()
                    assert (chk[other].to_numpy() == df[other].to_numpy()).all(), 'a column that must not change differs on disk'
                    assert (chk.loc[~shifted, cols_shift].to_numpy() == df.loc[~shifted, cols_shift].to_numpy()).all(), 'a single changed'
                    for c in cols_shift:
                        diff = np.nan_to_num(as_float(chk[c]) - (as_float(df[c]) + dl[c]), nan=0.0)
                        assert np.abs(diff).max() < 1e-12, (f, c, np.abs(diff).max())
                    os.replace(tmp, cur)
                    n_ch += 1
                man['files'][f] = {'old_sha256': h_old, 'new_sha256': sha256(cur), 'rows': int(len(df)), 'rows_shifted': int(shifted.sum()),
                                   'max_abs_shift': maxshift}
                json.dump(man, open(mpath, 'w'), indent=1)
            # summary.csv: new Spearman from the corrected files, the old value kept beside it
            sp = f'{d}/summary.csv'
            summ = pd.read_csv(sp)
            if 'spearman_combined_per_item_bias' not in summ.columns:
                summ['spearman_combined_per_item_bias'] = summ['spearman_combined']
            for i, r in summ.iterrows():
                if r['status'] == 'done' and os.path.exists(f'{d}/{r["DMS_id"]}.csv'):
                    df = read_str(f'{d}/{r["DMS_id"]}.csv')
                    summ.loc[i, 'spearman_combined'] = safe_spearman(as_float(df[pred_col(kind)]), as_float(df['DMS_score']))
            summ.to_csv(sp + '.tmp', index=False)
            os.replace(sp + '.tmp', sp)
            man['complete'] = True
            man['finished'] = time.strftime('%F %T')
            json.dump(man, open(mpath, 'w'), indent=1)
            print(f'corrected {ck}/{sub} [{kind}]: {n_ch} files rewritten, {n_rows} rows scanned, bias_wt {bw:+.6f} bias_mt {bm:+.6f}')


# --------------------------------------------------------------------------------------------------------------------- aggregates
def per_dms(root, ck, sub):
    s = pd.read_csv(f'{root}/{ck}/{sub}/summary.csv').drop_duplicates('DMS_id', keep='last')
    s = s[s['status'] == 'done'].set_index('DMS_id')['spearman_combined']
    return s


def build_dc8(root, cats):
    out = pd.DataFrame(index=cats.index)
    out['category'] = cats
    for e in ('1.0', '0.75', '0.5', '0.25'):
        cols = []
        for i, ck in enumerate(DEV, 1):
            out[f'dc8 eps={e} s{i}'] = per_dms(root, ck, f'sigma{e}').reindex(out.index)
            cols.append(f'dc8 eps={e} s{i}')
        out[f'dc8 eps={e} mean'] = out[cols].mean(axis=1)
        out[f'dc8 eps={e} n_seeds'] = out[cols].notna().sum(axis=1).astype(float)
    return out.reset_index()


def build_all(root, cats):
    out = pd.DataFrame(index=cats.index)
    out['category'] = cats
    out['ESM-MSR full'] = per_dms(root, REL, 'sigma1.0').reindex(out.index)
    for e in ('0', '0.25', '0.5', '0.75', '1.0'):
        out[f'ESM-MSR WT eps={e}'] = per_dms(root, REL, f'sigma{e}_wt_only').reindex(out.index)
    return out.reset_index()


def cmd_aggregates(a):
    back = a.root + '_pre_per_mutation_bias'
    for fname, builder in (('pgym_dc8_per_dms.csv', build_dc8), ('pgym_all_models_per_dms.csv', build_all)):
        cur = pd.read_csv(f'{a.root}/{fname}')
        old = pd.read_csv(f'{back}/{fname}')
        cats = old.set_index('DMS_id')['category']
        built = builder(back, cats).set_index('DMS_id')                    # from the OLD summaries: must reproduce the old file
        o = old.set_index('DMS_id')
        for c in built.columns:
            if c == 'category' or c not in o.columns:
                continue
            x, y = built[c].reindex(o.index), o[c]
            same = ((x - y).abs() <= 1e-9) | (x.isna() & y.isna())
            assert same.all(), f'{fname}: cannot reproduce column {c!r} from the old summaries ({int((~same).sum())} DMS differ, e.g. {list(o.index[~same][:3])}); not touching it'
        print(f'{fname}: the {len([c for c in built.columns if c in o.columns and c != "category"])} ESM-MSR columns reproduce from the old summaries')
        new = builder(a.root, cats).set_index('DMS_id')
        res = cur.set_index('DMS_id')
        for c in new.columns:
            if c != 'category' and c in res.columns:
                res[c] = new[c].reindex(res.index)
        if a.write:
            res.reset_index().to_csv(f'{a.root}/{fname}.tmp', index=False)
            os.replace(f'{a.root}/{fname}.tmp', f'{a.root}/{fname}')
            print(f'  wrote {fname}')
        ch = [(c, float((res[c] - o[c]).abs().max())) for c in res.columns if c in o.columns and c != 'category']
        print('  largest per-DMS change by column:', {c: round(v, 4) for c, v in ch if v > 0})


# --------------------------------------------------------------------------------------------------------------------- compare
VERIFY = {'rel_sigma1.0': (REL, 'sigma1.0'), 'rel_sigma1.0_wt_only': (REL, 'sigma1.0_wt_only'),
          'rel_sigma0.5_wt_only': (REL, 'sigma0.5_wt_only'), 's1_sigma1.0': (DEV[0], 'sigma1.0'), 's1_sigma0.5': (DEV[0], 'sigma0.5')}


def cmd_compare(a):
    back = a.root + '_pre_per_mutation_bias'
    vroot = a.root.replace('/pgym', '/pgym_verify_pmb')
    ok_all = True
    for name, (ck, sub) in VERIFY.items():
        if not os.path.isdir(f'{vroot}/{name}'):
            print(f'{name}: no rerun yet')
            continue
        kind = kind_of(sub)
        bw, bm = biases(ck)
        for f in files_of(f'{vroot}/{name}'):
            new = read_str(f'{vroot}/{name}/{f}')
            old = read_str(f'{back}/{ck}/{sub}/{f}')
            assert (new['mutant'].to_numpy() == old['mutant'].to_numpy()).all(), f'{name}/{f}: rows differ in order or content'
            k = n_mut(old)
            dl = deltas(kind, k, bw, bm)
            line = []
            for c, dv in dl.items():
                if c not in old.columns or c not in new.columns:
                    continue
                exp = as_float(old[c]) + dv
                got = as_float(new[c])
                err = np.nanmax(np.abs(got - exp))
                err1 = np.nanmax(np.abs(got - as_float(old[c]))[k == 1]) if (k == 1).any() else 0.0
                line.append(f'{c} max|rerun-expected| {err:.1e}')
                ok_all &= err < 5e-3 and err1 < 5e-3
            rho_new = safe_spearman(as_float(new[pred_col(kind)]), as_float(new['DMS_score']))
            rho_exp = safe_spearman(as_float(old[pred_col(kind)]) + dl[pred_col(kind)], as_float(old['DMS_score']))
            rho_old = safe_spearman(as_float(old[pred_col(kind)]), as_float(old['DMS_score']))
            print(f'{name:22s} {f[:34]:34s} k>1 rows {int((k>1).sum()):5d} kmax {int(k.max()):2d} | ' + '; '.join(line))
            print(f'{"":22s} {"":34s} Spearman: old {rho_old:.5f}  corrected-from-old {rho_exp:.5f}  real rerun {rho_new:.5f}')
    print('COMPARE', 'PASSED' if ok_all else 'FAILED')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('SUBCOMMANDS')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['check', 'backup', 'apply', 'aggregates', 'compare'])
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--include-ablation', action='store_true', dest='include_ablation')
    ap.add_argument('--write', action='store_true', help='aggregates: write the files (default: report only)')
    a = ap.parse_args()
    {'check': cmd_check, 'backup': cmd_backup, 'apply': cmd_apply, 'aggregates': cmd_aggregates, 'compare': cmd_compare}[a.cmd](a)
