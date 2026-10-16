"""
Assemble per-mutation predictions for the external benchmark datasets of
compare_supervised_external_summarize.ipynb, using ONE model replicate each.

The predictions live outside this repo (default PRED_ROOT below, one directory per
dataset). This is a replicate-1-only port of the notebook's loading cell: the same files,
joins, de-duplication and subset extraction, so the full-data Spearman values reproduce
the `_1` columns of the notebook's external_test.csv.

Each returned frame is indexed by `uid` and has one column per model (named as in the
notebook, with the replicate suffix `_1` where the notebook has replicates), plus `code`
(the protein / cluster id), `mut_type` and `ground_truth`.
"""
import contextlib
import io
import os

import numpy as np
import pandas as pd

from analysis_utils_msr import deduplicate_and_check, sort_mutations_by_position

PRED_ROOT = '/home/sareeves/software/esm-msr/analysis_notebooks/predictions'
REPO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'preprocessed')

# Datasets with a single ground-truth/prediction pool per DMS (no chain in the uid)
DMS_SINGLE = ['MYO_HUMAN_Kung_2025_display', 'GB1_Wu_2016_binding_domain',
              'DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain',
              'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain']
DMS_NO_ROSETTA = DMS_SINGLE + ['ESTA_BACSU_Nutschel_2020_dTm', 'domainome']
STABILITY_CHAIN_UID = ['PTMUL', 's669', 'ssym', 'q3421', 'k3822', 's571', 's783', 's8754', 's2648']
GET_ADDITIVE_PMPNN = DMS_SINGLE + ['PTMUL']
GROUPED = ['hyperopt_splits-val', 'hyperopt_splits-test', 'domainome']  # per-domain mean Spearman

# σ names follow the notebook: alpha20 -> σ=1.25, alpha16 -> plain ESM-MSR, alpha8 -> σ=0.5
ESM_ALPHAS = {'ESM-MSR (σ=1.25)': 20.0, 'ESM-MSR': 16.0, 'ESM-MSR (σ=0.5)': 8.0}
ESM_CKPT = 'epoch=05-val_rho_combined_avg=0.805.ckpt'

THERMO_S = 'single_hyperopt_splits_correct_bb_1_epoch=99_val_ddG_spearman=0.73.ckpt'
THERMO_E = 'epistatic_hyperopt_splits_correct_bb_1_epoch=84_val_ddG_spearman=0.74.ckpt'
THERMO_A = 'additive_single_hyperopt_splits_correct_bb_1_epoch=99_val_ddG_spearman=0.73.ckpt'

# ground_truth/metadata columns that are not models
META_COLS = ['code', 'mut_type', 'ddG_ML_esm', 'ddG_ML_thermo', 'ddG_ML_me', 'ground_truth']


def _quiet(fn, *args, **kwargs):
    """Run fn with the notebook helpers' chatty stdout suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _dedup(df, dataset, name):
    return _quiet(deduplicate_and_check, df, dataset, name)


def _load_rep(path, pred_col, name, true_col=None, extra_cols=()):
    """One replicate CSV -> [name, extra cols, true col]; mirrors load_replicate_ensemble."""
    df = pd.read_csv(path, index_col=0)
    if 'mut_info_seq_pos' in df.columns:
        df['mut_type'] = df['mut_info_seq_pos']
    keep = [pred_col] + [c for c in extra_cols if c in df.columns]
    if true_col and true_col in df.columns:
        keep.append(true_col)
    return df[[c for c in keep if c in df.columns]].copy().rename(columns={pred_col: name})


def _load_esm_msr(paths, prefix, dataset):
    """Mirrors process_esm_msr for one replicate; returns [prefix_1, prefix_additive_1, ...]."""
    df = pd.read_csv(paths[0], index_col=0)
    if 'mut_type' not in df.columns and 'mut_info' in df.columns:
        df['mut_type'] = df['mut_info']
    if 'ddG_ML' not in df.columns and 'ddG' in df.columns:
        df['ddG_ML'] = df['ddG']
    if 'ddg_pred' not in df.columns and 'combined_pred' in df.columns:
        df['ddg_pred'] = df['combined_pred']
    df[f'{prefix}_1'] = df['ddg_pred']
    keep = [f'{prefix}_1', 'mut_info', 'mut_type', 'ddG_ML', 'code']
    df[f'{prefix}_additive_1'] = df['wt_lora_pred'] if 'wt_lora_pred' in df.columns else np.nan
    keep.append(f'{prefix}_additive_1')
    res = df[[c for c in keep if c in df.columns]].copy()

    if dataset == 'domainome':
        res = res.loc[(~res.index.isna()) & (~res['ddG_ML'].isna())]
        idx = pd.Series(res.index)
        res['prefix'] = idx.apply(lambda x: '_'.join(x.split('_')[:-1])).values
        res['mut_info'] = idx.apply(lambda x: x.split('_')[-1]).values
        res = sort_mutations_by_position(res, 'mut_info', 'mut_type')
        res['uid'] = res['prefix'] + '_' + res['mut_type']
        res = res.set_index('uid')
    res.index.name = 'uid'
    return res


def _thermo(root, dataset):
    d = f'{root}/{dataset}/thermompnnd'
    if dataset == 'ESTA_BACSU_Nutschel_2020_dTm' or not (
            dataset in DMS_SINGLE or '-val' in dataset or '-test' in dataset or dataset == 'PTMUL'):
        try:
            return _load_rep(f'{d}/{THERMO_S}.csv', 'ddG_pred', 'ThermoMPNN_1', 'ddG_true',
                             ['code', 'mut_type']).rename(columns={'ddG_true': 'ddG_ML_thermo'})
        except FileNotFoundError as e:
            print(f'Missing ThermoMPNN data for {dataset}: {e}')
            return pd.DataFrame()

    if dataset == 'PTMUL':
        ts_epi = _load_rep(f'{root}/ptmuld/thermompnnd/{THERMO_E}.csv', 'ddG_pred', 'ThermoMPNN-D_1',
                           'ddG_true', ['code', 'mut_type']).rename(columns={'ddG_true': 'ddG_ML_thermo'})
        ts_add = _load_rep(f'{d}/{THERMO_A}.csv', 'ddG_pred_additive', 'ThermoMPNN_1', 'ddG_true',
                           ['code', 'mut_type']).rename(columns={'ddG_true': 'ddG_ML_thermo'})
        out = ts_add.join(ts_epi[[c for c in ts_epi.columns if 'ThermoMPNN-D' in c]], how='outer')
        out = out.drop_duplicates()  # as in the notebook
        out['ThermoMPNN-D_1'] = out['ThermoMPNN-D_1'].fillna(out['ThermoMPNN_1'])
        return out

    s_file = f'{d}/{THERMO_S}_singles.csv' if ('-val' in dataset or '-test' in dataset) else f'{d}/{THERMO_S}.csv'
    ts_single = _load_rep(s_file, 'ddG_pred', 'ThermoMPNN_1', 'ddG_true', ['code', 'mut_type'])
    ts_epi = _load_rep(f'{d}/{THERMO_E}.csv', 'ddG_pred', 'ThermoMPNN-D_1')
    ts_add = _load_rep(f'{d}/{THERMO_A}.csv', 'ddG_pred_additive', 'ThermoMPNN_1').dropna(subset=['ThermoMPNN_1'])
    epi = pd.concat([ts_single.rename(columns={'ThermoMPNN_1': 'ThermoMPNN-D_1'})[['ThermoMPNN-D_1']], ts_epi])
    add = pd.concat([ts_single[['ThermoMPNN_1']], ts_add])
    out = add.join(epi, how='inner')
    return out.join(ts_single[['code', 'mut_type', 'ddG_true']].rename(columns={'ddG_true': 'ddG_ML_thermo'}), how='left')


def _mutate_everything(root, dataset):
    d = f'{root}/{dataset}/mutate_everything'
    try:
        df_me = _load_rep(f'{d}/af_double_hyperopt_splits_1_checkpoint-99.csv', 'ddG_pred', 'MutateEverything_1',
                          'ddG_true', ['code', 'chain', 'mut_type']).rename(columns={'ddG_true': 'ddG_ML_me'})
        single = _load_rep(f'{d}/af_single_hyperopt_splits_1_checkpoint-19.csv', 'ddG_pred',
                           'MutateEverything_single_1', extra_cols=['code', 'chain', 'mut_type'])
    except FileNotFoundError as e:
        print(f'MutateEverything files missing for {dataset}: {e}')
        return pd.DataFrame(), pd.DataFrame()
    try:
        add = _load_rep(f'{d}/af_single_additive_hyperopt_splits_1_checkpoint-19.csv', 'ddG_pred_additive',
                        'MutateEverything_single_additive_1')
        comb = add.join(single, how='outer')
        # some files (e.g. domainome) carry no additive column: fall back to the single prediction
        if 'MutateEverything_single_additive_1' in comb.columns:
            comb['MutateEverything_additive_1'] = comb['MutateEverything_single_additive_1'].fillna(
                comb['MutateEverything_single_1'])
        else:
            comb['MutateEverything_additive_1'] = comb['MutateEverything_single_1']
    except FileNotFoundError:
        comb = single.copy()
        comb['MutateEverything_additive_1'] = comb['MutateEverything_single_1']
    comb = comb.drop(columns=['MutateEverything_single_1'], errors='ignore')
    return df_me, comb


def _esm3_zero_shot(root, dataset):
    """ESM3 small/medium/large zero-shot (full + additive); uid uses chain only for stability sets."""
    sizes = {'ESM3-small': 'esm3-small-2024-08', 'ESM3-medium': 'esm3-medium-2024-08', 'ESM3-large': 'esm3-large-2024-03'}
    use_chain = not (dataset in DMS_NO_ROSETTA or '-val' in dataset or '-test' in dataset)
    out = []
    for name, sub in sizes.items():
        path = f'{root}/{dataset}/esm3_forge/zero_shot/{sub}/predictions_unmasked.csv'
        try:
            df = pd.read_csv(path, index_col=0).rename(columns={'esm3_score': name, 'esm3_score_wt_seq': f'{name}_additive'})
        except FileNotFoundError as e:
            print(f'ESM3 zero-shot files missing for {dataset}: {e}')
            return []
        df = df.reset_index(drop=True)
        df['uid'] = df['code'] + (df['chain'] if use_chain else '') + '_' + df['mut_type']
        out.append(_dedup(df.set_index('uid'), dataset, f'df_{name}'))
    return out


def _rosetta(root, dataset):
    if dataset in DMS_NO_ROSETTA:
        return pd.DataFrame()
    try:
        df = pd.read_csv(f'{root}/{dataset}/rosetta/cartesian_ddg.csv', index_col=0)
    except FileNotFoundError:
        print(f'Rosetta files not found for {dataset}, skipping.')
        return pd.DataFrame()
    df['Rosetta Cartesian DDG_1'] = df['cartesian_ddg_1'] / 2.94
    if dataset == 'PTMUL' and 'cartesian_ddg_1_additive' in df.columns:
        df['Rosetta Cartesian DDG_additive_1'] = df['cartesian_ddg_1_additive'] / 2.94
    if 'uid_true' in df.columns:
        df = df.reset_index(drop=True).rename(columns={'uid_true': 'uid'})
    else:
        df.index.name = 'uid'
        df = df.reset_index()
    df['code'] = df['uid'].apply(lambda x: x.split('_')[0])
    df['mut_type'] = df['uid'].apply(lambda x: x.split('_')[-1])
    return df.set_index('uid')


def _proteinmpnn(root, dataset):
    try:
        df = pd.read_csv(f'{root}/{dataset}/proteinmpnn/proteinmpnn_020_unmasked.csv', index_col=0)
    except FileNotFoundError:
        return pd.DataFrame()
    df = df.reset_index(drop=True).rename(columns={'mpnn_score': 'ProteinMPNN'})
    df['uid'] = df['code'] + (df['chain'] if dataset in STABILITY_CHAIN_UID else '') + '_' + df['mut_type']
    df = df.set_index('uid')
    if dataset in GET_ADDITIVE_PMPNN:
        df = df.rename(columns={'mpnn_score_additive': 'ProteinMPNN_additive'})
    return df


def _spurs(root, dataset):
    try:
        df = pd.read_csv(f'{root}/{dataset}/SPURS/hf+hf.csv', index_col=0)
    except FileNotFoundError as e:
        print(f'SPURS predictions missing for {dataset}: {e}')
        return pd.DataFrame()
    df['SPURS'] = df['SPURS_single_additive']
    df['SPURS-multi'] = df['SPURS_multi']
    df = df[['code', 'mut_type', 'SPURS', 'SPURS-multi']]
    df.index.name = 'uid'
    return df


def _cols(df, key, exclude=None):
    return [c for c in df.columns if key in c and (exclude is None or exclude not in c)]


def load_base(dataset, root=PRED_ROOT):
    """One dataset's merged predictions (all models, replicate 1) -> (df, helper frames)."""
    dedup = lambda d, n: _dedup(d, dataset, n)

    thermo = dedup(_thermo(root, dataset), 'df_thermo')
    me, me_single = _mutate_everything(root, dataset)
    me, me_single = dedup(me, 'df_me'), dedup(me_single, 'df_me_single')
    ros = dedup(_rosetta(root, dataset), 'df_ros')
    mpnn = dedup(_proteinmpnn(root, dataset), 'df_mpnn')
    spurs = dedup(_spurs(root, dataset), 'df_spurs')
    zs = _esm3_zero_shot(root, dataset)

    esm = {}
    for name, alpha in ESM_ALPHAS.items():
        fixed = '_fixed' if (name != 'ESM-MSR (σ=1.25)' and 'gb1' in dataset.lower()) else ''
        path = f'{root}/{dataset}/msr_dual_ensemble_small_large/1/{ESM_CKPT}_alpha{alpha}_predictions{fixed}.csv'
        esm[name] = dedup(_load_esm_msr([path], name, dataset), f'df_{name}')
    esm_zs = _load_esm_msr([f'{root}/{dataset}/zeroshot_alpha0_predictions'
                            f"{'_fixed' if 'gb1' in dataset.lower() else ''}.csv"], 'ESM3-small-open', dataset)
    esm_zs = esm_zs.rename(columns={'ESM3-small-open_1': 'ESM3-small-open',
                                    'ESM3-small-open_additive_1': 'ESM3-small-open_additive'})
    esm_zs = dedup(esm_zs, 'df_esm_zs')

    base = esm['ESM-MSR']
    df = base[_cols(base, 'MSR', '_single') + ['code', 'mut_type', 'ddG_ML']].rename(columns={'ddG_ML': 'ddG_ML_esm'})
    df = (df.join(me[_cols(me, 'MutateEverything', '_single') + (['ddG_ML_me'] if 'ddG_ML_me' in me else [])], how='left')
            .join(thermo[_cols(thermo, 'ThermoMPNN') + (['ddG_ML_thermo'] if 'ddG_ML_thermo' in thermo else [])], how='left')
            .join(me_single[_cols(me_single, 'MutateEverything_additive')], how='left')
            .join(ros[_cols(ros, 'Rosetta')], how='left')
            .join(mpnn[_cols(mpnn, 'ProteinMPNN')], how='left')
            .join(spurs[[c for c in spurs.columns if 'SPURS' in c and '_single' not in c]], how='left'))
    for name in ('ESM-MSR (σ=1.25)', 'ESM-MSR (σ=0.5)'):
        df = df.join(esm[name][_cols(esm[name], 'MSR', '_single')], how='left')
    df = df.join(esm_zs[['ESM3-small-open', 'ESM3-small-open_additive']], how='left')
    for z, name in zip(zs, ['ESM3-small', 'ESM3-medium', 'ESM3-large']):
        df = df.join(z[[name, f'{name}_additive']], how='left')
    helpers = dict(thermo=thermo, me=me, me_single=me_single, ros=ros, mpnn=mpnn, spurs=spurs, esm=esm,
                   esm_zs=esm_zs, zs=dict(zip(['small', 'medium', 'large'], zs)))
    return df, helpers


def _read_mapped(name):
    return pd.read_csv(os.path.join(REPO_DATA, f'{name}_mapped.csv'), index_col=0)


def _uid_from_parts(df):
    return df['code'] + df['chain'] + '_' + df['wild_type'] + df['seq_pos'].astype(int).astype(str) + df['mutation']


def load_external(datasets, root=PRED_ROOT, verbose=True):
    """
    Returns {dataset: DataFrame}. Derived subsets (s461 from s669, k2369 from k3822, ptmuld from PTMUL)
    are produced when their parent is requested, exactly as in the notebook.
    """
    results = {}
    for dataset in datasets:
        if verbose:
            print(f'Loading {dataset}', flush=True)
        df, h = load_base(dataset, root)
        df['ground_truth'] = df['ddG_ML_esm']

        if dataset == 'ssym':
            wt = pd.read_csv(os.path.join(REPO_DATA, 'ssym_mapped.csv'), index_col=0).reset_index(drop=True)
            wt['uid'] = _uid_from_parts(wt)
            wt = _dedup(wt.set_index('uid'), dataset, 'ssym_wt_df')
            df = df.join(wt[['wt_code']]).rename(columns={'code': 'mut_code', 'wt_code': 'code'}).drop(
                columns='mut_code', errors='ignore')
            df['ground_truth'] = df['ddG_ML_esm']
        results[dataset] = df

        if dataset == 'PTMUL':
            ptmuld = _read_mapped('ptmuld')
            ptmuld['uid'] = ptmuld['code'] + ptmuld['chain'] + '_' + ptmuld['mut_info_seq_pos']
            ptmuld = _dedup(ptmuld.set_index('uid'), 'ptmuld', 'ptmuld_mapping_table')
            valid = ptmuld.index
            pick = lambda d: d.loc[d.index.intersection(valid)] if not d.empty else pd.DataFrame()
            esm = h['esm']
            d_esm = pick(esm['ESM-MSR'])
            sub = lambda d, key, extra=(): (pick(d)[_cols(pick(d), key) + [c for c in extra if c in pick(d).columns]]
                                            if not d.empty else pd.DataFrame())
            df2 = d_esm[_cols(d_esm, 'MSR') + ['ddG_ML', 'code', 'mut_type']].rename(columns={'ddG_ML': 'ddG_ML_esm'})
            parts = [sub(h['thermo'], 'ThermoMPNN', ['ddG_ML_thermo']),
                     sub(h['me'], 'MutateEverything', ['ddG_ML_me']),
                     sub(h['me_single'], 'MutateEverything_additive'),
                     sub(h['esm_zs'], 'ESM3'),
                     sub(h['zs']['small'], 'ESM3'), sub(h['zs']['medium'], 'ESM3'), sub(h['zs']['large'], 'ESM3'),
                     sub(esm['ESM-MSR (σ=1.25)'], 'MSR'), sub(esm['ESM-MSR (σ=0.5)'], 'MSR'),
                     sub(h['ros'], 'Rosetta'), sub(h['mpnn'], 'ProteinMPNN'),
                     pick(h['spurs'])[[c for c in h['spurs'].columns if 'SPURS' in c and '_single' not in c]]
                     if not h['spurs'].empty else pd.DataFrame()]
            for p in parts:
                df2 = df2.join(p, how='left')
            df2 = ptmuld[['ddG']].join(df2, how='inner')
            df2 = _dedup(df2, 'ptmuld', 'PTMULD df').rename(columns={'ddG': 'ground_truth'})
            results['ptmuld'] = df2

        if dataset == 's669':
            s461 = _read_mapped('s461')
            s461['uid'] = _uid_from_parts(s461)
            s461 = _dedup(s461.set_index('uid'), 's461', 's461_mapping_table')
            results['s461'] = s461[['ddG']].join(df.drop(columns=['ddG', 'ground_truth'], errors='ignore'), how='inner').rename(
                columns={'ddG': 'ground_truth'})

        if dataset == 'k3822':
            k = _read_mapped('k2369')
            k['uid'] = _uid_from_parts(k)
            k = _dedup(k.set_index('uid'), 'k2369', 'k2369_mapping_table')
            results['k2369'] = df.loc[df.index.intersection(k.index)].copy()
    return results


def model_columns(df):
    return [c for c in df.columns if c not in META_COLS]
