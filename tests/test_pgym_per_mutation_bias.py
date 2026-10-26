"""analysis_scripts/pgym_per_mutation_bias.py corrects stored ESM-MSR ProteinGym predictions for the per-mutation calibration
bias: each head moves by bias * (k - 1) for an item with k mutated positions, and every derived column by the matching linear
combination. A synthetic tree with known biases pins the arithmetic, that singles and untouched columns stay byte-identical, that
NaN predictions stay NaN, that a second apply is a no-op and that a file which is neither the backed-up original nor a recorded
corrected file is refused.
"""
import importlib.util
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('yaml')

SCRIPT = Path(__file__).resolve().parent.parent / 'analysis_scripts' / 'pgym_per_mutation_bias.py'


@pytest.fixture()
def tool(monkeypatch):
    spec = importlib.util.spec_from_file_location('pgym_per_mutation_bias', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    bias = {m.REL: (0.13, -0.035), **{d: (0.1 - 0.01 * i, -0.1 - 0.02 * i) for i, d in enumerate(m.DEV)}}
    monkeypatch.setattr(m, 'biases', lambda ck: bias[ck])
    m.TEST_BIAS = bias
    return m


def _make(m, d, kind, name, ks, rng, nan_row=False, epi=False):
    n = len(ks)
    wt, mt = rng.normal(size=n).astype(np.float32), rng.normal(size=n).astype(np.float32)
    cb = (np.float32(0.5) * wt + np.float32(0.5) * mt).astype(np.float32)
    muts = [':'.join(f'A{i + 1}C' for i in range(k)) for k in ks]
    df = pd.DataFrame({'code': name, 'chain': 'A', 'mut_type_renumbered': muts, 'pdb_file': '/x/y.pdb', 'mut_type_pdb': muts})
    df['wt_lora_pred'] = wt.astype(float)
    if kind != 'wt_only':
        df['mt_lora_pred'], df['combined_pred'] = mt.astype(float), cb.astype(float)
        df['combined_dddg_pred'] = (np.float32(0.5) * mt - np.float32(0.5) * wt).astype(float)
        if epi:
            df['epistasis_pred'] = (mt - wt).astype(float)       # the released run stored the full head difference, not combined - wt
        if nan_row:
            df.loc[ks.index(max(ks)), 'combined_pred'] = np.nan
    df['DMS_id'], df['mutant'], df['DMS_score'] = name, muts, rng.normal(size=n)
    df.to_csv(f'{d}/{name}.csv', index=False)


def _tree(m, tmp_path):
    root = str(tmp_path / 'pgym')
    rng = np.random.default_rng(0)
    for ck in [m.REL] + m.DEV:
        subs = {'sigma1.0': 'full', **({'sigma1.0_wt_only': 'wt_only'} if ck == m.REL else {})}
        for sub, kind in subs.items():
            d = f'{root}/{ck}/{sub}'
            os.makedirs(d)
            _make(m, d, kind, 'A_single', [1] * 30, rng, epi=ck == m.REL)
            _make(m, d, kind, 'B_mixed', [1] * 40 + [2] * 25 + [3] * 8 + [7] * 3, rng, epi=ck == m.REL)
            _make(m, d, kind, 'C_nan', [1] * 20 + [2] * 15 + [5], rng, nan_row=True, epi=ck == m.REL)
            rows = []
            for nm in ('A_single', 'B_mixed', 'C_nan'):
                df = pd.read_csv(f'{d}/{nm}.csv')
                rows.append({'DMS_id': nm, 'n': len(df), 'spearman_combined': m.safe_spearman(df[m.pred_col(kind)], df['DMS_score']),
                             'time_s': 1.0, 'status': 'done', 'error': ''})
            if ck == m.REL and sub == 'sigma1.0':
                rows[2]['status'] = 'skipped'            # a resumed run records an already-scored DMS this way, with its Spearman
            pd.DataFrame(rows).to_csv(f'{d}/summary.csv', index=False)
    for f in ('pgym_dc8_per_dms.csv', 'pgym_all_models_per_dms.csv'):
        open(f'{root}/{f}', 'w').write('DMS_id,category\nA_single,x\n')
    return root


def test_correction_matches_an_independent_computation(tool, tmp_path, capsys):
    m = tool
    root = _tree(m, tmp_path)
    a = types.SimpleNamespace(root=root, include_ablation=False, write=False)
    with pytest.raises(SystemExit) as e:
        m.cmd_check(a)
    assert e.value.code == 0
    m.cmd_backup(a)
    m.cmd_apply(a)
    back = root + '_pre_per_mutation_bias'
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root).split(os.sep)
            if len(rel) < 3 or not f.endswith('.csv') or f == 'summary.csv':
                continue
            ck, sub = rel[0], rel[1]
            kind, (bw, bm) = m.kind_of(sub), m.TEST_BIAS[ck]
            o, n = pd.read_csv(os.path.join(back, *rel)), pd.read_csv(p)
            k = o['mutant'].str.count(':') + 1
            exp = {'wt_lora_pred': bw * (k - 1)}
            if kind != 'wt_only':
                exp.update(mt_lora_pred=bm * (k - 1), combined_pred=0.5 * (bw + bm) * (k - 1), combined_dddg_pred=0.5 * (bm - bw) * (k - 1))
                if 'epistasis_pred' in o:
                    exp['epistasis_pred'] = (bm - bw) * (k - 1)
            for c, dv in exp.items():
                d = (n[c] - o[c] - dv).abs().max()
                assert np.isnan(d) or d < 1e-12, (p, c, d)
                assert (n.loc[k == 1, c].fillna(-9) == o.loc[k == 1, c].fillna(-9)).all(), 'a single moved'
            assert n['DMS_score'].equals(o['DMS_score']) and n['mutant'].equals(o['mutant'])
            if f == 'C_nan.csv' and kind != 'wt_only':
                assert n['combined_pred'].isna().sum() == 1
    single_only = f'{root}/{m.REL}/sigma1.0/A_single.csv'
    assert open(single_only, 'rb').read() == open(f'{back}/{m.REL}/sigma1.0/A_single.csv', 'rb').read()   # not even rewritten

    snap = {os.path.join(dp, f): open(os.path.join(dp, f), 'rb').read() for dp, _, fs in os.walk(root) for f in fs}
    m.cmd_apply(a)                                                    # a second apply does nothing
    assert snap == {p: open(p, 'rb').read() for p in snap}
    s = pd.read_csv(f'{root}/{m.REL}/sigma1.0/summary.csv')
    assert (s['spearman_combined_per_item_bias'] != s['spearman_combined']).sum() == 2    # B and C moved; singles-only A did not
    assert s.loc[2, 'status'] == 'skipped'                                                # ... including the row whose status is 'skipped'
    nw = pd.read_csv(f'{root}/{m.REL}/sigma1.0/C_nan.csv')
    assert s.loc[2, 'spearman_combined'] == pytest.approx(m.safe_spearman(nw['combined_pred'], nw['DMS_score']))


def test_a_file_that_is_not_the_backed_up_original_is_refused(tool, tmp_path):
    m = tool
    root = _tree(m, tmp_path)
    a = types.SimpleNamespace(root=root, include_ablation=False, write=False)
    m.cmd_backup(a)
    open(f'{root}/{m.REL}/sigma1.0/B_mixed.csv', 'a').write('\n')
    with pytest.raises(AssertionError, match='refusing to touch'):
        m.cmd_apply(a)


def test_apply_needs_a_backup(tool, tmp_path):
    m = tool
    root = _tree(m, tmp_path)
    with pytest.raises(AssertionError, match='backup'):
        m.cmd_apply(types.SimpleNamespace(root=root, include_ablation=False, write=False))


def test_summaries_refreshes_a_row_that_apply_left_stale(tool, tmp_path):
    """The bug this guards against: apply once refreshed only status == 'done' rows, so a 'skipped' row kept its per-item Spearman."""
    m = tool
    root = _tree(m, tmp_path)
    a = types.SimpleNamespace(root=root, include_ablation=False, write=False, only=None)
    m.cmd_backup(a)
    m.cmd_apply(a)
    sp = f'{root}/{m.REL}/sigma1.0/summary.csv'
    s = pd.read_csv(sp)
    stale = s['spearman_combined_per_item_bias'].copy()
    s.loc[2, 'spearman_combined'] = stale[2]                              # put the stale value back, as the old apply left it
    s.to_csv(sp, index=False)
    m.cmd_summaries(a)
    s2 = pd.read_csv(sp)
    nw = pd.read_csv(f'{root}/{m.REL}/sigma1.0/C_nan.csv')
    assert s2.loc[2, 'spearman_combined'] == pytest.approx(m.safe_spearman(nw['combined_pred'], nw['DMS_score']))
    assert s2.loc[2, 'spearman_combined'] != pytest.approx(stale[2])
    assert (s2['spearman_combined_per_item_bias'] == stale).all()


def test_aggregates_keep_unchanged_cells_exactly_and_shift_only_the_corrected_ones(tool, tmp_path):
    m = tool
    root = _tree(m, tmp_path)
    cats = pd.Series({'A_single': 'x', 'B_mixed': 'x', 'C_nan': 'x'}, name='category')
    cats.index.name = 'DMS_id'
    a = types.SimpleNamespace(root=root, include_ablation=False, write=True, only=None)
    # the synthetic tree has none of the sigma directories the real builders read, so use minimal builders over the same helpers
    m.build_dc8 = lambda r, c: pd.DataFrame({'DMS_id': c.index, 'category': c.values})
    m.build_all = lambda r, c: pd.DataFrame({'DMS_id': c.index, 'category': c.values,
                                             'ESM-MSR full': m.per_dms(r, m.REL, 'sigma1.0').reindex(c.index).values})
    m.cmd_backup(a)
    # the released 'full' column: take the summary, but make one single-mutant-only cell differ from it by 3e-7 (compiled earlier)
    s0 = pd.read_csv(f'{root}/{m.REL}/sigma1.0/summary.csv').set_index('DMS_id')['spearman_combined']
    agg = pd.DataFrame({'category': cats, 'ESM-MSR full': s0.reindex(cats.index)})
    agg.loc['A_single', 'ESM-MSR full'] += 3e-7
    agg.reset_index().to_csv(f'{root}/pgym_all_models_per_dms.csv', index=False)
    pd.DataFrame({'DMS_id': cats.index, 'category': 'x'}).to_csv(f'{root}/pgym_dc8_per_dms.csv', index=False)
    shutil_copy = __import__('shutil').copy2
    shutil_copy(f'{root}/pgym_all_models_per_dms.csv', f'{root}_pre_per_mutation_bias/pgym_all_models_per_dms.csv')
    m.cmd_apply(a)
    m.cmd_aggregates(a)
    out = pd.read_csv(f'{root}/pgym_all_models_per_dms.csv').set_index('DMS_id')['ESM-MSR full']
    new = pd.read_csv(f'{root}/{m.REL}/sigma1.0/summary.csv').set_index('DMS_id')['spearman_combined']
    assert out['A_single'] == pytest.approx(agg.loc['A_single', 'ESM-MSR full'], abs=1e-15)      # singles-only: untouched, incl. its 3e-7
    for d in ('B_mixed', 'C_nan'):
        assert out[d] == pytest.approx(new[d], abs=1e-9)                                           # corrected cells follow the new summary
