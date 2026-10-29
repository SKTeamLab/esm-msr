"""analysis_scripts/score_baseline_runs.py: the filters and aggregation that put the retrained SPURS rows in the BENCHMARKS tables."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis_scripts'))
import score_baseline_runs as S


@pytest.fixture(scope='module')
def update_stats():
    return S.load_update_stats()


def _tsuboyama(tmp_path, n=30):
    rng = np.random.default_rng(0)
    truth = rng.normal(size=n)
    df = pd.DataFrame({'code_wt': ['A'] * n, 'mut_type': [f'A{i}G' for i in range(n)], 'ddG_ML': truth,
                       'dddG_ML': np.nan, 'cens': 0, 'pred': truth + rng.normal(scale=0.3, size=n)})
    bad = df.iloc[:10].copy()                                   # bounded rows: perfectly predicted, flagged cens == 1
    bad['cens'], bad['pred'] = 1, bad['ddG_ML']
    path = tmp_path / 'tsuboyama_val.csv'
    pd.concat([df, bad]).to_csv(path, index=False)
    return path, df


def test_tsuboyama_scores_measured_rows_only(update_stats, tmp_path):
    path, measured = _tsuboyama(tmp_path)
    got = S.score_tsuboyama(update_stats, str(path), 'pred', None)['spearman_all']
    assert got == pytest.approx(measured['ddG_ML'].corr(measured['pred'], method='spearman'))
    with_bounded = pd.read_csv(path)
    assert with_bounded['ddG_ML'].corr(with_bounded['pred'], method='spearman') > got + 0.01      # the filter is what keeps SPURS honest


def test_proteingym_summary_is_the_mean_of_category_means_over_the_kept_dms(tmp_path):
    dms = ['d1', 'd2', 'd3', S.PG_DROP.copy().pop()]
    cats = pd.DataFrame({'DMS_id': dms, 'category': ['Stability', 'Stability', 'Binding', 'Binding']})
    cats.to_csv(tmp_path / 'cats.csv', index=False)
    pd.DataFrame({'DMS_id': dms, 'rho': [0.8, 0.6, 0.2, 0.9]}).to_csv(tmp_path / 'summary.csv', index=False)
    out = S.pg_summary(str(tmp_path / 'summary.csv'), 'rho', str(tmp_path / 'cats.csv'))
    assert out['n_dms'] == 3                                     # the dropped assay is left out of every aggregate
    assert out['PG Stability'] == pytest.approx(0.7)
    assert out['PG avg'] == pytest.approx((0.7 + 0.2) / 2)       # category means, not DMS count
    assert out['PG DMS mean'] == pytest.approx((0.8 + 0.6 + 0.2) / 3)


def test_proteingym_summary_refuses_a_missing_dms(tmp_path):
    pd.DataFrame({'DMS_id': ['d1', 'd2'], 'category': ['Stability', 'Binding']}).to_csv(tmp_path / 'cats.csv', index=False)
    pd.DataFrame({'DMS_id': ['d1'], 'rho': [0.5]}).to_csv(tmp_path / 'summary.csv', index=False)
    with pytest.raises(SystemExit):
        S.pg_summary(str(tmp_path / 'summary.csv'), 'rho', str(tmp_path / 'cats.csv'))
