"""analysis_scripts/score_me_proteingym.py: the sign, the join and the summary rule of ME's ProteinGym rows (toy data, no cluster)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis_scripts'))
import score_me_proteingym as S


def _labels():
    return pd.DataFrame({'mut_type_renumbered': ['A1G', 'A1C', 'A1D', 'A1:C2D'], 'DMS_score': [0.1, 0.4, 0.2, 0.9]})


def test_upstream_files_are_destabilizing_positive_so_the_prediction_is_minus_ddg_pred():
    res = pd.DataFrame({'mut_info': ['A1G', 'A1C', 'A1D', 'A1:C2D'], 'ddg_pred': [-0.1, -0.4, -0.2, -0.9]})      # a perfect model, ddg_pred = -score
    assert S.score_assay(res, _labels()) == pytest.approx(1.0)
    res['ddg_pred'] = -res['ddg_pred']
    assert S.score_assay(res, _labels()) == pytest.approx(-1.0)


def test_rows_without_a_prediction_are_dropped_and_extra_prediction_rows_ignored():
    res = pd.DataFrame({'mut_info': ['A1G', 'A1C', 'A1D', 'Z9Z'], 'ddg_pred': [-0.1, -0.4, -0.2, 5.0]})          # the multi-mutant has none; Z9Z is not a label
    assert S.score_assay(res, _labels()) == pytest.approx(1.0)


def test_labelled_files_read_pred_combined():
    res = pd.DataFrame({'pred_combined': [0.1, np.nan, 0.3], 'DMS_score': [1.0, 2.0, 3.0]})
    assert S.score_assay(res, None) == pytest.approx(1.0)


def test_summary_is_the_mean_of_category_means_over_the_213_not_the_dropped_four():
    ids = [f'D{i}' for i in range(6)] + sorted(S.PG_DROP)
    cat = pd.Series(['Activity', 'Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability'] + ['Stability'] * 4, index=ids)
    per = pd.Series([0.2, 0.4, 0.5, 0.6, 0.7, 0.8] + [0.0] * 4, index=ids)
    out = S.summarise(per, cat)
    assert out['Activity'] == pytest.approx(0.3) and out['Stability'] == pytest.approx(0.8)
    assert abs(out['PG avg'] - np.mean([0.3, 0.5, 0.6, 0.7, 0.8])) < 1e-12 and abs(out['DMS mean'] - 0.5333333333333333) < 1e-12 and out['n'] == 6
