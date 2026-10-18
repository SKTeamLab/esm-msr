"""--ext_val_doubles: an external double-mutant assay scored on its own, never pooled into the hierarchy.

The point of these sets is the CELL level: the Tsuboyama validation split supports only a couple of dozen
position pairs, so its seed SD is the size of every arm effect. An external assay contributes hundreds of
complete blocks - but it is one library, often on a different (fitness) scale, so pooling it would both swamp
and corrupt the pooled levels. These tests pin that separation and the rank-only metric choice.
"""
import sys
import types
from collections import defaultdict

import numpy as np
import pytest

from esm_msr import epi_metrics, training

CLS = training.ESM3EpistasisLightningModule
ROWS, COLS = list('ACDEF'), list('GHIKL')


def block_library(seed=0, n_pairs=6, scale=1.0, offset=0.0):
    """n_pairs complete 5x5 matrices with their singles, in the shape _log_ext_epistasis takes.

    ``scale``/``offset`` apply a MONOTONE relabelling of the measured score: every metric logged for an external
    set must be invariant to it.
    """
    rng = np.random.default_rng(seed)
    keys, st, y, pred = [], [], [], []
    for p in range(n_pairs):
        i, j = 10 * p + 1, 10 * p + 5
        sa = {a: rng.normal() for a in ROWS}
        sb = {b: rng.normal() for b in COLS}
        inter = rng.normal(size=(len(ROWS), len(COLS)))
        for a in ROWS:
            keys.append((('L', i, a),)); st.append('single'); y.append(sa[a]); pred.append(sa[a])
        for b in COLS:
            keys.append((('L', j, b),)); st.append('single'); y.append(sb[b]); pred.append(sb[b])
        for ia, a in enumerate(ROWS):
            for ib, b in enumerate(COLS):
                keys.append((('L', i, a), ('L', j, b))); st.append('double')
                y.append(sa[a] + sb[b] + inter[ia, ib])
                pred.append(sa[a] + sb[b] + inter[ia, ib])      # a perfect predictor, on the LATENT scale
    n = len(y)
    y = np.array(y) * scale + offset
    singles = {k[0]: v for k, s, v in zip(keys, st, y) if s == 'single'}
    dddG = np.array([v - singles[k[0]] - singles[k[1]] if s == 'double' else np.nan for k, s, v in zip(keys, st, y)])
    return {'mut_key': keys, 'subset_type': st, 'cens': np.zeros(n, int), 'cens_src': np.zeros(n, int),
            'ddG': y, 'dddG': dddG, 'dG_wt': np.full(n, 2.0), 'y_floor': np.full(n, -0.14),
            'wt': np.array(pred), 'mt': np.array(pred), 'comb': np.array(pred)}


def stub():
    s = types.SimpleNamespace(logged={}, link_head=None)
    s.log = lambda k, v, **kw: s.logged.__setitem__(k, float(v))
    s.EXT_METRICS = CLS.EXT_METRICS
    s._ext_standardise = CLS._ext_standardise
    return s


def test_only_the_rank_based_levels_are_logged():
    s = stub()
    out = CLS._log_ext_epistasis(s, 'ext_assay', block_library())
    kinds = {k.split('/')[0].replace('val_extepi_', '').rsplit('_', 1)[0] for k in s.logged}
    assert kinds <= set(CLS.EXT_METRICS) | {'n'}      # plus the epi_n_* counts, which are diagnostics
    assert 'cell_rank' in kinds
    # the magnitude levels are the ones a fitness scale would corrupt, and they must be absent
    assert not any(x in k for k in s.logged for x in ('global_', 'pair_rho', 'line_rho', 'cell_mag', 'beyond'))
    assert all(k.startswith('extepi_') and k.endswith('_ext_assay') for k in out)


def test_the_loader_name_is_in_every_logged_key():
    s = stub()
    CLS._log_ext_epistasis(s, 'ext_assay', block_library())
    assert all(k.endswith('/ext_assay') for k in s.logged), s.logged


def test_a_perfect_predictor_scores_at_the_top():
    s = stub()
    CLS._log_ext_epistasis(s, 'ext_assay', block_library())
    assert s.logged['val_extepi_cell_rank_mt/ext_assay'] > 0.95
    assert s.logged['val_extepi_cell_flipacc_mt/ext_assay'] > 0.95


def test_the_metrics_survive_a_monotone_relabelling_of_the_score():
    """Why a fitness-scale assay is admissible at all: these levels are within-column RANK statistics."""
    a, b = stub(), stub()
    CLS._log_ext_epistasis(a, 'ext_assay', block_library(scale=1.0))
    CLS._log_ext_epistasis(b, 'ext_assay', block_library(scale=3.7, offset=-2.0))
    assert set(a.logged) == set(b.logged)
    for k in a.logged:
        assert a.logged[k] == pytest.approx(b.logged[k], abs=1e-9), k


def test_standardising_is_what_makes_the_flip_threshold_comparable():
    """FLIP_DELTA is an absolute 0.6 kcal/mol, so without this the relabelled assay selects a different set of
    reversals entirely (measured: 121 against 277)."""
    y = np.array([1.0, 2.0, 3.0, 10.0])
    z = CLS._ext_standardise(y * 3.7 - 2.0)
    np.testing.assert_allclose(z, CLS._ext_standardise(y), rtol=1e-9)
    assert abs(np.median(z)) < 1e-9
    assert CLS._ext_standardise(np.array([5.0, 5.0, 5.0])).tolist() == [5.0, 5.0, 5.0]   # no spread: left alone


def test_a_failure_is_warned_about_and_never_raises():
    s = stub()
    assert CLS._log_ext_epistasis(s, 'ext_assay', {'mut_key': [None, None], 'subset_type': ['single', 'single']}) == {}
    assert s.logged == {}


def test_an_ext_loader_is_kept_out_of_the_pooled_hierarchy():
    """The separation this whole feature rests on: ext_ items must not reach the pooled epistasis table."""
    src = open('src/esm_msr/training.py').read()
    assert "if name.startswith(EXT_PREFIX):" in src
    assert training.EXT_PREFIX == 'ext_'
    # and the pooled scorer is still the one fed by the non-ext loaders
    assert src.index("if name.startswith(EXT_PREFIX):") < src.index("pooled['mut_key'].extend(keys)")


def test_the_flag_is_repeatable_and_off_by_default():
    from esm_msr import config
    argv = sys.argv
    base = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
    try:
        sys.argv = base
        assert config.parse_arguments().ext_val_doubles is None
        sys.argv = base + ['--ext_val_doubles', 'a.csv', '--ext_val_doubles', 'b.csv']
        assert config.parse_arguments().ext_val_doubles == ['a.csv', 'b.csv']
    finally:
        sys.argv = argv


def test_a_missing_file_is_skipped_rather_than_fatal():
    from esm_msr.preprocess_megascale import load_external_doubles
    loaders, names = load_external_doubles(['/nonexistent/x.csv'], None, None, '/tmp', False)
    assert (loaders, names) == ([], [])
