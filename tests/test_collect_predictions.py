"""analysis_scripts/collect_predictions.py: the naming rules that decide where each collected prediction file goes (no cluster needed)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis_scripts'))
import collect_predictions as C


@pytest.mark.parametrize('raw,want', [('PTMUL', 'ptmul'), ('s571', 's571'), ('splits_oct06_capped-val', 'tsuboyama_capped_val'),
                                      ('splits_oct06_capped-test', 'tsuboyama_capped_test'), ('hyperopt_splits-val', 'tsuboyama_hyperopt_val'),
                                      ('GB1_Wu_2016_binding_domain', 'GB1_Wu_2016_binding_domain')])
def test_dataset_names_are_the_ones_benchmark_matrix_uses(raw, want):
    assert C.ds_name(raw) == want


@pytest.mark.parametrize('tok,want', [('1', 'sigma1'), ('1.0', 'sigma1'), ('0.0', 'sigma0'), ('0.25', 'sigma0.25'), ('0.5', 'sigma0.5')])
def test_both_spellings_of_unit_sigma_collapse_to_one_directory(tok, want):
    assert C.sigma_dir(tok) == want


def test_gamut_regex_takes_unmasked_prediction_files_only():
    ok = 'epoch=07-val_rho_combined_avg=0.839.ckpt_sigma1.0_unmasked_predictions.csv'
    assert C.GAMUT_RX.match(ok).group('sig') == '1.0'
    assert C.GAMUT_RX.match('epoch=04-val_rho_combined_avg=0.855.ckpt_sigma0.25_unmasked_predictions.csv').group('sig') == '0.25'
    for bad in ('last.ckpt_sigma1_wt-unmasked_mt-independent_predictions.csv',                   # another mask strategy: not a table input
                'epoch=07.ckpt_sigma1_unmasked_predictions.csv.pre_gb1_fix',                    # superseded by the GB1 patch
                'epoch=07.ckpt_sigma1_unmasked_Combined.csv'):                                  # a stats file
        assert not C.GAMUT_RX.match(bad)


def test_same_file_needs_size_and_mtime_to_agree(tmp_path):
    f = tmp_path / 'x.csv'
    f.write_text('abc')
    st = f.stat()
    assert C.same_file(f, st.st_size, st.st_mtime)
    assert not C.same_file(f, st.st_size + 1, st.st_mtime)
    assert not C.same_file(f, st.st_size, st.st_mtime + 60)           # a file the cluster has since rewritten is not "already here"
    assert not C.same_file(tmp_path / 'missing.csv', 3, 0)


def test_rerunning_skips_a_destination_that_already_holds_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'ROOT', tmp_path)
    src = tmp_path / 'old' / 'a.csv'
    src.parent.mkdir()
    src.write_text('x' * 10)
    plan = C.Plan()
    plan.add('new/a.csv', '/home/sreeves/a.csv', 10, 0.0, local=str(src))
    C.run(plan)
    assert (tmp_path / 'new' / 'a.csv').read_text() == 'x' * 10 and not src.exists()      # moved, not copied
    C.run(plan)                                                                           # second run: nothing to do, nothing breaks
    assert (tmp_path / 'new' / 'a.csv').exists()


def test_destination_collisions_are_refused(monkeypatch):
    def fake(plan):
        plan.add('same/dst.csv', '/h/a', 1, 0.0)
        plan.add('same/dst.csv', '/h/b', 1, 0.0)
    monkeypatch.setattr(C, 'plan_msr', fake)
    with pytest.raises(SystemExit):
        C.build_plan(['msr'])
