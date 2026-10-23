"""Structure files are found by their original Zenodo names, and benchmark tables built on another machine still resolve."""
import pandas as pd

from esm_msr import data, preprocess
from esm_msr.preprocess_megascale import structure_file_stems


def test_repo_root_is_the_repo_not_src():
    assert (preprocess.REPO_ROOT / 'pyproject.toml').exists()


def test_existing_path_is_kept(tmp_path):
    f = tmp_path / '1ABC_A_processed.pdb'
    f.write_text('')
    assert data.resolve_structure_path(str(f)) == str(f)


def test_missing_absolute_path_falls_back_to_structures_dir(tmp_path, monkeypatch):
    (tmp_path / '1ABC_A_processed.pdb').write_text('')
    monkeypatch.setattr(data, 'STRUCTURES_DIR', str(tmp_path))
    assert data.resolve_structure_path('/home/someone/else/1ABC_A_processed.pdb') == str(tmp_path / '1ABC_A_processed.pdb')


def test_unresolvable_path_is_returned_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'STRUCTURES_DIR', str(tmp_path))
    assert data.resolve_structure_path('/nowhere/X.pdb') == '/nowhere/X.pdb'


def _stems(tmp_path, *files):
    for f in files:
        (tmp_path / f).write_text('')
    s = pd.Series(['EA:run2_0290_0001', 'v2R31S:R32S_2N5D', '1ABC'])
    return list(structure_file_stems(str(tmp_path) + '/', s))


def test_original_colon_name_is_used_when_that_file_exists(tmp_path):
    out = _stems(tmp_path, 'EA:run2_0290_0001.pdb', 'v2R31S:R32S_2N5D.pdb', '1ABC.pdb')
    assert out == ['EA:run2_0290_0001', 'v2R31S:R32S_2N5D', '1ABC']


def test_underscore_name_is_used_when_only_that_file_exists(tmp_path):
    """A folder prepared before original names were used (the cluster's copy): no colon files at all."""
    out = _stems(tmp_path, 'EA_run2_0290_0001.pdb', 'v2R31S_R32S_2N5D.pdb', '1ABC.pdb')
    assert out == ['EA_run2_0290_0001', 'v2R31S_R32S_2N5D', '1ABC']


def test_colon_name_wins_when_both_exist(tmp_path):
    out = _stems(tmp_path, 'v2R31S:R32S_2N5D.pdb', 'v2R31S_R32S_2N5D.pdb')
    assert out[1] == 'v2R31S:R32S_2N5D'


def test_a_missing_structure_is_reported_under_the_original_name(tmp_path):
    assert _stems(tmp_path) == ['EA:run2_0290_0001', 'v2R31S:R32S_2N5D', '1ABC']
