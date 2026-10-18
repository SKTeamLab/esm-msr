"""Structure files are found by their original Zenodo names, and benchmark tables built on another machine still resolve."""
import pandas as pd

from esm_msr import data, preprocess


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
