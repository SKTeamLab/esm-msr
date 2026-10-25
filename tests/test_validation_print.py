"""The zero-shot validation is printed once, not once per protein (Lightning repeats epoch-level metrics under every dataloader)."""
import torch

from esm_msr.training import format_validation_results


def results(n_proteins=4):
    """What trainer.validate returns with one dataloader per protein: every dict carries every epoch-end metric."""
    shared = {f'val_rho_combined/P{i}': 0.5 + 0.01 * i for i in range(n_proteins)}
    shared.update({f'val_rmse_combined/P{i}': 1.0 + 0.1 * i for i in range(n_proteins)})
    shared.update({'val_rho_combined_avg': torch.tensor(0.515), 'val_epi_cell_rank_epi': 0.12})
    return [dict(shared) for _ in range(n_proteins)]


def test_each_protein_and_each_metric_appears_once():
    text = format_validation_results(results(4))
    for i in range(4):
        assert sum(line.startswith(f'P{i} ') for line in text.splitlines()) == 1
    assert text.count('val_rho_combined_avg') == 1
    assert text.count('val_epi_cell_rank_epi') == 1
    assert '0.5150' in text and '0.1200' in text


def test_per_protein_metrics_are_columns_of_one_table():
    lines = format_validation_results(results(3)).splitlines()
    header = lines[0].split()
    assert header == ['protein', 'val_rho_combined', 'val_rmse_combined']
    assert lines[1].split() == ['P0', '0.5000', '1.0000']


def test_lightning_dataloader_suffixes_are_merged():
    r = [{'val_x/dataloader_idx_0': 1.0}, {'val_x/dataloader_idx_1': 1.0}]
    assert format_validation_results(r).splitlines() == ['val_x  1.0000']


def test_nothing_logged_prints_nothing():
    assert format_validation_results([]) == ''
    assert format_validation_results(None) == ''
