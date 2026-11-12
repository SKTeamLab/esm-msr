"""--comet_offline writes an archive for `comet upload`; Lightning's own online=False would silently log nothing."""
import pytest

pytest.importorskip('comet_ml')

from esm_msr.training import OfflineCometLogger


def test_offline_logger_is_a_live_offline_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv('COMET_OFFLINE_DIRECTORY', str(tmp_path))
    logger = OfflineCometLogger(api_key='unused-offline', project='p', name='n')
    experiment = logger.experiment
    assert type(experiment).__name__ == 'OfflineExperiment'
    assert not experiment.disabled
    experiment.add_tags(['authoritative'])
    logger.log_metrics({'x': 1.0}, step=0)
    experiment.end()
    zips = list(tmp_path.glob('*.zip'))
    assert zips
    import zipfile
    with zipfile.ZipFile(zips[0]) as zf:
        assert '"authoritative"' in zf.read('experiment.json').decode('utf-8')
