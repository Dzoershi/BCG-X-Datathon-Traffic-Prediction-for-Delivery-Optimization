"""Freeze guardrails and outage isolation; no network or model weights."""
import json

import numpy as np
import pandas as pd
import pytest

from traffic_prediction import baseline_freeze as freeze
from traffic_prediction.constants import CUTOFF
from traffic_prediction.data import dataset, traffic
from traffic_prediction.models.baselines import DayTypeProfileBaseline


def test_frozen_inputs_reject_tampering_and_wrong_environment(tmp_path):
    data = pd.DataFrame({'datetime': pd.DatetimeIndex([pd.Timestamp('2024-01-01'), CUTOFF]), 'x': [1., 2.]})
    data.to_parquet(tmp_path / 'traffic.parquet')
    pd.DataFrame({'x': [1]}).to_parquet(tmp_path / 'calendar.parquet')
    manifest = dict(files={p.name: traffic.file_sha256(p) for p in tmp_path.iterdir()},
                    code_sha256={}, environment=freeze.environment(), selected_model='day_type',
                    traffic_content_sha256=dataset.content_sha256(data))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    pd.testing.assert_frame_equal(freeze.load_frozen(tmp_path)[0], data)
    manifest['code_sha256'] = {'src/traffic_prediction/models/baselines.py': 'wrong'}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='code checksum'):
        freeze.load_frozen(tmp_path)
    manifest['code_sha256'] = {}
    manifest['environment']['python'] = 'wrong'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='environment'):
        freeze.load_frozen(tmp_path)
    manifest['environment'] = freeze.environment()
    path.write_text(json.dumps(manifest))
    (tmp_path / 'traffic.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        freeze.load_frozen(tmp_path)


def test_outage_never_reads_hidden_target_or_mutates_history():
    times = pd.date_range('2025-01-01', '2025-04-03 23:00', freq='h')
    future = pd.date_range('2025-04-05', periods=120, freq='h')
    all_times = times.union(future)
    calendar = pd.DataFrame({'datetime': all_times, 'day_of_week': all_times.dayofweek, 'is_holiday': False})
    history = pd.DataFrame({'arc': 'A', 'role': 'evalue', 'qualite_mesure': 'ok', 'datetime': times,
                            'debit_horaire': 10. + times.hour, 'taux_occupation': 2.})
    original = history.copy(deep=True)
    first = freeze.MaskedBaseline(DayTypeProfileBaseline, calendar, 'A', 28).predict(history, future)
    pd.testing.assert_frame_equal(history, original)
    hidden = history.datetime.ge('2025-03-07')
    history.loc[hidden, ['debit_horaire', 'taux_occupation']] = [1e9, 99.]
    second = freeze.MaskedBaseline(DayTypeProfileBaseline, calendar, 'A', 28).predict(history, future)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first[['debit_horaire', 'taux_occupation']]).all().all()


def test_run_requires_expected_evidence_before_writing_outputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        freeze.run(tmp_path / 'missing-inputs', tmp_path / 'outputs')
    assert not (tmp_path / 'outputs').exists()
