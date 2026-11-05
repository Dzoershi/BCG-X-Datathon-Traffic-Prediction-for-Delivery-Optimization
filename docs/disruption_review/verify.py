"""Independent report checks; run from the repository root with the frozen local inputs."""
import json, math
from pathlib import Path
import pandas as pd
from traffic_prediction.data.traffic import file_sha256

root = Path('outputs/metrics/disruption_57')
run = json.loads((root/'run.json').read_text(encoding='utf-8'))
snapshot = Path('data/raw/disruption_57')/run['snapshot']
manifest = json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
for source in manifest['sources'] + [s for s in manifest['documents'] if s['status']=='captured']:
    assert file_sha256(snapshot/source['file']) == source['sha256']
detail = pd.read_csv(root/'historical_predictions.csv', parse_dates=['datetime','reference_datetime','cutoff'])
raw = pd.read_parquet('outputs/metrics/quality_52/observations.parquet').set_index(['arc','datetime'])
original = pd.read_parquet('outputs/metrics/complexity_55/hourly_predictions.parquet').set_index(
    ['model','scenario','cutoff','arc','datetime','variable'])
adjusted = 0
for row in detail.itertuples():
    cached = original.loc[(row.model, row.scenario, row.cutoff, row.arc, row.datetime, row.variable)]
    assert row.truth_valid == cached.truth_valid
    assert math.isclose(row.prediction, cached.prediction, rel_tol=1e-12, abs_tol=1e-10)
    if row.adjusted:
        adjusted += 1
        assert row.reference_datetime < row.cutoff
        assert row.datetime.day == 11 and 8 <= row.datetime.hour <= 14
        short = 'q' if row.variable == 'debit_horaire' else 'k'
        ref = raw.loc[(row.arc, row.reference_datetime)]
        assert ref[f'{short}_status'] == 'valid' and ref.legacy_quality == 'ok'
        assert math.isclose(row.adjusted_prediction, ref[f'{short}_value'], abs_tol=1e-10)
    else:
        assert row.adjusted_prediction == row.prediction
scores = pd.read_csv(root/'historical_scores.csv')
for row in scores.itertuples():
    sample = detail.loc[detail.model.eq(row.model) & detail.variable.eq(row.variable)]
    if row.scope != 'all_arcs_window':
        sample = sample.loc[sample.arc.eq('Champs-Elysées')]
    if row.scope == 'champs_nov11':
        sample = sample.loc[sample.datetime.dt.day.eq(11)]
    if row.scope == 'champs_closure':
        sample = sample.loc[sample.adjusted]
    assert len(sample) == row.n_expected and int(sample.adjusted.sum()) == row.n_adjusted
    sample = sample.loc[sample.truth_valid]
    field = 'prediction' if row.policy == 'raw_model' else 'adjusted_prediction'
    squared = [(p-y)**2 for p,y in zip(sample[field],sample.truth)]
    assert len(sample) == row.n_scored
    assert math.isclose(math.sqrt(math.fsum(squared)/len(squared)), row.rmse, rel_tol=1e-10)
hours = pd.read_csv(root/'candidate_hours.csv',parse_dates=['datetime'])
counts = pd.read_csv(root/'candidate_hour_counts.csv',parse_dates=['datetime'])
assert counts.n_candidates.sum() == len(hours)
for row in counts.itertuples():
    assert len(hours.loc[hours.arc.eq(row.arc) & hours.datetime.eq(row.datetime) & hours.kind.eq(row.kind)
                        & hours.status.eq(row.status)]) == row.n_candidates
assert hours.action.eq('none').all()
verification = {'historical_prediction_records_checked':len(detail),'historical_score_cells_checked':len(scores),
    'adjusted_records':adjusted,'candidate_hour_rows':len(hours),'hour_count_cells_checked':len(counts),
    'raw_snapshot_checksums_verified':True,'final_test_observations_used':False,
    'method':'Independent Python math.fsum RMSE and per-row frozen-prediction/reference/eligibility checks; per-hour candidate recounts.',
    'historical_predictions_sha256':file_sha256(root/'historical_predictions.csv')}
print(json.dumps(verification,indent=2))
