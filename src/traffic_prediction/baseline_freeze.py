"""Offline, fingerprint-checked baseline reference and submission for issue #54."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from traffic_prediction import evaluate, validation
from traffic_prediction.constants import CUTOFF, TARGET_TIMES
from traffic_prediction.data import dataset, traffic
from traffic_prediction.data.audit import _json
from traffic_prediction.models.baselines import (
    DayTypeProfileBaseline, FourWeekMeanBaseline, HolidayAwarePreviousWeekBaseline, PreviousWeekBaseline,
)
from traffic_prediction.predict import build_submission, export_submission

ROOT = traffic.REPO_ROOT
DEFAULT_INPUTS = ROOT / 'docs/baseline_freeze/inputs'
FACTORIES = {'day_type': DayTypeProfileBaseline, 'holiday_aware': HolidayAwarePreviousWeekBaseline,
             'previous_week': PreviousWeekBaseline, 'four_week': FourWeekMeanBaseline}
PACKAGES = ('numpy', 'pandas', 'pyarrow', 'duckdb', 'PyYAML')


def environment() -> dict:
    return {'python': platform.python_version(), **{name: version(name) for name in PACKAGES}}


def load_frozen(inputs: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((inputs / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if traffic.file_sha256(inputs / name) != digest:
            raise ValueError(f'Frozen input checksum mismatch: {name}')
    for name, digest in manifest['code_sha256'].items():
        if traffic.file_sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen code checksum mismatch: {name}; review and explicitly re-freeze')
    if environment() != manifest['environment']:
        raise ValueError('Use the recorded Python and requirements-baseline54.txt environment')
    data, calendar = (pd.read_parquet(inputs / name) for name in ('traffic.parquet', 'calendar.parquet'))
    if data.datetime.max() != CUTOFF or data.datetime.min() != pd.Timestamp('2024-01-01'):
        raise ValueError('Frozen traffic must cover the declared period through the official cutoff')
    if dataset.content_sha256(data) != manifest['traffic_content_sha256']:
        raise ValueError('Frozen traffic content mismatch')
    if manifest['selected_model'] not in FACTORIES:
        raise ValueError('Unknown frozen baseline')
    return data, calendar, manifest


class MaskedBaseline:
    """Hide target history before fitting, retain untouched truth in the evaluator."""
    def __init__(self, factory, calendar, target, days):
        self.model, self.target, self.days = factory(calendar), target, days

    def predict(self, history, target_times):
        history = history.copy(deep=True)
        start = history.datetime.max().normalize() + pd.Timedelta(days=1-self.days)
        hidden = history.arc.eq(self.target) & history.datetime.ge(start)
        history.loc[hidden, list(evaluate.TARGETS)] = np.nan
        history.loc[hidden, 'qualite_mesure'] = 'absente'
        result = self.model.predict(history, target_times)
        self.sources_ = self.model.sources_
        return result


def run(inputs: Path, output: Path, *, record_expected: bool = False) -> dict:
    expected = inputs / 'expected.json'
    if record_expected and expected.exists():
        raise ValueError('Expected artifacts already frozen; use a new snapshot directory')
    frozen = None if record_expected else json.loads(expected.read_text())
    data, calendar, manifest = load_frozen(inputs)
    cutoffs = evaluate.load_cutoffs()
    models = {name: factory(calendar) for name, factory in FACTORIES.items()}
    frames = [validation.run_candidates(models, data, cutoffs, 'development'),
              validation.run_candidates(models, data, [validation.RECENT_CUTOFF], 'recent_check')]
    durations = [28, 90, 180, 365]
    proxies, skipped = validation.run_outages(data, calendar, cutoffs, durations)
    frames.append(proxies)
    target = next(arc.name for arc in traffic.load_arcs() if arc.iu_ac == 4264)
    for days in durations:
        scenario = f'outage_{days}d'
        eligible = proxies.loc[proxies.scenario.eq(scenario), 'cutoff'].drop_duplicates().tolist()
        for name, factory in FACTORIES.items():
            for cutoff in eligible:
                model = MaskedBaseline(factory, calendar, target, days)
                result = evaluate.backtest(model, data, [cutoff], experiments_path=None)
                frames.append(result.hourly.loc[result.hourly.arc.eq(target)].assign(
                    model=name, phase='development', scenario=scenario))
    hourly = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output / 'hourly_predictions.parquet', index=False, compression='zstd')
    validation.save_scores(hourly, output)
    verified = validation.verify_metrics(output)
    selected = manifest['selected_model']
    tables, sources = [], []
    for repeat in (1, 2):
        model = FACTORIES[selected](calendar)
        # The reference evaluates the baseline policy only; no ceremony correction.
        table = build_submission(data, model)
        path = export_submission(table, 'baseline54', output / f'repeat_{repeat}')
        tables.append(path.read_bytes())
        sources.append(model.sources_.sort_index())
    if tables[0] != tables[1] or not sources[0].equals(sources[1]):
        raise ValueError('Repeated exports or prediction sources differ')
    (output / 'output_baseline54.csv').write_bytes(tables[0])
    provenance = sources[0].rename('forecast_source').reset_index()
    provenance.to_csv(output / 'submission_sources.csv', index=False, lineterminator='\n')
    report = {'selected_model': selected, 'cutoff': str(CUTOFF), 'target_start': str(TARGET_TIMES.min()),
              'target_end': str(TARGET_TIMES.max()), 'submission_rows': len(table),
              'traffic_content_sha256': manifest['traffic_content_sha256'], 'environment': environment(),
              'identical_repeat_csv_bytes': True, 'identical_repeat_sources': True,
              'submission_sha256': traffic.file_sha256(output / 'output_baseline54.csv'),
              'cutoffs': [str(c) for c in cutoffs], 'outage_days': durations, 'skipped_outages': skipped,
              'verification': verified,
              'artifact_sha256': {p.name: traffic.file_sha256(p) for p in sorted(output.iterdir())
                                  if p.suffix in ('.csv', '.parquet')}}
    if frozen is not None and report['artifact_sha256'] != frozen['artifact_sha256']:
        raise ValueError('Frozen scores, hourly predictions, provenance or submission bytes changed')
    if record_expected:
        _json(expected, {'artifact_sha256': report['artifact_sha256']})
    _json(output / 'run.json', report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/metrics/baseline_54')
    parser.add_argument('--record-expected', action='store_true', help='Explicitly bootstrap a NEW reviewed freeze only')
    args = parser.parse_args()
    run(args.inputs, args.output_dir, record_expected=args.record_expected)


if __name__ == '__main__':
    main()
