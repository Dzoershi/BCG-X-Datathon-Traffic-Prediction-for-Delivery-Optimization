"""Check that every module of the package can be imported."""

import importlib

import pytest

MODULE_NAMES = [
    "traffic_prediction",
    "traffic_prediction.data",
    "traffic_prediction.data.traffic",
    "traffic_prediction.data.calendar_events",
    "traffic_prediction.data.weather",
    "traffic_prediction.data.disruptions",
    "traffic_prediction.data.build",
    "traffic_prediction.data.query",
    "traffic_prediction.data.update",
    "traffic_prediction.data.validate",
    "traffic_prediction.models",
    "traffic_prediction.models.base",
    "traffic_prediction.models.baselines",
    "traffic_prediction.models.chronos",
    "traffic_prediction.features",
    "traffic_prediction.train",
    "traffic_prediction.predict",
    "traffic_prediction.evaluate",
]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
