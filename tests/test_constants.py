"""Submission dates are defined once, in traffic_prediction.constants (issue #46)."""

import ast
import re
from pathlib import Path

import pandas as pd

from traffic_prediction import constants, predict
from traffic_prediction.data import calendar_events, disruptions

SOURCE = Path(constants.__file__).parent
SUBMISSION_DATE = re.compile(r"2026-11-(0[5-9]|1[01])")


def docstrings(tree: ast.AST) -> set[int]:
    """Ids of the docstring nodes of the module, its classes and functions."""
    nodes = [tree, *(node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))]
    return {id(node.body[0].value) for node in nodes
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}


def test_no_submission_date_hard_coded_outside_constants():
    found = []
    for path in sorted(SOURCE.rglob("*.py")):
        if path == Path(constants.__file__):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = docstrings(tree)
        found += [f"{path.relative_to(SOURCE)}:{node.lineno}: {node.value!r}" for node in ast.walk(tree)
                  if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
                  and SUBMISSION_DATE.search(node.value)]
    assert not found, "Use traffic_prediction.constants instead:\n" + "\n".join(found)


def test_window_shape():
    assert constants.CUTOFF == pd.Timestamp("2026-11-05 23:00") and constants.CUTOFF.dayofweek == 3
    assert len(constants.TARGET_TIMES) == 120 and constants.TARGET_TIMES.freq == "h"
    assert constants.TARGET_TIMES[0] == constants.CUTOFF + pd.Timedelta(hours=25)  # Friday is skipped.
    assert (constants.TARGET_TIMES[0].dayofweek, constants.TARGET_TIMES[-1].dayofweek) == (5, 2)
    assert constants.TEST_START == "2026-11-07" and constants.PREDICTION_END == "2026-11-11"


def test_every_module_uses_the_same_window():
    assert predict.CUTOFF is constants.CUTOFF
    assert predict.TARGET_TIMES is constants.TARGET_TIMES
    assert calendar_events.PREDICTION_END is constants.PREDICTION_END
    assert disruptions.TEST_START is constants.TEST_START
