"""Submission dates and export precision, defined once for the whole pipeline (issue #46).

Timezone-free Paris local hours with end-of-hour labels, as in the processed data
(convention to be confirmed with BCG X). Every other module imports these names.
"""

import pandas as pd

CUTOFF = pd.Timestamp("2026-11-05 23:00")  # Last observed hour: Thursday 5 November.
TARGET_START = pd.Timestamp("2026-11-07 00:00")  # Saturday; Friday 6 November is not scored.
TARGET_END = pd.Timestamp("2026-11-11 23:00")  # Wednesday 11 November, a public holiday.
TARGET_TIMES = pd.date_range(TARGET_START, TARGET_END, freq="h")  # 120 submitted hours per arc.
TEST_START = f"{TARGET_START:%Y-%m-%d}"  # First Paris day of the target window.
PREDICTION_END = f"{TARGET_END:%Y-%m-%d}"  # Last Paris day of calendar, weather and disruptions.
SUBMISSION_DECIMALS = 2
