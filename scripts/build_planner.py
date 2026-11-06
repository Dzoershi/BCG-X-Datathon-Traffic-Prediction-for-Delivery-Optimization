"""Build the weekly traffic planner (HTML prototype) from the submitted forecast.

From the repository root::

    python scripts/build_planner.py
    python scripts/build_planner.py --submission outputs/submissions/output_X-Tra_Delivery.csv

The page is self-contained (data inlined, no network) and opens in any browser.
It reads only the submission CSV: labels mark the end of the hour (Open Data Paris),
so the page re-keys each value on the hour it starts. Champs-Elysées hours closed
for the 11 November ceremony are excluded from every recommendation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "planner_template.html"


def build(submission: Path, output: Path) -> Path:
    df = pd.read_csv(submission, sep=";", index_col=0)
    expected = {"arc", "datetime", "debit_horaire", "taux_occupation"}
    if missing := expected - set(df.columns):
        raise ValueError(f"Submission is missing columns {sorted(missing)}")
    if len(df) != 360:
        raise ValueError(f"Expected 360 rows, got {len(df)}")
    data = df[["arc", "datetime", "debit_horaire", "taux_occupation"]].to_dict("records")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--submission", type=Path, default=ROOT / "outputs/submissions/output_X-Tra_Delivery.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/planner/X-Tra_Delivery_traffic_planner.html")
    args = parser.parse_args()
    print(f"Planner written to {build(args.submission, args.output)}")


if __name__ == "__main__":
    main()
