# Weekly traffic planner and presentation figures

Prototype shown on slide 5 of deliverable 2: a self-contained HTML page built from the
submitted forecast. It opens in any browser, offline.

```powershell
python scripts/build_planner.py            # docs/planner/X-Tra_Delivery_traffic_planner.html
python scripts/slot_choice_eval.py         # slide 6 "Impact" figures, docs/planner/slot_choice_results.csv
python scripts/presentation_figures.py     # docs/presentation/*.png (slide 7 and appendix B charts)
```

`slot_choice_eval.py` needs the backtest forecasts first:
`python scripts/run_local_backtests.py baseline_holiday_aware timestral_5m lightgbm_raw`.

## What the planner shows

- One view per road and one for all three roads together. In the combined view, each road
  is divided by its own average daytime occupancy, then the three are averaged, so the busy
  Champs-Élysées does not dominate.
- The clearest 4- or 8-hour day shift per day (the brief allows shifts of 4 to 8 hours),
  with a badge: clear, moderate or busy relative to the week's daytime hours.
- When a night shift would pay off: a night hour costs €14 instead of €10, so night is
  cheaper per parcel only above 1.4 times the day productivity.
- How far to trust it: figures from the 8 backtest weeks.

## Rules and limits

- Labels mark the end of the hour (`08:00` = 07:00 to 08:00); the page re-keys each value
  on its starting hour.
- Champs-Élysées hours closed for the 11 November ceremony are never recommended.
- One shift per day is a prototype simplification; combining shifts and drivers is roadmap step 3.
- Clearer traffic at night is necessary, not sufficient: parcels per hour must be measured in the pilot.

## Slot-choice results (8 backtest weeks, Convention and Saint-Pères)

About 40 day-road cases per target. Forecast errors fall by 10% (flow) and 16% (occupancy);
busy hours flagged and clearest shift found stay within noise of the last-week baseline.
