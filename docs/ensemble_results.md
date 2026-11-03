# Résultats des combinaisons de modèles (#32)

Généré par `scripts/evaluate_combinations.py` le 2026-11-03 09:58 UTC ; ne pas modifier à la main. Toutes les lignes, avec plus de décimales : `docs/ensemble_results.csv`.

## Protocole

- Prévisions horaires des 8 semaines de backtest de `configs/baseline.yaml`, rejouées par `scripts/run_local_backtests.py` ; aucun modèle n'est réentraîné ici.
- Données : dataset #22, empreinte de contenu `c646f8ba`, fingerprint `04c96b9a`.
- Score : RMSE globale (*pooled*) sur les heures valides de Convention + Saint-Pères ; débit et occupation sont toujours notés, et les poids toujours choisis, séparément.
- `Par semaine` : RMSE de chaque semaine puis moyenne. `Semaine du rendu` : erreurs par type de jour repondérées sur 48 h de week-end, 48 h ouvrées et 24 h fériées (estimation, pas une mesure).
- **Honnête** : les poids (et pour la validation imbriquée, l'ensemble lui-même) sont choisis sur les 7 autres semaines puis appliqués à la semaine notée, à tour de rôle. Les règles de la doc ont été repérées sur ces mêmes semaines : leurs scores sont optimistes.
- IC 95 % : écart de RMSE globale, bootstrap de 5000 tirages des semaines. En gras : l'intervalle exclut zéro, l'écart résiste au rééchantillonnage des semaines.

## Runs de modèles utilisés

| Run | Lancé (UTC) | Données (sha256) | Durée | Réglages |
|---|---|---|---:|---|
| `baseline_day_type` | 2026-11-03T09:23 | dataset #22 (`c646f8ba`) | 1 s | DayTypeProfileBaseline (default settings) |
| `baseline_holiday_aware` | 2026-11-02T16:06 | dataset #22 (`c646f8ba`) | 1 s | HolidayAwarePreviousWeekBaseline (default settings) |
| `chronos2` | 2026-11-03T09:23 | dataset #22 (`c646f8ba`) | 41 s | Chronos-2 zero-shot (amazon/chronos-2@29ec3766), context 8192 h, covariates ['is_holiday', 'is_school_holiday', 'is_day_before_holiday', 'is_bridge', 'is_sunday_or_holiday', 'day_type'], multivariate, median of quantiles; fallback HolidayAwarePreviousWeekBaseline if fewer than 168 ok hours in the last 28 days |
| `lightgbm_raw` | 2026-11-02T16:06 | dataset #22 (`c646f8ba`) | 15 s | LightGBM global, separate targets, flow=raw; parameters {"deterministic": true, "force_col_wise": true, "importance_type": "gain", "learning_rate": 0.05, "max_depth": 6, "min_child_samples": 30, "n_estimators": 200, "n_jobs": 4, "num_leaves": 31, "objective": "regression", "random_state": 42, "reg_lambda": 1.0, "verbosity": -1}; lags [168, 336, 504, 672] h; weather=lagged; training profiles gap=168 h; fallback HolidayAwarePreviousWeekBaseline if fewer than 168 valid hours in the last 28 days |
| `timestral_209m` | 2026-11-03T09:24 | dataset #22 (`c646f8ba`) | 252 s | Timestral-209M@9874d252, source a534bdbd, context 32768 h, multivariate, calendar ['is_holiday', 'is_school_holiday', 'is_day_before_holiday', 'is_bridge', 'is_sunday_or_holiday', 'day_type'], median, upstream clampsym defaults; HolidayAwarePreviousWeekBaseline fallback if fewer than 168 ok hours in the retained context's last 28 days |
| `timestral_5m` | 2026-11-03T09:24 | dataset #22 (`c646f8ba`) | 15 s | Timestral-5M@bdd20cfc, source a534bdbd, context 32768 h, multivariate, calendar ['is_holiday', 'is_school_holiday', 'is_day_before_holiday', 'is_bridge', 'is_sunday_or_holiday', 'day_type'], median, upstream clampsym defaults; HolidayAwarePreviousWeekBaseline fallback if fewer than 168 ok hours in the retained context's last 28 days |

## Débit (véh/h)

Meilleure stratégie honnête : **baseline_holiday_aware + lightgbm_raw** (87,74). Écarts : stratégie − `baseline_holiday_aware` et stratégie − meilleure honnête.

### Modèles seuls — honnête : sans choix

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| lightgbm_raw | 89,06 | 79,51 | 79,36 | 87,15 | 123,11 | 92,67 | 6/8 | **[−12,80 ; −3,09]** | [−0,59 ; 3,39] |  |
| timestral_209m_qmean | 91,49 | 79,04 | 76,44 | 92,15 | 131,30 | 95,82 | 5/8 | [−14,07 ; 0,12] | [−8,03 ; 12,81] |  |
| timestral_209m | 91,98 | 79,32 | 77,56 | 92,25 | 131,46 | 96,27 | 4/8 | [−14,20 ; 0,91] | [−7,99 ; 13,68] |  |
| baseline_holiday_aware | 97,46 | 85,47 | 94,09 | 84,69 | 142,27 | 102,27 | 0/8 | [0,00 ; 0,00] | **[5,06 ; 13,73]** |  |
| timestral_5m_qmean | 98,19 | 85,40 | 73,39 | 89,45 | 176,71 | 107,70 | 3/8 | [−4,68 ; 7,81] | **[2,37 ; 19,49]** |  |
| timestral_5m | 98,20 | 85,35 | 73,94 | 88,93 | 176,84 | 107,73 | 4/8 | [−4,60 ; 7,89] | **[2,13 ; 19,55]** |  |
| baseline_day_type | 104,33 | 93,83 | 90,41 | 106,92 | 137,47 | 107,80 | 3/8 | [−5,24 ; 21,41] | **[6,59 ; 28,92]** |  |
| chronos2 | 104,37 | 89,20 | 75,36 | 99,34 | 183,81 | 113,91 | 2/8 | [−2,49 ; 14,14] | **[3,49 ; 26,86]** |  |
| chronos2_qmean | 104,63 | 89,50 | 75,67 | 99,92 | 183,45 | 114,08 | 2/8 | [−1,63 ; 13,84] | **[4,17 ; 26,97]** |  |

### Règles de docs/methodology.md (choisies sur ces semaines) — honnête : non

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| C4 C3 + lightgbm flow ferie | 85,90 | 76,31 | 73,94 | 84,69 | 123,11 | 89,93 | 7/8 | **[−15,94 ; −6,47]** | **[−3,32 ; −0,14]** |  |
| C2 C1 + lightgbm flow ferie | 86,44 | 76,76 | 75,36 | 84,69 | 123,11 | 90,40 | 7/8 | **[−15,27 ; −6,11]** | [−2,41 ; 0,02] |  |
| C3 ts5m ouvre / baseline we / ts209m ferie | 87,38 | 77,05 | 73,94 | 84,69 | 131,46 | 92,26 | 6/8 | **[−16,40 ; −3,07]** | [−7,19 ; 7,43] |  |
| C5 chronos ouvre / baseline we / ts209m ferie | 87,92 | 77,53 | 75,36 | 84,69 | 131,46 | 92,72 | 6/8 | **[−15,33 ; −2,71]** | [−5,57 ; 7,91] |  |
| ts209m ouvre+ferie / baseline we | 88,76 | 78,26 | 77,56 | 84,69 | 131,46 | 93,45 | 6/8 | **[−15,62 ; −0,40]** | [−4,94 ; 10,03] |  |
| C1 chronos ouvre / baseline we+ferie | 89,92 | 79,82 | 75,36 | 84,69 | 142,27 | 95,86 | 7/8 | **[−12,98 ; −2,77]** | [−2,27 ; 7,66] |  |

### Ensembles, poids par type de jour — honnête : oui

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| E5 baseline + ts5m + lightgbm [blend] | 88,02 | 77,45 | 74,57 | 83,68 | 135,92 | 93,38 | 7/8 | **[−12,90 ; −5,47]** | [−4,07 ; 4,92] | ouvre: time5m=0.90[0.8-1.0] lighraw=0.10[0.0-0.2]<br>weekend: baseawar=0.51[0.3-0.6] time5m=0.20[0.1-0.5] lighraw=0.29[0.2-0.4]<br>ferie: baseawar=0.05[0.0-0.4] time5m=0.04[0.0-0.3] lighraw=0.91[0.3-1.0] |
| E1 baseline + ts5m + ts209m [blend] | 88,45 | 76,97 | 74,71 | 85,68 | 133,47 | 93,45 | 6/8 | **[−13,91 ; −3,53]** | [−7,30 ; 8,09] | ouvre: time5m=0.79[0.6-0.9] time209m=0.21[0.1-0.4]<br>weekend: baseawar=0.57[0.3-0.7] time5m=0.35[0.3-0.4] time209m=0.07[0.0-0.4]<br>ferie: baseawar=0.23[0.1-0.3] time5m=0.03[0.0-0.1] time209m=0.75[0.6-0.8] |
| E2 E1 + chronos2 [blend] | 88,49 | 77,06 | 74,70 | 85,68 | 133,75 | 93,52 | 6/8 | **[−13,67 ; −3,67]** | [−7,03 ; 7,94] | ouvre: time5m=0.61[0.4-0.9] time209m=0.09[0.0-0.3] chronos2=0.30[0.1-0.4]<br>weekend: baseawar=0.57[0.3-0.7] time5m=0.35[0.3-0.4] time209m=0.07[0.0-0.4]<br>ferie: baseawar=0.23[0.1-0.3] time209m=0.75[0.6-0.8] chronos2=0.03[0.0-0.1] |
| E5 baseline + ts5m + lightgbm [best model] | 89,20 | 78,15 | 73,94 | 92,23 | 123,11 | 92,85 | 7/8 | **[−12,05 ; −5,13]** | [−2,67 ; 6,15] | ouvre: time5m=1.00[1.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time5m=0.12[0.0-1.0]<br>ferie: lighraw=1.00[1.0-1.0] |
| E4 E1 + lightgbm [blend] | 89,55 | 78,22 | 75,55 | 85,84 | 137,41 | 94,90 | 6/8 | **[−11,89 ; −3,41]** | [−4,38 ; 7,72] | ouvre: time5m=0.71[0.4-0.9] time209m=0.20[0.0-0.4] lighraw=0.09[0.0-0.2]<br>weekend: baseawar=0.51[0.3-0.6] time5m=0.18[0.1-0.3] time209m=0.05[0.0-0.4] lighraw=0.26[0.0-0.4]<br>ferie: time5m=0.01[0.0-0.1] time209m=0.31[0.0-0.8] lighraw=0.67[0.1-1.0] |
| E3 E1 + baseline day type [blend] | 89,71 | 78,42 | 76,41 | 86,26 | 135,52 | 94,79 | 6/8 | **[−11,56 ; −3,16]** | [−3,88 ; 7,97] | ouvre: basetype=0.16[0.1-0.3] time5m=0.57[0.1-0.7] time209m=0.26[0.1-0.6]<br>weekend: baseawar=0.57[0.3-0.7] basetype=0.01[0.0-0.1] time5m=0.34[0.3-0.4] time209m=0.07[0.0-0.4]<br>ferie: baseawar=0.01[0.0-0.1] basetype=0.39[0.0-0.5] time5m=0.01[0.0-0.1] time209m=0.59[0.5-0.8] |
| E6 baseline + ts209m + lightgbm [blend] | 89,94 | 78,55 | 77,47 | 86,51 | 134,02 | 94,80 | 7/8 | **[−11,04 ; −3,98]** | [−3,75 ; 7,41] | ouvre: baseawar=0.01[0.0-0.1] time209m=0.60[0.5-0.8] lighraw=0.39[0.2-0.5]<br>weekend: baseawar=0.54[0.3-0.6] time209m=0.11[0.0-0.6] lighraw=0.35[0.1-0.4]<br>ferie: time209m=0.31[0.0-0.8] lighraw=0.69[0.2-1.0] |
| E1 baseline + ts5m + ts209m [best model] | 92,19 | 79,71 | 73,94 | 95,71 | 131,46 | 96,47 | 5/8 | **[−12,20 ; −0,29]** | [−6,02 ; 12,83] | ouvre: time5m=1.00[1.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: time209m=1.00[1.0-1.0] |
| E2 E1 + chronos2 [best model] | 92,71 | 80,21 | 75,40 | 95,71 | 131,46 | 96,93 | 5/8 | **[−10,95 ; −0,29]** | [−4,59 ; 12,88] | ouvre: time5m=0.88[0.0-1.0] chronos2=0.12[0.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: time209m=1.00[1.0-1.0] |
| E4 E1 + lightgbm [best model] | 93,72 | 81,25 | 73,94 | 95,71 | 140,16 | 98,89 | 5/8 | **[−8,72 ; −0,08]** | [−2,40 ; 13,13] | ouvre: time5m=1.00[1.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: time209m=0.12[0.0-1.0] lighraw=0.88[0.0-1.0] |
| E3 E1 + baseline day type [best model] | 94,37 | 82,02 | 73,94 | 95,71 | 143,73 | 99,91 | 5/8 | [−9,05 ; 0,46] | [−3,11 ; 13,91] | ouvre: time5m=1.00[1.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: basetype=0.25[0.0-1.0] time209m=0.75[0.0-1.0] |
| E6 baseline + ts209m + lightgbm [best model] | 95,95 | 83,20 | 80,11 | 95,71 | 140,16 | 100,80 | 5/8 | [−6,17 ; 2,10] | **[0,39 ; 14,80]** | ouvre: time209m=0.88[0.0-1.0] lighraw=0.12[0.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: time209m=0.12[0.0-1.0] lighraw=0.88[0.0-1.0] |

### Ensemble choisi par type de jour (validation imbriquée) — honnête : oui

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Best E1-E6 configuration per day kind | 89,71 | 78,70 | 76,95 | 83,68 | 140,16 | 95,39 | 6/8 | **[−12,50 ; −2,45]** | [−3,11 ; 7,79] | ouvre: E1 [best model] x6, E3 [blend] x1, E5 [best model] x1<br>weekend: E5 [blend] x8<br>ferie: E1 [best model] x1, E5 [best model] x7 |

### Paires w·A + (1−w)·B, poids par type de jour — honnête : oui

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **baseline_holiday_aware + lightgbm_raw** | 87,74 | 77,89 | 80,85 | 81,39 | 125,89 | 91,84 | 6/8 | **[−13,73 ; −5,06]** | [0,00 ; 0,00] | w(A): ou=0.0-0.2 we=0.5-0.6 fe=0.0-0.3 |
| lightgbm_raw + timestral_5m_qmean | 88,57 | 78,13 | 73,87 | 88,82 | 128,36 | 92,92 | 6/8 | **[−11,28 ; −5,39]** | [−3,53 ; 4,61] | w(A): ou=0.0-0.2 we=0.3-0.8 fe=0.8-1.0 |
| lightgbm_raw + timestral_5m | 88,91 | 78,31 | 74,57 | 88,97 | 128,46 | 93,23 | 6/8 | **[−11,35 ; −5,15]** | [−3,64 ; 5,60] | w(A): ou=0.0-0.2 we=0.2-0.8 fe=0.8-1.0 |
| baseline_holiday_aware + timestral_5m_qmean | 89,27 | 78,52 | 73,39 | 83,62 | 144,94 | 95,67 | 6/8 | **[−13,08 ; −2,38]** | [−4,45 ; 8,49] | w(A): ou=0.0-0.0 we=0.4-0.7 fe=0.6-0.9 |
| baseline_day_type + lightgbm_raw | 89,54 | 80,08 | 80,58 | 87,15 | 123,11 | 93,09 | 6/8 | **[−12,55 ; −2,11]** | [−0,11 ; 4,03] | w(A): ou=0.0-0.2 we=0.0-0.0 fe=0.0-0.0 |
| baseline_holiday_aware + timestral_5m | 89,83 | 78,72 | 73,94 | 84,40 | 145,12 | 96,17 | 6/8 | **[−11,77 ; −2,43]** | [−4,35 ; 8,89] | w(A): ou=0.0-0.0 we=0.3-0.7 fe=0.6-0.9 |
| timestral_209m_qmean + timestral_5m | 90,14 | 77,71 | 74,29 | 90,33 | 132,64 | 94,82 | 5/8 | **[−15,07 ; −1,88]** | [−8,29 ; 10,52] | w(A): ou=0.2-0.5 we=0.0-0.5 fe=0.9-1.0 |
| timestral_209m_qmean + timestral_5m_qmean | 90,27 | 77,81 | 73,90 | 90,98 | 132,60 | 94,93 | 5/8 | **[−14,81 ; −1,95]** | [−7,96 ; 10,58] | w(A): ou=0.1-0.4 we=0.0-0.6 fe=0.9-1.0 |
| baseline_holiday_aware + timestral_209m_qmean | 90,41 | 78,61 | 78,95 | 87,27 | 131,61 | 94,89 | 6/8 | **[−12,37 ; −1,84]** | [−6,09 ; 10,13] | w(A): ou=0.0-0.3 we=0.3-0.8 fe=0.1-0.3 |
| timestral_209m + timestral_5m | 90,48 | 78,01 | 74,71 | 90,46 | 133,40 | 95,21 | 5/8 | **[−14,87 ; −1,68]** | [−7,90 ; 10,87] | w(A): ou=0.1-0.4 we=0.0-0.5 fe=0.9-1.0 |
| timestral_209m + timestral_5m_qmean | 90,59 | 78,15 | 74,10 | 91,25 | 133,36 | 95,31 | 5/8 | **[−14,57 ; −1,66]** | [−7,61 ; 10,89] | w(A): ou=0.0-0.3 we=0.0-0.6 fe=0.9-1.0 |
| baseline_holiday_aware + timestral_209m | 90,78 | 78,84 | 79,84 | 87,37 | 131,57 | 95,21 | 6/8 | **[−12,36 ; −1,32]** | [−6,14 ; 10,78] | w(A): ou=0.0-0.3 we=0.3-0.8 fe=0.1-0.3 |
| chronos2_qmean + timestral_209m_qmean | 91,99 | 78,98 | 75,19 | 93,41 | 133,64 | 96,56 | 4/8 | [−14,44 ; 0,97] | [−8,49 ; 14,13] | w(A): ou=0.4-0.7 we=0.0-0.2 fe=0.0-0.1 |
| chronos2 + timestral_209m | 92,02 | 78,93 | 75,24 | 93,41 | 133,68 | 96,58 | 4/8 | [−14,81 ; 1,38] | [−8,79 ; 14,43] | w(A): ou=0.6-0.8 we=0.0-0.2 fe=0.0-0.1 |
| chronos2_qmean + timestral_209m | 92,12 | 79,00 | 75,53 | 93,44 | 133,61 | 96,67 | 4/8 | [−14,72 ; 1,49] | [−8,82 ; 14,61] | w(A): ou=0.5-0.8 we=0.0-0.2 fe=0.0-0.1 |
| chronos2 + timestral_209m_qmean | 92,20 | 79,05 | 75,02 | 94,00 | 133,71 | 96,75 | 4/8 | [−14,56 ; 1,45] | [−8,54 ; 14,59] | w(A): ou=0.4-0.7 we=0.0-0.3 fe=0.0-0.1 |
| baseline_day_type + timestral_209m | 92,47 | 80,14 | 75,83 | 94,37 | 132,64 | 96,86 | 6/8 | **[−10,87 ; −0,78]** | [−4,23 ; 11,73] | w(A): ou=0.2-0.4 we=0.0-0.3 fe=0.1-0.5 |
| lightgbm_raw + timestral_209m_qmean | 92,57 | 81,17 | 76,62 | 93,46 | 133,87 | 97,09 | 7/8 | **[−8,22 ; −2,14]** | [−1,69 ; 10,41] | w(A): ou=0.1-0.5 we=0.2-1.0 fe=0.2-1.0 |
| baseline_day_type + timestral_209m_qmean | 92,68 | 80,47 | 74,79 | 94,57 | 135,46 | 97,39 | 5/8 | **[−10,39 ; −0,69]** | [−3,80 ; 11,72] | w(A): ou=0.2-0.3 we=0.0-0.3 fe=0.0-0.5 |
| baseline_day_type + timestral_5m | 92,68 | 81,07 | 74,94 | 90,40 | 145,11 | 98,62 | 6/8 | **[−6,61 ; −2,48]** | [−0,39 ; 9,17] | w(A): ou=0.0-0.2 we=0.0-0.2 fe=0.6-1.0 |
| lightgbm_raw + timestral_209m | 92,73 | 81,22 | 76,71 | 93,67 | 134,02 | 97,24 | 7/8 | **[−8,30 ; −2,03]** | [−1,72 ; 10,72] | w(A): ou=0.2-0.5 we=0.2-1.0 fe=0.2-1.0 |
| baseline_day_type + timestral_5m_qmean | 92,80 | 81,20 | 74,80 | 90,87 | 144,91 | 98,70 | 6/8 | **[−6,43 ; −2,41]** | [−0,08 ; 9,18] | w(A): ou=0.0-0.2 we=0.0-0.2 fe=0.6-1.0 |
| baseline_holiday_aware + chronos2 | 92,84 | 80,96 | 75,88 | 90,39 | 144,12 | 98,62 | 6/8 | **[−7,93 ; −1,30]** | [−2,27 ; 11,36] | w(A): ou=0.0-0.1 we=0.4-0.9 fe=0.7-0.9 |
| chronos2_qmean + lightgbm_raw | 93,01 | 81,28 | 76,05 | 96,84 | 128,80 | 96,86 | 5/8 | [−9,13 ; 0,49] | [−1,49 ; 13,06] | w(A): ou=0.6-0.9 we=0.0-0.7 fe=0.0-0.2 |
| baseline_holiday_aware + chronos2_qmean | 93,03 | 81,23 | 76,17 | 90,70 | 143,88 | 98,75 | 6/8 | **[−7,59 ; −1,22]** | [−1,91 ; 11,55] | w(A): ou=0.0-0.1 we=0.4-0.9 fe=0.7-0.9 |
| chronos2 + lightgbm_raw | 93,65 | 81,59 | 75,91 | 98,33 | 128,93 | 97,45 | 5/8 | [−9,25 ; 1,97] | [−1,58 ; 14,59] | w(A): ou=0.6-0.9 we=0.0-0.8 fe=0.0-0.2 |
| baseline_day_type + baseline_holiday_aware | 94,66 | 84,23 | 90,08 | 84,44 | 136,64 | 99,15 | 5/8 | [−7,69 ; 2,67] | **[4,27 ; 9,95]** | w(A): ou=0.3-0.7 we=0.1-0.3 fe=0.4-0.7 |
| chronos2_qmean + timestral_5m | 99,58 | 86,13 | 73,65 | 92,44 | 177,19 | 108,94 | 2/8 | [−3,51 ; 8,33] | **[2,28 ; 20,42]** | w(A): ou=0.2-0.4 we=0.0-0.4 fe=0.0-0.1 |
| chronos2_qmean + timestral_5m_qmean | 99,71 | 86,28 | 73,59 | 92,88 | 177,06 | 109,03 | 2/8 | [−3,07 ; 8,24] | **[2,81 ; 20,29]** | w(A): ou=0.1-0.4 we=0.0-0.4 fe=0.0-0.1 |
| chronos2 + timestral_5m | 99,78 | 86,31 | 74,03 | 92,38 | 177,65 | 109,17 | 2/8 | [−3,23 ; 8,46] | **[2,60 ; 20,45]** | w(A): ou=0.1-0.5 we=0.0-0.4 fe=0.0-0.2 |
| chronos2 + timestral_5m_qmean | 99,79 | 86,35 | 73,60 | 92,82 | 177,51 | 109,16 | 2/8 | [−3,01 ; 8,31] | **[2,91 ; 20,28]** | w(A): ou=0.1-0.4 we=0.0-0.4 fe=0.0-0.2 |
| baseline_day_type + chronos2 | 99,83 | 86,67 | 76,36 | 105,17 | 145,82 | 104,93 | 3/8 | [−4,54 ; 8,23] | **[2,23 ; 21,15]** | w(A): ou=0.0-0.2 we=0.0-0.5 fe=0.7-1.0 |
| baseline_day_type + chronos2_qmean | 99,98 | 86,70 | 77,08 | 105,66 | 144,00 | 104,83 | 3/8 | [−4,38 ; 8,47] | **[2,57 ; 21,38]** | w(A): ou=0.0-0.3 we=0.0-0.5 fe=0.7-1.0 |


## Occupation (points de %)

Meilleure stratégie honnête : **baseline_holiday_aware + timestral_5m** (1,692). Écarts : stratégie − `baseline_holiday_aware` et stratégie − meilleure honnête.

### Modèles seuls — honnête : sans choix

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| timestral_209m | 1,726 | 2,800 | 2,130 | 1,369 | 1,099 | 1,675 | 5/8 | **[−0,537 ; −0,088]** | [−0,109 ; 0,129] |  |
| timestral_209m_qmean | 1,732 | 2,832 | 2,107 | 1,391 | 1,237 | 1,690 | 6/8 | **[−0,499 ; −0,116]** | [−0,071 ; 0,150] |  |
| timestral_5m | 1,780 | 2,886 | 2,061 | 1,372 | 1,958 | 1,794 | 6/8 | **[−0,441 ; −0,074]** | [−0,006 ; 0,213] |  |
| chronos2 | 1,909 | 2,972 | 2,109 | 1,524 | 2,332 | 1,948 | 3/8 | [−0,361 ; 0,060] | **[0,059 ; 0,372]** |  |
| timestral_5m_qmean | 1,916 | 3,023 | 2,156 | 1,453 | 2,385 | 1,960 | 5/8 | [−0,353 ; 0,119] | **[0,052 ; 0,454]** |  |
| baseline_holiday_aware | 2,042 | 3,095 | 2,714 | 1,283 | 1,220 | 1,976 | 0/8 | [0,000 ; 0,000] | **[0,183 ; 0,484]** |  |
| chronos2_qmean | 2,045 | 3,085 | 2,182 | 1,617 | 2,771 | 2,118 | 3/8 | [−0,326 ; 0,262] | **[0,096 ; 0,569]** |  |
| lightgbm_raw | 2,045 | 3,179 | 2,464 | 1,549 | 1,908 | 2,029 | 2/8 | [−0,292 ; 0,391] | **[0,093 ; 0,727]** |  |
| baseline_day_type | 2,239 | 3,369 | 2,731 | 1,834 | 1,389 | 2,171 | 3/8 | [−0,257 ; 0,774] | **[0,114 ; 1,138]** |  |

### Règles de docs/methodology.md (choisies sur ces semaines) — honnête : non

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| C3 ts5m ouvre / baseline we / ts209m ferie | 1,658 | 2,765 | 2,061 | 1,283 | 1,099 | 1,612 | 6/8 | **[−0,539 ; −0,189]** | [−0,099 ; 0,016] |  |
| C4 C3 + lightgbm flow ferie | 1,658 | 2,765 | 2,061 | 1,283 | 1,099 | 1,612 | 6/8 | **[−0,539 ; −0,189]** | [−0,099 ; 0,016] |  |
| C5 chronos ouvre / baseline we / ts209m ferie | 1,685 | 2,780 | 2,109 | 1,283 | 1,099 | 1,637 | 6/8 | **[−0,490 ; −0,196]** | [−0,066 ; 0,047] |  |
| C1 chronos ouvre / baseline we+ferie | 1,695 | 2,791 | 2,109 | 1,283 | 1,220 | 1,654 | 6/8 | **[−0,486 ; −0,174]** | [−0,059 ; 0,055] |  |
| C2 C1 + lightgbm flow ferie | 1,695 | 2,791 | 2,109 | 1,283 | 1,220 | 1,654 | 6/8 | **[−0,486 ; −0,174]** | [−0,059 ; 0,055] |  |
| ts209m ouvre+ferie / baseline we | 1,697 | 2,790 | 2,130 | 1,283 | 1,099 | 1,648 | 5/8 | **[−0,569 ; −0,080]** | [−0,147 ; 0,133] |  |

### Ensembles, poids par type de jour — honnête : oui

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| E1 baseline + ts5m + ts209m [blend] | 1,704 | 2,803 | 2,110 | 1,330 | 1,142 | 1,658 | 7/8 | **[−0,483 ; −0,161]** | [−0,018 ; 0,039] | ouvre: baseawar=0.09[0.0-0.1] time5m=0.60[0.5-0.9] time209m=0.31[0.0-0.5]<br>weekend: baseawar=0.57[0.2-0.7] time5m=0.33[0.1-0.4] time209m=0.10[0.0-0.4]<br>ferie: baseawar=0.31[0.2-0.4] time5m=0.09[0.0-0.2] time209m=0.60[0.5-0.7] |
| E5 baseline + ts5m + lightgbm [blend] | 1,705 | 2,812 | 2,099 | 1,308 | 1,307 | 1,670 | 6/8 | **[−0,474 ; −0,167]** | [0,000 ; 0,037] | ouvre: baseawar=0.08[0.0-0.1] time5m=0.91[0.8-1.0] lighraw=0.01[0.0-0.1]<br>weekend: baseawar=0.59[0.3-0.7] time5m=0.40[0.2-0.7] lighraw=0.01[0.0-0.1]<br>ferie: baseawar=0.80[0.6-0.9] time5m=0.20[0.1-0.4] |
| E4 E1 + lightgbm [blend] | 1,717 | 2,818 | 2,126 | 1,339 | 1,141 | 1,669 | 6/8 | **[−0,473 ; −0,150]** | **[0,004 ; 0,044]** | ouvre: baseawar=0.09[0.0-0.1] time5m=0.59[0.5-0.8] time209m=0.31[0.0-0.5] lighraw=0.01[0.0-0.1]<br>weekend: baseawar=0.56[0.2-0.7] time5m=0.31[0.0-0.4] time209m=0.11[0.0-0.4] lighraw=0.01[0.0-0.1]<br>ferie: baseawar=0.24[0.2-0.3] time5m=0.03[0.0-0.2] time209m=0.66[0.6-0.7] lighraw=0.08[0.0-0.1] |
| E5 baseline + ts5m + lightgbm [best model] | 1,726 | 2,814 | 2,061 | 1,447 | 1,220 | 1,683 | 6/8 | **[−0,472 ; −0,143]** | [−0,030 ; 0,084] | ouvre: time5m=1.00[1.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time5m=0.12[0.0-1.0]<br>ferie: baseawar=1.00[1.0-1.0] |
| E2 E1 + chronos2 [blend] | 1,739 | 2,821 | 2,123 | 1,407 | 1,142 | 1,690 | 7/8 | **[−0,461 ; −0,149]** | [−0,017 ; 0,119] | ouvre: baseawar=0.10[0.0-0.2] time5m=0.31[0.1-0.5] time209m=0.31[0.0-0.5] chronos2=0.28[0.1-0.4]<br>weekend: baseawar=0.57[0.2-0.7] time5m=0.30[0.1-0.4] time209m=0.06[0.0-0.2] chronos2=0.06[0.0-0.5]<br>ferie: baseawar=0.31[0.2-0.4] time5m=0.01[0.0-0.1] time209m=0.60[0.5-0.7] chronos2=0.08[0.0-0.1] |
| E3 E1 + baseline day type [blend] | 1,741 | 2,853 | 2,176 | 1,330 | 1,145 | 1,692 | 7/8 | **[−0,437 ; −0,131]** | **[0,005 ; 0,121]** | ouvre: baseawar=0.09[0.0-0.1] basetype=0.04[0.0-0.2] time5m=0.54[0.1-0.8] time209m=0.34[0.0-0.6]<br>weekend: baseawar=0.57[0.2-0.7] time5m=0.33[0.1-0.4] time209m=0.10[0.0-0.4]<br>ferie: baseawar=0.20[0.2-0.2] basetype=0.18[0.0-0.2] time5m=0.03[0.0-0.2] time209m=0.60[0.6-0.6] |
| E6 baseline + ts209m + lightgbm [blend] | 1,743 | 2,844 | 2,180 | 1,341 | 1,084 | 1,690 | 6/8 | **[−0,444 ; −0,126]** | **[0,003 ; 0,096]** | ouvre: baseawar=0.20[0.1-0.3] time209m=0.72[0.5-0.9] lighraw=0.08[0.0-0.3]<br>weekend: baseawar=0.56[0.2-0.7] time209m=0.41[0.3-0.8] lighraw=0.03[0.0-0.1]<br>ferie: baseawar=0.23[0.1-0.3] time209m=0.69[0.6-0.8] lighraw=0.09[0.0-0.1] |
| E6 baseline + ts209m + lightgbm [best model] | 1,756 | 2,832 | 2,130 | 1,455 | 1,099 | 1,704 | 5/8 | **[−0,497 ; −0,066]** | [−0,074 ; 0,148] | ouvre: time209m=1.00[1.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: time209m=1.00[1.0-1.0] |
| E1 baseline + ts5m + ts209m [best model] | 1,758 | 2,829 | 2,133 | 1,455 | 1,099 | 1,706 | 5/8 | **[−0,471 ; −0,085]** | [−0,021 ; 0,130] | ouvre: time5m=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: time209m=1.00[1.0-1.0] |
| E3 E1 + baseline day type [best model] | 1,758 | 2,829 | 2,133 | 1,455 | 1,099 | 1,706 | 5/8 | **[−0,471 ; −0,085]** | [−0,021 ; 0,130] | ouvre: time5m=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: time209m=1.00[1.0-1.0] |
| E4 E1 + lightgbm [best model] | 1,758 | 2,829 | 2,133 | 1,455 | 1,099 | 1,706 | 5/8 | **[−0,471 ; −0,085]** | [−0,021 ; 0,130] | ouvre: time5m=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>ferie: time209m=1.00[1.0-1.0] |
| E2 E1 + chronos2 [best model] | 1,817 | 2,865 | 2,133 | 1,614 | 1,099 | 1,762 | 5/8 | **[−0,454 ; −0,030]** | [−0,021 ; 0,263] | ouvre: time5m=0.88[0.0-1.0] time209m=0.12[0.0-1.0]<br>weekend: baseawar=0.88[0.0-1.0] chronos2=0.12[0.0-1.0]<br>ferie: time209m=1.00[1.0-1.0] |

### Ensemble choisi par type de jour (validation imbriquée) — honnête : oui

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Best E1-E6 configuration per day kind | 1,697 | 2,790 | 2,090 | 1,340 | 1,139 | 1,651 | 6/8 | **[−0,499 ; −0,164]** | [−0,035 ; 0,035] | ouvre: E1 [best model] x2, E1 [blend] x1, E5 [best model] x5<br>weekend: E1 [blend] x2, E5 [blend] x5, E6 [blend] x1<br>ferie: E2 [blend] x1, E6 [blend] x7 |

### Paires w·A + (1−w)·B, poids par type de jour — honnête : oui

| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **baseline_holiday_aware + timestral_5m** | 1,692 | 2,797 | 2,082 | 1,296 | 1,307 | 1,658 | 6/8 | **[−0,484 ; −0,183]** | [0,000 ; 0,000] | w(A): ou=0.0-0.1 we=0.3-0.7 fe=0.6-0.9 |
| baseline_holiday_aware + timestral_209m | 1,708 | 2,794 | 2,125 | 1,331 | 1,076 | 1,657 | 7/8 | **[−0,513 ; −0,146]** | [−0,101 ; 0,085] | w(A): ou=0.1-0.3 we=0.2-0.7 fe=0.2-0.4 |
| timestral_209m + timestral_5m | 1,716 | 2,809 | 2,091 | 1,376 | 1,216 | 1,674 | 7/8 | **[−0,498 ; −0,145]** | [−0,038 ; 0,078] | w(A): ou=0.0-0.5 we=0.2-0.7 fe=0.7-1.0 |
| timestral_209m + timestral_5m_qmean | 1,724 | 2,814 | 2,105 | 1,384 | 1,188 | 1,680 | 6/8 | **[−0,497 ; −0,141]** | [−0,055 ; 0,116] | w(A): ou=0.4-0.7 we=0.6-0.9 fe=0.8-1.0 |
| baseline_holiday_aware + timestral_209m_qmean | 1,726 | 2,828 | 2,146 | 1,322 | 1,192 | 1,681 | 6/8 | **[−0,466 ; −0,135]** | [−0,045 ; 0,127] | w(A): ou=0.0-0.2 we=0.3-0.8 fe=0.3-0.7 |
| chronos2_qmean + timestral_209m | 1,730 | 2,794 | 2,093 | 1,429 | 1,139 | 1,682 | 7/8 | **[−0,515 ; −0,134]** | [−0,080 ; 0,157] | w(A): ou=0.3-0.6 we=0.0-0.3 fe=0.0-0.1 |
| timestral_209m_qmean + timestral_5m | 1,737 | 2,846 | 2,102 | 1,394 | 1,315 | 1,700 | 7/8 | **[−0,462 ; −0,136]** | [−0,022 ; 0,117] | w(A): ou=0.0-0.7 we=0.0-0.6 fe=0.8-1.0 |
| baseline_holiday_aware + timestral_5m_qmean | 1,743 | 2,854 | 2,170 | 1,313 | 1,272 | 1,702 | 6/8 | **[−0,405 ; −0,163]** | [−0,011 ; 0,118] | w(A): ou=0.0-0.1 we=0.4-0.8 fe=0.8-1.0 |
| baseline_day_type + timestral_209m_qmean | 1,744 | 2,846 | 2,128 | 1,391 | 1,241 | 1,701 | 6/8 | **[−0,487 ; −0,101]** | [−0,072 ; 0,189] | w(A): ou=0.0-0.1 we=0.0-0.0 fe=0.1-0.4 |
| baseline_day_type + timestral_5m | 1,748 | 2,858 | 2,091 | 1,372 | 1,561 | 1,729 | 7/8 | **[−0,452 ; −0,131]** | [−0,008 ; 0,126] | w(A): ou=0.0-0.1 we=0.0-0.0 fe=0.6-1.0 |
| baseline_day_type + timestral_209m | 1,748 | 2,841 | 2,170 | 1,374 | 1,088 | 1,696 | 6/8 | **[−0,488 ; −0,083]** | [−0,077 ; 0,177] | w(A): ou=0.1-0.3 we=0.0-0.1 fe=0.1-0.3 |
| baseline_holiday_aware + chronos2 | 1,752 | 2,825 | 2,090 | 1,479 | 1,202 | 1,706 | 7/8 | **[−0,439 ; −0,154]** | [−0,041 ; 0,184] | w(A): ou=0.1-0.2 we=0.2-0.9 fe=0.8-0.9 |
| chronos2 + timestral_209m | 1,758 | 2,816 | 2,105 | 1,483 | 1,167 | 1,710 | 6/8 | **[−0,483 ; −0,111]** | [−0,053 ; 0,191] | w(A): ou=0.4-0.8 we=0.0-0.7 fe=0.0-0.2 |
| chronos2_qmean + timestral_209m_qmean | 1,768 | 2,855 | 2,128 | 1,449 | 1,283 | 1,726 | 7/8 | **[−0,467 ; −0,114]** | [−0,047 ; 0,213] | w(A): ou=0.1-0.5 we=0.0-0.3 fe=0.0-0.1 |
| lightgbm_raw + timestral_209m | 1,768 | 2,865 | 2,185 | 1,405 | 1,096 | 1,714 | 5/8 | **[−0,449 ; −0,071]** | **[0,004 ; 0,138]** | w(A): ou=0.0-0.4 we=0.0-0.3 fe=0.1-0.2 |
| timestral_209m_qmean + timestral_5m_qmean | 1,770 | 2,874 | 2,152 | 1,417 | 1,289 | 1,728 | 7/8 | **[−0,451 ; −0,105]** | [−0,015 ; 0,176] | w(A): ou=0.4-1.0 we=0.5-1.0 fe=0.9-1.0 |
| chronos2 + timestral_209m_qmean | 1,775 | 2,850 | 2,100 | 1,512 | 1,272 | 1,733 | 7/8 | **[−0,460 ; −0,098]** | [−0,034 ; 0,230] | w(A): ou=0.2-0.6 we=0.0-0.8 fe=0.0-0.1 |
| baseline_holiday_aware + chronos2_qmean | 1,778 | 2,853 | 2,165 | 1,426 | 1,260 | 1,734 | 6/8 | **[−0,433 ; −0,116]** | [−0,006 ; 0,198] | w(A): ou=0.1-0.2 we=0.4-0.9 fe=0.8-1.0 |
| lightgbm_raw + timestral_209m_qmean | 1,781 | 2,898 | 2,156 | 1,447 | 1,268 | 1,738 | 6/8 | **[−0,442 ; −0,073]** | [−0,005 ; 0,211] | w(A): ou=0.0-0.3 we=0.0-0.4 fe=0.0-0.2 |
| lightgbm_raw + timestral_5m | 1,810 | 2,934 | 2,077 | 1,426 | 1,985 | 1,824 | 6/8 | **[−0,401 ; −0,053]** | **[0,022 ; 0,254]** | w(A): ou=0.0-0.1 we=0.0-0.4 fe=0.2-0.8 |
| baseline_day_type + timestral_5m_qmean | 1,812 | 2,925 | 2,156 | 1,453 | 1,576 | 1,789 | 6/8 | **[−0,401 ; −0,081]** | **[0,000 ; 0,261]** | w(A): ou=0.0-0.0 we=0.0-0.0 fe=0.7-1.0 |
| chronos2_qmean + timestral_5m | 1,820 | 2,911 | 2,083 | 1,456 | 1,958 | 1,830 | 6/8 | **[−0,423 ; −0,039]** | [−0,006 ; 0,279] | w(A): ou=0.0-0.3 we=0.0-0.4 fe=0.0-0.0 |
| chronos2 + timestral_5m | 1,821 | 2,908 | 2,061 | 1,490 | 1,958 | 1,831 | 6/8 | **[−0,420 ; −0,046]** | [−0,007 ; 0,277] | w(A): ou=0.2-0.4 we=0.0-0.7 fe=0.0-0.0 |
| baseline_day_type + chronos2 | 1,835 | 2,907 | 2,144 | 1,558 | 1,494 | 1,804 | 5/8 | **[−0,394 ; −0,039]** | **[0,005 ; 0,307]** | w(A): ou=0.0-0.2 we=0.0-0.3 fe=0.7-1.0 |
| baseline_holiday_aware + lightgbm_raw | 1,871 | 2,988 | 2,417 | 1,285 | 1,243 | 1,818 | 5/8 | [−0,348 ; 0,077] | **[0,030 ; 0,419]** | w(A): ou=0.2-0.4 we=0.8-0.9 fe=0.8-1.0 |
| baseline_day_type + chronos2_qmean | 1,913 | 2,976 | 2,199 | 1,689 | 1,497 | 1,877 | 4/8 | [−0,399 ; 0,103] | **[0,012 ; 0,449]** | w(A): ou=0.0-0.1 we=0.0-0.4 fe=0.8-1.0 |
| chronos2 + lightgbm_raw | 1,916 | 3,007 | 2,140 | 1,640 | 1,959 | 1,917 | 5/8 | [−0,314 ; 0,029] | **[0,076 ; 0,393]** | w(A): ou=0.8-1.0 we=0.4-1.0 fe=0.1-0.4 |
| lightgbm_raw + timestral_5m_qmean | 1,928 | 3,055 | 2,204 | 1,528 | 2,121 | 1,944 | 3/8 | [−0,329 ; 0,121] | **[0,058 ; 0,468]** | w(A): ou=0.0-0.3 we=0.0-0.6 fe=0.5-1.0 |
| baseline_day_type + baseline_holiday_aware | 1,934 | 3,049 | 2,537 | 1,276 | 1,184 | 1,873 | 4/8 | [−0,281 ; 0,148] | **[0,034 ; 0,540]** | w(A): ou=0.4-0.6 we=0.1-0.2 fe=0.3-0.4 |
| chronos2 + timestral_5m_qmean | 1,948 | 3,033 | 2,093 | 1,577 | 2,519 | 2,004 | 2/8 | [−0,348 ; 0,136] | **[0,074 ; 0,459]** | w(A): ou=0.6-0.8 we=0.0-0.9 fe=0.0-1.0 |
| chronos2_qmean + timestral_5m_qmean | 1,957 | 3,036 | 2,150 | 1,581 | 2,385 | 1,996 | 3/8 | [−0,355 ; 0,153] | **[0,042 ; 0,504]** | w(A): ou=0.3-0.6 we=0.0-0.7 fe=0.0-0.0 |
| chronos2_qmean + lightgbm_raw | 2,011 | 3,093 | 2,228 | 1,764 | 2,001 | 2,008 | 2/8 | [−0,306 ; 0,178] | **[0,093 ; 0,548]** | w(A): ou=0.7-1.0 we=0.2-1.0 fe=0.0-0.2 |
| baseline_day_type + lightgbm_raw | 2,062 | 3,199 | 2,564 | 1,590 | 1,389 | 2,006 | 3/8 | [−0,279 ; 0,415] | **[0,098 ; 0,763]** | w(A): ou=0.0-0.5 we=0.0-0.2 fe=1.0-1.0 |

## Poids figés des ensembles

Choisis sur les 8 semaines à la fois, par type de jour et par variable : les poids qu'utiliserait le rendu. Leur score honnête est celui des tableaux ci-dessus.

| Variable | Ensemble | Type de jour | Poids |
|---|---|---|---|
| debit_horaire | E1 baseline + ts5m + ts209m [best model] | ouvre | timestral_5m 1,0 |
| debit_horaire | E1 baseline + ts5m + ts209m [best model] | weekend | baseline_holiday_aware 1,0 |
| debit_horaire | E1 baseline + ts5m + ts209m [best model] | ferie | timestral_209m 1,0 |
| debit_horaire | E1 baseline + ts5m + ts209m [blend] | ouvre | timestral_5m 0,8, timestral_209m 0,2 |
| debit_horaire | E1 baseline + ts5m + ts209m [blend] | weekend | baseline_holiday_aware 0,6, timestral_5m 0,4 |
| debit_horaire | E1 baseline + ts5m + ts209m [blend] | ferie | baseline_holiday_aware 0,2, timestral_209m 0,8 |
| debit_horaire | E2 E1 + chronos2 [best model] | ouvre | timestral_5m 1,0 |
| debit_horaire | E2 E1 + chronos2 [best model] | weekend | baseline_holiday_aware 1,0 |
| debit_horaire | E2 E1 + chronos2 [best model] | ferie | timestral_209m 1,0 |
| debit_horaire | E2 E1 + chronos2 [blend] | ouvre | timestral_5m 0,6, timestral_209m 0,1, chronos2 0,3 |
| debit_horaire | E2 E1 + chronos2 [blend] | weekend | baseline_holiday_aware 0,6, timestral_5m 0,4 |
| debit_horaire | E2 E1 + chronos2 [blend] | ferie | baseline_holiday_aware 0,2, timestral_209m 0,8 |
| debit_horaire | E3 E1 + baseline day type [best model] | ouvre | timestral_5m 1,0 |
| debit_horaire | E3 E1 + baseline day type [best model] | weekend | baseline_holiday_aware 1,0 |
| debit_horaire | E3 E1 + baseline day type [best model] | ferie | timestral_209m 1,0 |
| debit_horaire | E3 E1 + baseline day type [blend] | ouvre | baseline_day_type 0,2, timestral_5m 0,5, timestral_209m 0,3 |
| debit_horaire | E3 E1 + baseline day type [blend] | weekend | baseline_holiday_aware 0,6, timestral_5m 0,4 |
| debit_horaire | E3 E1 + baseline day type [blend] | ferie | baseline_day_type 0,4, timestral_209m 0,6 |
| debit_horaire | E4 E1 + lightgbm [best model] | ouvre | timestral_5m 1,0 |
| debit_horaire | E4 E1 + lightgbm [best model] | weekend | baseline_holiday_aware 1,0 |
| debit_horaire | E4 E1 + lightgbm [best model] | ferie | lightgbm_raw 1,0 |
| debit_horaire | E4 E1 + lightgbm [blend] | ouvre | timestral_5m 0,7, timestral_209m 0,2, lightgbm_raw 0,1 |
| debit_horaire | E4 E1 + lightgbm [blend] | weekend | baseline_holiday_aware 0,5, timestral_5m 0,2, lightgbm_raw 0,3 |
| debit_horaire | E4 E1 + lightgbm [blend] | ferie | timestral_209m 0,3, lightgbm_raw 0,7 |
| debit_horaire | E5 baseline + ts5m + lightgbm [best model] | ouvre | timestral_5m 1,0 |
| debit_horaire | E5 baseline + ts5m + lightgbm [best model] | weekend | baseline_holiday_aware 1,0 |
| debit_horaire | E5 baseline + ts5m + lightgbm [best model] | ferie | lightgbm_raw 1,0 |
| debit_horaire | E5 baseline + ts5m + lightgbm [blend] | ouvre | timestral_5m 0,9, lightgbm_raw 0,1 |
| debit_horaire | E5 baseline + ts5m + lightgbm [blend] | weekend | baseline_holiday_aware 0,5, timestral_5m 0,2, lightgbm_raw 0,3 |
| debit_horaire | E5 baseline + ts5m + lightgbm [blend] | ferie | lightgbm_raw 1,0 |
| debit_horaire | E6 baseline + ts209m + lightgbm [best model] | ouvre | timestral_209m 1,0 |
| debit_horaire | E6 baseline + ts209m + lightgbm [best model] | weekend | baseline_holiday_aware 1,0 |
| debit_horaire | E6 baseline + ts209m + lightgbm [best model] | ferie | lightgbm_raw 1,0 |
| debit_horaire | E6 baseline + ts209m + lightgbm [blend] | ouvre | timestral_209m 0,6, lightgbm_raw 0,4 |
| debit_horaire | E6 baseline + ts209m + lightgbm [blend] | weekend | baseline_holiday_aware 0,6, timestral_209m 0,1, lightgbm_raw 0,3 |
| debit_horaire | E6 baseline + ts209m + lightgbm [blend] | ferie | timestral_209m 0,3, lightgbm_raw 0,7 |
| taux_occupation | E1 baseline + ts5m + ts209m [best model] | ouvre | timestral_5m 1,0 |
| taux_occupation | E1 baseline + ts5m + ts209m [best model] | weekend | baseline_holiday_aware 1,0 |
| taux_occupation | E1 baseline + ts5m + ts209m [best model] | ferie | timestral_209m 1,0 |
| taux_occupation | E1 baseline + ts5m + ts209m [blend] | ouvre | baseline_holiday_aware 0,1, timestral_5m 0,6, timestral_209m 0,3 |
| taux_occupation | E1 baseline + ts5m + ts209m [blend] | weekend | baseline_holiday_aware 0,6, timestral_5m 0,3, timestral_209m 0,1 |
| taux_occupation | E1 baseline + ts5m + ts209m [blend] | ferie | baseline_holiday_aware 0,3, timestral_5m 0,1, timestral_209m 0,6 |
| taux_occupation | E2 E1 + chronos2 [best model] | ouvre | timestral_5m 1,0 |
| taux_occupation | E2 E1 + chronos2 [best model] | weekend | baseline_holiday_aware 1,0 |
| taux_occupation | E2 E1 + chronos2 [best model] | ferie | timestral_209m 1,0 |
| taux_occupation | E2 E1 + chronos2 [blend] | ouvre | baseline_holiday_aware 0,1, timestral_5m 0,3, timestral_209m 0,3, chronos2 0,3 |
| taux_occupation | E2 E1 + chronos2 [blend] | weekend | baseline_holiday_aware 0,6, timestral_5m 0,3, timestral_209m 0,1 |
| taux_occupation | E2 E1 + chronos2 [blend] | ferie | baseline_holiday_aware 0,3, timestral_209m 0,6, chronos2 0,1 |
| taux_occupation | E3 E1 + baseline day type [best model] | ouvre | timestral_5m 1,0 |
| taux_occupation | E3 E1 + baseline day type [best model] | weekend | baseline_holiday_aware 1,0 |
| taux_occupation | E3 E1 + baseline day type [best model] | ferie | timestral_209m 1,0 |
| taux_occupation | E3 E1 + baseline day type [blend] | ouvre | baseline_holiday_aware 0,1, timestral_5m 0,6, timestral_209m 0,3 |
| taux_occupation | E3 E1 + baseline day type [blend] | weekend | baseline_holiday_aware 0,6, timestral_5m 0,3, timestral_209m 0,1 |
| taux_occupation | E3 E1 + baseline day type [blend] | ferie | baseline_holiday_aware 0,2, baseline_day_type 0,2, timestral_209m 0,6 |
| taux_occupation | E4 E1 + lightgbm [best model] | ouvre | timestral_5m 1,0 |
| taux_occupation | E4 E1 + lightgbm [best model] | weekend | baseline_holiday_aware 1,0 |
| taux_occupation | E4 E1 + lightgbm [best model] | ferie | timestral_209m 1,0 |
| taux_occupation | E4 E1 + lightgbm [blend] | ouvre | baseline_holiday_aware 0,1, timestral_5m 0,6, timestral_209m 0,3 |
| taux_occupation | E4 E1 + lightgbm [blend] | weekend | baseline_holiday_aware 0,6, timestral_5m 0,3, timestral_209m 0,1 |
| taux_occupation | E4 E1 + lightgbm [blend] | ferie | baseline_holiday_aware 0,2, timestral_209m 0,7, lightgbm_raw 0,1 |
| taux_occupation | E5 baseline + ts5m + lightgbm [best model] | ouvre | timestral_5m 1,0 |
| taux_occupation | E5 baseline + ts5m + lightgbm [best model] | weekend | baseline_holiday_aware 1,0 |
| taux_occupation | E5 baseline + ts5m + lightgbm [best model] | ferie | baseline_holiday_aware 1,0 |
| taux_occupation | E5 baseline + ts5m + lightgbm [blend] | ouvre | baseline_holiday_aware 0,1, timestral_5m 0,9 |
| taux_occupation | E5 baseline + ts5m + lightgbm [blend] | weekend | baseline_holiday_aware 0,6, timestral_5m 0,4 |
| taux_occupation | E5 baseline + ts5m + lightgbm [blend] | ferie | baseline_holiday_aware 0,8, timestral_5m 0,2 |
| taux_occupation | E6 baseline + ts209m + lightgbm [best model] | ouvre | timestral_209m 1,0 |
| taux_occupation | E6 baseline + ts209m + lightgbm [best model] | weekend | baseline_holiday_aware 1,0 |
| taux_occupation | E6 baseline + ts209m + lightgbm [best model] | ferie | timestral_209m 1,0 |
| taux_occupation | E6 baseline + ts209m + lightgbm [blend] | ouvre | baseline_holiday_aware 0,2, timestral_209m 0,7, lightgbm_raw 0,1 |
| taux_occupation | E6 baseline + ts209m + lightgbm [blend] | weekend | baseline_holiday_aware 0,6, timestral_209m 0,4 |
| taux_occupation | E6 baseline + ts209m + lightgbm [blend] | ferie | baseline_holiday_aware 0,2, timestral_209m 0,7, lightgbm_raw 0,1 |
