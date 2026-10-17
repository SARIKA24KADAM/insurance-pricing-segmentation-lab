# Step-by-step project guide

This guide walks through the synthetic auto insurance pricing project from setup through interpretation. It describes the files, modeling choices, generated outputs, and limits of the analysis.

## 1. Understand the project goal

The project asks whether a simple frequency-severity model can estimate relative claim costs and how an illustrative premium indication would differ across portfolio segments.

It is an educational experiment using generated policy data. It is not an insurer's pricing model, an actuarial recommendation, or evidence of real-world performance.

## 2. Find the project files

The project is in its own repository: [insurance-pricing-segmentation-lab](https://github.com/SARIKA24KADAM/insurance-pricing-segmentation-lab).

- `src/insurance_pricing/data.py` creates the deterministic synthetic policy data.
- `src/insurance_pricing/model.py` prepares the features and fits frequency and severity models.
- `src/insurance_pricing/evaluation.py` computes holdout metrics, calibration, and segment summaries.
- `src/insurance_pricing/pipeline.py` runs the workflow and writes CSV, JSON, and PNG outputs.
- `tests/test_data_and_evaluation.py` contains focused unit tests.
- `docs/model_card.md` summarizes intended use, methods, and limitations.

## 3. Install Python

Use Python 3.10 or newer. Confirm that Python and pip are available:

```bash
python --version
python -m pip --version
```

## 4. Create and activate a virtual environment

Run these commands from the project root.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 5. Install the project

Install the listed libraries and the local package:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

NumPy and pandas handle tabular data, scikit-learn provides the regressions and preprocessing, and Matplotlib creates PNG charts.

## 6. Run the end-to-end pipeline

From the project root, run:

```bash
python -m insurance_pricing.pipeline --output reports --policies 30000 --seed 42
```

The default seed makes the generated example reproducible. Change `--policies` to change the portfolio size, or change `--seed` to generate another synthetic sample. The pipeline creates its data internally; no external dataset or customer information is needed.

## 7. Follow the modeling steps

The pipeline performs these operations:

1. **Generate policy-year records.** It creates policy years 2021â€“2024, exposure, driver and vehicle attributes, territory, claim counts, aggregate claim cost, and a noisy synthetic current premium.
2. **Split by time.** Policy years 2021â€“2023 are used for training. The latest year, 2024, is reserved as an out-of-time holdout.
3. **Fit claim frequency.** A Poisson regression estimates annual claim frequency. The target is claim count divided by exposure, and exposure is also supplied as the model's sample weight.
4. **Fit claim severity.** A Gamma regression estimates mean cost per claim using policies with claims. Claim count is the sample weight, so policies with more claims contribute proportionately.
5. **Estimate pure premium.** Predicted annual frequency is multiplied by predicted mean severity.
6. **Evaluate on the holdout.** The pipeline reports frequency Poisson deviance, severity Gamma deviance, and pure-premium mean absolute and root mean squared errors.
7. **Check calibration.** Holdout policies are divided into predicted-risk deciles and observed loss cost is compared with expected pure premium.
8. **Summarize portfolio impact.** It groups the holdout by territory, vehicle type, and driver age band. A 1.30 expense-and-margin load and a 25% rate-change cap are illustrative scenario assumptions in the code.
9. **Save tables and charts.** CSV, JSON, and PNG files are written under `reports/`.

## 8. Review the outputs

After running the command, inspect:

- `reports/model_metrics.json` for the holdout period, sample size, and model scores.
- `reports/calibration_by_risk_decile.csv` for expected and observed frequency, severity, and loss cost by decile.
- `reports/portfolio_segment_impact.csv` for segment experience and the illustrative capped premium changes.
- `reports/graphs/pure_premium_calibration.png` for observed versus expected loss cost by decile.
- `reports/graphs/observed_expected_by_decile.png` for observed-to-expected ratios. A ratio of 1 means observed and expected loss cost match for that group.
- `reports/graphs/territory_rate_impact.png` for the capped scenario changes by territory.

`reports/policy_sample.csv` is also generated for local inspection, but it is excluded from Git because it is a generated policy-level file. The summary tables and charts are the shareable artifacts.

## 9. Interpret results carefully

The out-of-time split helps show how the model behaves on a later simulated year. Deviance scores assess the frequency and severity fits; MAE and RMSE summarize pure-premium error. Calibration plots show whether predicted risk levels align with observed simulated losses across groups. Segment rate changes show the effect of the stated scenario assumptions.

Do not interpret these metrics as a claim that the model outperforms an insurer's current pricing or is suitable for customer decisions. The generator defines the data relationships, and all premium and expense assumptions are invented for demonstration.

## 10. Modify the experiment responsibly

Useful extensions include varying the seed and sample size, comparing a regularized GLM with a gradient boosting model, bootstrapping uncertainty around segment results, adding a clearly documented trend scenario, or comparing uncapped with capped portfolio impacts. Preserve the out-of-time holdout and record changed assumptions so comparisons remain understandable.

Do not introduce real policyholder data without appropriate authorization and governance. Real insurance pricing would need actuarial ownership, representative data, coverage and exposure definitions, trend and development treatment, jurisdiction-specific compliance and fairness analysis, independent validation, and monitoring.

## 11. Run the unit tests when making code changes

Install pytest if needed, then run:

```bash
python -m pip install pytest
python -m pytest
```

The current tests check deterministic synthetic generation, nonnegative targets, and basic calibration table properties. They do not validate actuarial suitability or production readiness.

