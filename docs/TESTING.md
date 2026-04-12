# AI Risk — Regression Checklist

Manual regression checklist for verifying that important flows still work correctly after code changes. Run the applicable sections after any non-trivial edit to `app.py`, `ai_risk_inference.py`, `modeling.py`, `risk_data.py`, `sts_calculator.py`, `model_metadata.py`, `export_helpers.py`, or `temporal_validation.py`.

Legend: **C** = critical (silent breakage or wrong results possible) | **O** = optional (cosmetic or low-blast-radius)

---

## 1. Training Flow

Prerequisites: a valid multi-sheet `.xlsx` file (Preoperative, Pre-Echocardiogram, Postoperative).

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 1.1 | Upload file, trigger training | Training completes without crash; leaderboard populated | **C** |
| 1.2 | `ia_risk_bundle.joblib` present after training | File created/updated on disk; modification timestamp advances | **C** |
| 1.3 | Leaderboard shows all available candidates | At minimum: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, StackingEnsemble | **C** |
| 1.4 | OOF metrics shown for each model | AUC, AUPRC, Brier score all populated (not NaN/blank) | **C** |
| 1.5 | Auto-selected model passes clinical guardrails | Best model produces predictions below 8% for at least some patients; AUC above floor | **C** |
| 1.6 | Model version string set | Models tab shows a non-empty version string (e.g., `2024-Q3-v…`) | O |
| 1.7 | Feature count shown in training metadata | Models tab and execution report both display `n_features` | O |

---

## 2. Single-Patient AI Risk Prediction

Prerequisites: bundle loaded. Use a patient with complete data.

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 2.1 | Submit valid form → AI Risk probability displayed | Probability between 0% and 100%, not NaN | **C** |
| 2.2 | Completeness level shown | One of: complete / adequate / partial / low — never blank | **C** |
| 2.3 | EuroSCORE II displayed for the same patient | Non-null percentage value | **C** |
| 2.4 | STS Score displayed or explicitly absent | Either a numeric probability or a clear "not available / procedure not supported" message — never silent blank | **C** |
| 2.5 | SHAP waterfall renders | Waterfall chart visible; top contributing variables shown | **C** |
| 2.6 | Remove all optional fields, resubmit | App still runs; completeness level degrades (not crashes) | **C** |
| 2.7 | Enter an invalid numeric value (e.g. letters in Age) | Graceful handling; probability shown or meaningful error message — no Python traceback shown | **C** |
| 2.8 | `_patient_identifier_from_row` fallback | If Name is blank, incident/log uses `row_1` not a crash | O |

---

## 3. Batch Prediction

Prerequisites: bundle loaded. Use a CSV or XLSX file with at least 5 rows including at least one patient with missing fields.

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 3.1 | Upload batch file → AI Risk column populated | Every row has a numeric probability or NaN (not blank/error) | **C** |
| 3.2 | EuroSCORE II column populated for all rows | Non-null values for rows with sufficient data | **C** |
| 3.3 | STS Score column populated | Probabilities present for supported procedures; NaN or "failed" for unsupported — no silent blank column | **C** |
| 3.4 | XLSX download works | Downloaded file opens in Excel with all score columns | **C** |
| 3.5 | CSV download works | Downloaded file contains all score columns and correct row count | **C** |
| 3.6 | Row with impossible/malformed values | That row produces NaN in AI Risk column; remaining rows unaffected; incident captured | **C** |
| 3.7 | AI Risk incidents visible in execution report | Execution report expander lists any rows that failed AI Risk inference with patient_id and reason | **C** |
| 3.8 | Batch result row count = upload row count | No rows silently dropped | **C** |

---

## 4. Temporal Validation

Prerequisites: bundle loaded. Upload a cohort file from a time period different from training.

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 4.1 | Upload temporal cohort → AI Risk probabilities computed | Probabilities present for all rows; NaN only for inference failures | **C** |
| 4.2 | Frozen model is used — no retraining | Models tab model version unchanged after temporal run | **C** |
| 4.3 | Discrimination metrics shown for the temporal cohort | AUC, AUPRC, Brier score shown for the temporal cohort | **C** |
| 4.4 | Calibration metrics shown | Calibration intercept and slope shown for the temporal cohort | **C** |
| 4.5 | Comparison against training-cohort performance shown | Pairwise comparison section visible in the Temporal Validation tab | **C** |
| 4.6 | Temporal validation Markdown report downloadable | Download button present; file opens and contains cohort summary and metric tables | O |
| 4.7 | Locked-model metadata table shown | Training date range, event rate, feature count, threshold displayed in the tab | O |

---

## 5. Chronological Blocking in Temporal Validation

Use cohorts constructed to exercise each overlap case. Requires control over the training and validation date ranges.

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 5.1 | Validation cohort strictly after training end date | UI shows green / success status: "No temporal overlap detected — ideal for temporal validation" | **C** |
| 5.2 | Validation cohort overlapping training period | UI shows warning: "Temporal overlap detected … patients in the overlapping period may have been used for training" | **C** |
| 5.3 | Validation cohort entirely before training start date | UI shows error: "validation cohort is entirely BEFORE the training cohort … retrograde validation" | **C** |
| 5.4 | Training or validation date range unparseable ("Unknown") | UI shows warning: "Could not parse temporal ranges — overlap check skipped" — no crash | **C** |
| 5.5 | Severity of overlap warning does not block the rest of the tab | Metrics and report are still computed even when overlap is detected | **C** |
| 5.6 | Severity of retrograde error is displayed prominently | Error is shown at the top of the Temporal Validation tab, not just in the execution report | O |

---

## 6. STS Score Cache Behavior

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 6.1 | Run app on a dataset, then run again without changing the data | Second run is faster; execution report shows `cached: N` patients, `fetched: 0` | **C** |
| 6.2 | Modify one patient's data and rerun | Only the modified patient is re-queried; all others served from cache | **C** |
| 6.3 | Delete the STS Score cache directory and rerun | All patients queried fresh; new cache written; execution report shows `cached: 0, fetched: N` | **C** |
| 6.4 | Cache key is payload-based, not position-based | Moving a patient from row 3 to row 7 without changing their data does not cause a cache miss | **C** |
| 6.5 | Execution report shows cache hit / fetch counts | `cached`, `fetched` (and `stale` if applicable) counts visible in execution report | O |

---

## 7. STS Score Stale Fallback and Failed Behavior

To test these scenarios, it helps to temporarily disable network access or use a patient whose procedure is unsupported.

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 7.1 | STS Score query fails for one patient, no prior cache | That patient's STS Score = NaN; remaining patients unaffected | **C** |
| 7.2 | STS Score query fails for one patient, prior cache exists | Stale cached value returned; execution report shows patient flagged as `stale_fallback` | **C** |
| 7.3 | Less than 50% of patients fail, at least one usable result | Execution report status chip: ⚠️ warning (not ❌ blocking error); analysis continues | **C** |
| 7.4 | All patients fail or fail_ratio ≥ 0.5 | Execution report status chip: ❌ blocking error; inline banner displayed | **C** |
| 7.5 | Patient with unsupported procedure (e.g. TAVR only) | STS Score = NaN or "procedure not supported"; no crash | **C** |
| 7.6 | Retry count reflected in execution report | If retries occur, execution report notes them (or at minimum the final failed count) | O |

---

## 8. Execution Report Visibility

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 8.1 | After any run, compact status row visible | At least one status chip visible in sidebar/header area (not hidden or blank) | **C** |
| 8.2 | Ingestion & normalization chip present | Chip shows ✅ OK on clean data; ⚠️ or ❌ if eligibility issues occur | **C** |
| 8.3 | STS Score execution chip present | Chip reflects actual STS Score outcome (OK / warning / error) | **C** |
| 8.4 | AI Risk training chip present | Chip reflects training outcome (OK / warning if guardrails triggered) | **C** |
| 8.5 | Execution report expander (Overview tab) opens | Expander visible; per-step counters show rows_in, rows_out, imputed, excluded | **C** |
| 8.6 | Blocking error triggers inline banner | If any phase is ❌, an inline error banner appears at the top of the page — not only in the expander | **C** |
| 8.7 | Incident lists shown for failed patients | Failed patients listed by patient_id with stage and reason in the expander | **C** |
| 8.8 | No execution report after fresh page load (no data) | No phantom status chips from a previous session | O |

---

## 9. Youden Threshold Visibility in the Leaderboard

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 9.1 | Leaderboard visible in Models tab | All trained candidates listed with OOF metrics | **C** |
| 9.2 | Per-model Youden threshold shown | Youden column present in leaderboard; each model has its own value | **C** |
| 9.3 | Youden threshold differs from 8% for at least one model | Youden threshold is not always 8% — if every model shows exactly 8%, something is wrong | **C** |
| 9.4 | Statistical Comparison tab: threshold switch works | Toggle between fixed 8% mode and Youden mode changes sensitivity/specificity/PPV/NPV columns; AUC/AUPRC/Brier remain unchanged | **C** |
| 9.5 | Fixed 8% is the default in Statistical Comparison tab | On first load (without changing the switch), the threshold shown is 8% | **C** |
| 9.6 | Force-selecting a non-default model from the leaderboard | Prediction tab and batch tab use the force-selected model; model version shown updates | O |

---

## 10. Export Flows

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 10.1 | Statistical Comparison tab: XLSX export | Download produces a valid `.xlsx` file; each section (Discrimination, Calibration, DeLong, Bootstrap, NRI/IDI) on a separate sheet | **C** |
| 10.2 | Statistical Comparison tab: CSV export | Download produces a UTF-8 `.csv` with all sections concatenated and section headers | **C** |
| 10.3 | Statistical Comparison tab: PDF export | Download produces a readable `.pdf`; all tables present (requires fpdf2 installed) | O |
| 10.4 | Temporal Validation tab: Markdown report download | Download produces a `.md` file with cohort summary and metric tables | O |
| 10.5 | Batch & Export tab: XLSX download | Downloaded file contains AI Risk, EuroSCORE II, and STS Score columns; row count matches uploaded file | **C** |
| 10.6 | Batch & Export tab: CSV download | Same content as XLSX; numeric values not quoted; NaN represented as empty cell | **C** |
| 10.7 | Export after force-selecting a non-default model | Export reflects the force-selected model's predictions, not the auto-selected default | **C** |
| 10.8 | PDF export with non-ASCII characters (e.g. Portuguese labels) | PDF renders without character substitution errors; no `latin-1` encoding crash | O |

---

## 11. Old Bundle Compatibility

Prerequisites: a `.joblib` bundle saved under a prior model version (before the most recent code change).

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 11.1 | Load an old bundle on app startup | App loads without crash; no `AttributeError` or `KeyError` on bundle fields | **C** |
| 11.2 | Single-patient prediction with old bundle | Probability returned; no crash from schema mismatch | **C** |
| 11.3 | Batch prediction with old bundle | All rows scored; no crash | **C** |
| 11.4 | `prepared.feature_columns == artifacts.feature_columns` | If schemas match: no warning shown. If they differ: a warning is shown in the UI at bundle load time — not a silent mismatch | **C** |
| 11.5 | Old bundle missing a new metadata field | App uses a safe default (e.g., `metadata.get("new_field", "N/A")`); no KeyError | **C** |
| 11.6 | Temporal validation with old bundle | Frozen inference runs; no crash from missing reference_df or feature_columns | **C** |

---

## 12. AI Risk Incident Reporting in Batch and Temporal Flows

| # | Scenario | Expected Result | C/O |
|:--|:--|:--|:--:|
| 12.1 | Batch row with all-NaN features | Incident captured with stage=`ai_risk_inference`; probability=NaN; remaining rows unaffected | **C** |
| 12.2 | Batch row where `predict_proba` fails (e.g. unexpected dtype) | Incident captured with exception type and message; batch continues; no unhandled exception | **C** |
| 12.3 | Patient identifier in incident uses Name column if present | Incident `patient_id` shows the patient's Name, not a generic `row_N` | **C** |
| 12.4 | Patient identifier falls back to `row_N` when Name is absent | Incident `patient_id` is `row_3` (1-based) when Name/Nome/_patient_key all missing | **C** |
| 12.5 | Incidents visible in execution report expander | Failed rows listed under the AI Risk batch section in the expander | **C** |
| 12.6 | Temporal validation: failing row → NaN probability, not crash | `apply_frozen_model_to_temporal_cohort` returns NaN for that row; incident appended to `incidents` list | **C** |
| 12.7 | Temporal validation incidents surfaced in UI | Temporal validation tab or execution report shows count of failed rows | **C** |
| 12.8 | Zero incidents reported on clean data | No spurious incidents when the dataset and bundle are both well-formed | O |

---

## Critical Tests — Never Skip

The following tests should be run after any non-trivial change. If any of these fail, the change should not be merged.

| Priority | Test(s) | Why |
|:--|:--|:--|
| 🔴 Highest | 1.1, 1.2, 1.5 | Training breakage is invisible until a user tries to retrain |
| 🔴 Highest | 2.1, 2.4 | Silent wrong probability or missing STS Score — core clinical output |
| 🔴 Highest | 3.1, 3.6, 3.7 | Batch crash or silent row drop corrupts research results |
| 🔴 Highest | 4.1, 4.2 | Temporal validation using wrong model (retrained) invalidates the study |
| 🔴 Highest | 5.1, 5.2, 5.3 | Chronological check is the primary methodological safeguard in temporal validation |
| 🔴 Highest | 7.3, 7.4 | Incorrect severity classification can hide a broken STS Score run |
| 🔴 Highest | 11.1, 11.4 | Old bundles must load cleanly; silent schema mismatch produces wrong probabilities |
| 🔴 Highest | 12.1, 12.2 | Inference failure must be captured as incident, not unhandled exception |
| 🟡 High | 6.1, 6.2 | Cache regression causes unnecessary re-queries and may change results if STS Score interface changes |
| 🟡 High | 9.4, 9.5 | Threshold switch affecting discrimination metrics would invalidate comparisons |
| 🟡 High | 10.1, 10.5 | Broken export is discovered only when the user tries to download |

---

## Remaining Validation Gaps

The following areas are not covered by this manual checklist and represent known gaps:

| Gap | Notes |
|:--|:--|
| Automated unit tests | No test suite currently exists. Tests for `ai_risk_inference.py`, `temporal_validation.py`, and `export_helpers.py` would be high value because these modules have no Streamlit dependencies and are straightforward to test in isolation. |
| STS Score WebSocket regression | Reliably testing the live WebSocket path requires network access to `acsdriskcalc.research.sts.org`. The stale-fallback and failed-patient paths (7.2–7.4) require network mocking or temporary firewall rules. |
| Multi-format ingestion (DB, Parquet) | The checklist covers XLSX/CSV but not `.db`/`.sqlite`/`.parquet` sources. These share the same normalization path but have separate loading code. |
| Calibration correctness | No test verifies that the per-model calibration strategy (isotonic vs. sigmoid) is still applied correctly inside each CV fold after changes to `modeling.py`. |
| Language toggle (EN/PT) | The bilingual UI is not covered. A change that breaks Portuguese labels would not be caught by any test above. Low impact for research use, but worth a spot-check after UI changes. |
| Bundle schema versioning | There is no formal schema version field in the bundle. Test 11.5 (graceful handling of missing fields) relies on `dict.get()` defaults throughout the codebase — no automated check verifies that every new field has a safe fallback. |
