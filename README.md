# Modeling & Machine Learning Scripts for EIS Analysis

This folder contains Jupyter notebooks and scripts for modeling Electrochemical Impedance Spectroscopy (EIS) data, with a focus on nanoporous gold (NPG) in H₂SO₄ electrolyte. The workflows cover data extraction, physics-based fitting, Physics-Informed Neural Networks (PINNs), inverse estimation of concentration and temperature, sensitivity analysis, and classical machine learning regression.

---

## Table of Contents

1. [Requirements & Installation](#requirements--installation)
2. [Script Overview](#script-overview)
3. [Recommended Execution Order](#recommended-execution-order)
4. [Detailed Script Descriptions](#detailed-script-descriptions)
5. [Data Formats & Paths](#data-formats--paths)
6. [Output Artifacts](#output-artifacts)
7. [Troubleshooting](#troubleshooting)

---

## Requirements & Installation

### Python Version

- **Python 3.9 or higher** (3.10+ recommended for full type-hint support)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For **symbolic regression** in `regression_ANN.ipynb` (PySR or gplearn), uncomment and install:

```bash
pip install pysr
# or
pip install gplearn
```

### GPU Support (Optional)

For faster training of PINN and inverse models, install PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Script Overview

| Script | Purpose | Key Dependencies |
|--------|---------|------------------|
| `Data_extraction.ipynb` | Extract, cut, fit, and interpolate EIS data; reorganize folder structure | numpy, pandas, scipy, matplotlib |
| `solution_resistance_batch.ipynb` | Collect Rₛ from fitted data; detect degenerate Rₛ pairs | pandas |
| `solution_conductivity_batch.ipynb` | Convert Rₛ → conductivity; find degenerate conductivity pairs | pandas |
| `dynamic_batch_fitting.ipynb` | Physics-aware batch fitting of EIS spectra (Rs+Zarc+TL model) | numpy, pandas, scipy, matplotlib |
| `CENN_forward_model.ipynb` | Train PINN forward model: Z(ω) from (C, T, f) | torch, sklearn |
| `CENN_inverse.ipynb` | Inverse estimation of C and T from EIS spectra using trained PINN | torch |
| `regression_ANN.ipynb` | Classical ML regression (Ridge, SVR, RF, ANN, etc.) for Z′ and −Z″ | sklearn, joblib |
| `sensitivity.ipynb` | Sensitivity analysis: RMSE ranking, Pearson correlations, Jacobian-based band sensitivity | torch, pandas |

---

## Recommended Execution Order

1. **Data preparation**
   - `Data_extraction.ipynb` — Extract and preprocess EIS data
   - `dynamic_batch_fitting.ipynb` — Fit equivalent circuit (Rs + Zarc + TL) to spectra

2. **Solution resistance & conductivity**
   - `solution_resistance_batch.ipynb` — Collect Rₛ from fit results
   - `solution_conductivity_batch.ipynb` — Convert to conductivity and analyze degeneracies

3. **Forward modeling**
   - `CENN_forward_model.ipynb` — Train PINN forward model

4. **Inverse & sensitivity**
   - `CENN_inverse.ipynb` — Inverse C,T estimation and frequency selection
   - `sensitivity.ipynb` — Sensitivity and correlation analysis

5. **Classical ML (alternative to PINN)**
   - `regression_ANN.ipynb` — Train and compare classical ML models

---

## Detailed Script Descriptions

### 1. Data_extraction.ipynb

**Purpose:** Cut EIS spectra to a specified frequency range (e.g., 1 kHz–5 kHz), fit a simple model, interpolate, and reorganize folder structure.

**Main operations:**
- Reads EIS CSV files from a directory tree
- Cuts data to a user-defined frequency band
- Fits and interpolates within that band
- Writes `single_frequency_summary.csv` per measurement
- Optionally reorganizes folders (extract single-frequency summaries, move whole-spectrum data)

**Configurable:**
- `ROOT_DIR` — Root directory for EIS data
- Frequency range (e.g., 1000–5000 Hz)
- File glob patterns

**Input:** EIS CSV files with columns such as `Frequency (Hz)`, `Z' (Ω)`, `-Z'' (Ω)` (or similar aliases)

**Output:** `single_frequency_summary.csv` per measurement, optional Nyquist plots

---

### 2. solution_resistance_batch.ipynb

**Purpose:** Collect solution resistance (Rₛ) from fitted data and detect degenerate Rₛ pairs.

**Main operations:**
1. **Collection:** Scans folders for `fit_params_named.csv`, extracts Rₛ, saves to `all_Rs_values.csv`
2. **Degeneracy detection:** Finds (C,T) pairs with nearly identical Rₛ (absolute or relative tolerance)
3. **Filtered degeneracy:** Same as above but only for pairs with different C and T
4. **Visualization:** Bar/line plot of degenerate pairs with dual x-axis labels

**Configurable:**
- `ROOT_DIR` — Root directory
- `ABS_TOL` — Absolute Rₛ difference threshold (Ω)
- `REL_TOL` — Relative tolerance (optional)
- `THRESH` — Threshold for dissimilar-C&T pairs
- `DROP_MEAS` — Average replicates before comparing

**Input:** Directory tree with `fitting/*/fit_params_named.csv` containing `parameter` and `value` columns

**Output:** `all_Rs_values.csv`, `degenerate_Rs_pairs.csv`, `degenerate_Rs_abs<THRESH>.csv`

---

### 3. solution_conductivity_batch.ipynb

**Purpose:** Convert Rₛ (Ω) to conductivity (mS/cm) and find degenerate conductivity pairs.

**Main operations:**
1. **Conversion:** σ (mS/cm) = 1000 / Rₛ (Ω), assuming cell constant K = 1 cm⁻¹
2. **Degeneracy detection:** Same logic as Rₛ, but for conductivity
3. **Advanced:** Supports Darling model–corrected conductivity and percentage error thresholds
4. **Visualization:** Heatmaps and plots of degenerate conductivity pairs

**Configurable:**
- `CSV_PATH` — Path to `all_Rs_values.csv` or conductivity CSV
- `THRESH` — Conductivity difference threshold (mS/cm)

**Input:** `all_Rs_values.csv` (from `solution_resistance_batch.ipynb`)

**Output:** `all_conductivity_values_mScm.csv`, `degenerate_σ_abs<THRESH>.csv`

---

### 4. dynamic_batch_fitting.ipynb

**Purpose:** Physics-aware batch fitting of EIS spectra using the equivalent circuit **Rs + (Rp||CPE) + Transmission Line (TL)**.

**Model:** Z(ω) = Rₛ + Zarc(Rp, Y0, n0) + Z_TL(r, y0, n1, L)

**Main operations:**
- Loads EIS CSV files (flexible column detection)
- Parses concentration (mM) and temperature (°C) from path/filename
- Uses dynamic priors from previously fitted (C,T) neighbors
- Hyperparameter grid: robust loss, high-frequency weighting, prior strength
- Saves fit parameters, Nyquist plots, and raw vs. fit data

**Configurable:**
- `ROOT_DIR`, `FILE_GLOB`
- `TARGET_RMSE`, `MAX_RETRIES`, `JITTER_SCALE`
- `PRIOR_STRENGTH_MIN/MAX`, `NEIGHBOR_WINDOW`
- `FREQ_UNIT_HINT` — "auto", "hz", "khz", "mhz"

**Input:** EIS CSV files matching `FILE_GLOB` (e.g., `EIS_whole_spectrum_*_H2SO4_*C_*kHz_0.1Hz_pH=*.csv`)

**Output:** Per-file: `fitting_TL/<stem>/params_TL.csv`, `raw_vs_fit_TL.csv`, `nyquist_TL.png`; summary: `summary_fitting_TL_physics.csv`

---

### 5. CENN_forward_model.ipynb

**Purpose:** Train a Physics-Informed Neural Network (PINN) that predicts Z′ and −Z″ from (concentration, temperature, frequency).

**Model:** Z(ω) = Rₛ + Zarc(Rp, Y0, n0) + Z_TL(r, y0, n1, L), with θ = [Rs, Rp, Y0, n0, r, y0, n1, L] as a function of (C, T).

**Main operations:**
- Discovers CSV files (nested or flat structure)
- Flexible column resolution for frequency, Z′, −Z″
- Grouped train/test split by (C,T)
- Optional teacher priors from EIS fit parameters
- Physics priors: Arrhenius, monotonicity, θ-invariance
- Saves model, metrics, parity/residual plots, θ(C,T) heatmaps

**Configurable (CONFIG dict):**
- `paths.input_root`, `paths.output_dir`, `paths.fitparams_root`
- `data.train_filters` — concentration/temperature range
- `data.frequency_filter_hz` — single-frequency mode
- `pinn.train` — epochs, lr, width, depth, loss_mode, etc.
- `pinn.teacher` — use teacher priors, weight
- `extrapolation.range_conc_mM`
- `heatmap` — C,T grid for θ heatmaps

**Input:** CSV files with frequency, Z′, −Z″; optional `EIS_FitParams_*mM_*C.csv` for teacher priors

**Output:** `pinn_model.pt`, `compiled_dataset.csv`, `metrics_test.csv`, `test_predictions_pinn.csv`, parity/residual/hist plots, `theta_heatmaps/`, `config_used.json`

---

### 6. CENN_inverse.ipynb

**Purpose:** Inverse estimation of concentration (C) and temperature (T) from EIS spectra using the trained PINN forward model.

**Main operations:**
- Loads trained PINN from `pinn_model.pt`
- Ranks frequencies by Jacobian-based sensitivity (JᵀJ determinant, eigenvalues, angle)
- Supports frequency windows and band-diversity quotas
- Selects top-K frequencies to meet calibration targets (C_MAE, T_MAE)
- Multi-start optimization (Adam + LBFGS) for inverse solve
- Evaluates on calibration and held-out test spectra
- Saves runtime module `ct_inverse_runtime.py` for deployment

**Configurable:**
- `MODEL_PATH`, `DATA_ALL`, `DATA_TEST`, `OUT_DIR`
- `TARGET_C_MAE`, `TARGET_T_MAE` (mM, °C)
- `C_MIN`, `C_MAX`, `T_MIN`, `T_MAX` — training domain bounds
- `K_LIST` — candidate K values (e.g., [1, 3, 6, 10])
- `ALLOWED_WINDOWS` — restrict to frequency windows
- `BANDS` — band-diversity quotas (subHz, 1–100 Hz, 100–1k Hz, 1k–10k Hz)

**Input:** `pinn_model.pt`, `compiled_dataset.csv`, `test_predictions_pinn.csv`

**Output:** `chosen_frequency_set.json`, `global_frequency_ranking.csv`, `calibration_inverse_topK.csv`, `test_inverse_predictions.csv`, `ct_inverse_runtime.py`, `ct_inverse_runtime_config.json`, bar plots

---

### 7. regression_ANN.ipynb

**Purpose:** Classical machine learning regression for predicting Z′ and −Z″ from (C, T, f).

**Models:** Ridge, ElasticNet, PLS, Polynomial Ridge, Huber, TheilSen, SVR, Kernel Ridge, MLP, GPR, Random Forest, Extra Trees, Gradient Boosting, optional Symbolic Regression (PySR/gplearn).

**Main operations:**
- Group-aware train/test split and GroupKFold CV
- RandomizedSearchCV with group-aware cross-validation
- Optional Arrhenius feature engineering
- Learning curves (group-aware)
- Saves best model, all model artifacts, parity/residual/hist plots

**Configurable (CONFIG):**
- `paths.input_root`, `paths.output_dir`
- `data.train_filters`, `features.use_features`, `feature_engineering.arrhenius`
- `cv.n_splits`, `cv.n_iter`
- `models.<name>.enabled`, `param_distributions`

**Input:** Same as CENN_forward_model — CSV files with concentration, temperature, frequency, Z′, −Z″

**Output:** `best_model.joblib`, `model_report.csv`, `compiled_dataset.csv`, `test_predictions_*.csv`, parity/residual/hist/learning plots, `config_used.json`

---

### 8. sensitivity.ipynb

**Purpose:** Sensitivity and correlation analysis for EIS data and PINN models.

**Main operations:**
1. **RMSE ranking:** Find top-K (C,T) pairs with lowest fitting RMSE from `params_TL.csv`
2. **Pearson correlation:** Compute r(C,Z′), r(C,−Z″), r(T,Z′), r(T,−Z″) per frequency band
3. **Jacobian band information:** JᵀJ per band, averaged over (C,T) grid
4. **Jacobian band sensitivity:** Squared sensitivities ∂Z′/∂C, ∂Z′/∂T, ∂(−Z″)/∂C, ∂(−Z″)/∂T per band
5. **Band sensitivity decomposition:** Share of Z′ and −Z″ sensitivity per band; C vs. T contributions
6. **Dimensionless sensitivity:** Normalize by ΔC, ΔT, and Z′/Z″ RMS

**Configurable:**
- `ROOT` — Path to `params_TL.csv` files
- `MODEL_PATH`, `DATA_PATH` — PINN and EIS data
- `FREQ_RANGES` — Frequency bands for Pearson analysis
- `BANDS` — Band definitions for Jacobian analysis
- `conc_range`, `temp_range`, `TOP_K`

**Input:** `params_TL.csv`, `combined_eis.csv`, `pinn_model.pt`

**Output:** `pearson_by_freq_range.csv`, `pearson_heatmap_by_range.png`, `jacobian_band_information.csv`, `jacobian_band_sensitivity_Zr_Zim.csv`, `band_sensitivity_decomposed.csv`, `band_dimensionless_sensitivity.csv`

---

## Data Formats & Paths

### EIS CSV Format

Expected columns (flexible aliases supported in most scripts):

| Column | Aliases |
|--------|---------|
| Frequency | `Frequency (Hz)`, `frequency (hz)`, `freq (hz)`, `f (hz)`, `f_hz` |
| Z′ (real) | `Z' (Ω)`, `z_real`, `zre`, `re(z)` |
| −Z″ (imag) | `-Z'' (Ω)`, `-z_imag`, `-zim`, `Z_imag_neg` |

### Fit Parameters Format

For `fit_params_named.csv` or `params_TL.csv`:

- `parameter` / `value` columns, or
- Unnamed columns with RMSE in 5th column (index 4)

### Path Conventions

- Concentration/temperature are often parsed from folder or file names, e.g.:
  - `10mM`, `15mM`
  - `26C`, `34C`, `40°C`
- Nested structure: `ROOT/<conc>/<temp>/.../single_frequency_summary.csv`
- Flat structure: `ROOT/**/*.csv` with matching column names

---

## Output Artifacts

| Script | Key Outputs |
|--------|-------------|
| Data_extraction | `single_frequency_summary.csv`, fit plots |
| solution_resistance_batch | `all_Rs_values.csv`, `degenerate_Rs_*.csv` |
| solution_conductivity_batch | `all_conductivity_values_mScm.csv`, `degenerate_σ_*.csv` |
| dynamic_batch_fitting | `params_TL.csv`, `summary_fitting_TL_physics.csv` |
| CENN_forward_model | `pinn_model.pt`, `compiled_dataset.csv`, metrics, plots |
| CENN_inverse | `chosen_frequency_set.json`, `ct_inverse_runtime.py` |
| regression_ANN | `best_model.joblib`, `model_report.csv` |
| sensitivity | Pearson CSVs, Jacobian CSVs, heatmaps |

---

## Troubleshooting

### Path errors

- Update `ROOT_DIR`, `MODEL_PATH`, `DATA_PATH`, etc. to match your system.
- Use absolute paths or paths relative to the notebook’s working directory.

### Column resolution

- Add your column names to `column_aliases` in the CONFIG or use the scripts’ flexible matching.
- Ensure encoding is UTF-8 where needed (`read_csv_kwargs={"encoding": "utf-8"}`).

### PyTorch `weights_only`

- Newer PyTorch uses `weights_only=True` by default for `torch.load`.
- Scripts use `weights_only=False` where custom objects are stored; adjust if you see related errors.

### CUDA / GPU

- Set `device="cuda"` or `device="auto"` in PINN configs when a GPU is available.
- Fallback to CPU if CUDA is not installed.

### Symbolic regression

- `regression_ANN.ipynb` disables symbolic models if neither `pysr` nor `gplearn` is installed.
- Install one of them and re-run if you need symbolic regression.

---

## Citation

If you use these scripts in your research, please cite the accompanying paper and this repository.

---

## License

See the project root for license information.
