# Panel-DOSE: heterogeneous effects of digital adoption in panel data

Replication package for the manuscript

> **Searching for the digital dividend: Internet diffusion and productivity growth in 141 economies, 1996-2025**
> Quang-Vinh Dang, British University Vietnam (submitted to *Journal of Digital Economy*).

## Contents

| Path | Description |
|------|-------------|
| `code/panel_dose.py` | Panel-DOSE estimator (Mundlak-augmented, country-level cross-fitting, two-way within transformation, O'Sullivan-penalised B-spline sieve with leave-countries-out CV, cluster-robust and sup-t inference, shape tests, influence functions); installable with `pip install .` |
| `code/download_wdi.py` | Downloads the World Development Indicators series (World Bank API) |
| `code/data_prep.py` | Builds the main WDI/UNDP panel (1996-2025) and the PWT 11.0 (1996-2023) and PWT 10.0 (1996-2019) panels |
| `code/simulation.py` | Monte Carlo study (12 estimators incl. ablations and an oracle, 4 effect shapes, sensitivity to sample size and confounding) |
| `code/empirical.py` | Empirical application: internet diffusion and labour-productivity growth, 141 countries, 1996-2025, with dynamics, PWT 11.0/10.0 comparisons and robustness checks |
| `code/empirical_post.py` | Dose response, two-way clustered standard errors, Holm adjustment and minimum detectable effects |
| `code/plasmode.py` | Plasmode simulation built on the empirical panel |
| `code/make_outputs.py` | Produces all LaTeX tables and figures |
| `code/sync_tables.py` | Copies the generated tables into `manuscript/main.tex` |
| `data/raw/` | WDI series (`wdi/`), UNDP HDR 2025 time series, Penn World Table 11.0 (`pwt110.xlsx`) and 10.0 (`pwt100.xlsx`), ITU series via Our World in Data |
| `data/processed/panel.csv` | Main estimation sample (141 countries, 1996-2025) |
| `data/processed/panel_pwt11.csv` | PWT 11.0 robustness sample (117 countries, 1996-2023) |
| `data/processed/panel_pwt.csv` | PWT 10.0 robustness sample (113 countries, 1996-2019) |
| `data/processed/panel_untrimmed.csv`, `panel_unscreened.csv` | Samples for the trimming and data-screening checks |
| `data/processed/screening_log.csv`, `coverage.csv` | Flagged adoption values; sample coverage by income group |
| `data/raw/wdi/SHA256SUMS` | Checksums of the archived WDI snapshot (retrieved September 2026) |
| `results/` | Raw simulation draws, summaries and empirical estimates |
| `manuscript/` | LaTeX source (`main.tex`, `refs.bib`), tables, figures, compiled PDF, highlights |

## Reproducing the results

```bash
pip install -r requirements.txt
python code/download_wdi.py     # data/raw/wdi/*.csv (skip if already present)
python code/data_prep.py        # data/processed/panel.csv, panel_pwt11.csv, panel_pwt.csv
python code/simulation.py 100 4 # results/simulation_raw.csv (about 1 h on 4 cores)
python code/empirical.py 4      # results/empirical_summary.csv, results/empirical.json (4 = cores)
python code/empirical_post.py   # appends dose response, two-way SEs, Holm, MDE
python code/plasmode.py 100 4   # results/plasmode_raw.csv
python code/make_outputs.py     # manuscript/tables/*.tex, manuscript/figures/*.pdf
python code/sync_tables.py      # refresh tables inside manuscript/main.tex
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Using the estimator

```python
from panel_dose import PanelDOSE
import lightgbm as lgb

est = PanelDOSE(lambda: lgb.LGBMRegressor(n_estimators=400, verbose=-1),
                mode="vc", z_in_controls=True, n_rep=5)
est.fit(df, y="dlp", d="internet_l1", controls=controls, unit="iso", time="year", z="hc_l1")
curve = est.effect(grid)          # theta(z) with pointwise and simultaneous 95% bands
avg, se = est.average_effect()    # average effect
diff, se_diff = est.group_contrast(mask_high, mask_low)
```

## Data sources

* World Bank, World Development Indicators (July 2026 release), https://databank.worldbank.org
* UNDP, Human Development Report 2025, composite indices time series, https://hdr.undp.org

* Feenstra, R. C., Inklaar, R., Timmer, M. P. (2015). The next generation of the Penn World Table. *American Economic Review* 105(10), 3150-3182. Data: https://www.rug.nl/ggdc/productivity/pwt/
* International Telecommunication Union and World Bank, World Development Indicators, distributed by Our World in Data (https://ourworldindata.org).
