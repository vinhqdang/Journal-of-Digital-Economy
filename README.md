# Panel-DOSE: heterogeneous associations of digital adoption in panel data

Replication package for the manuscript

> **Searching for the digital dividend: Internet diffusion and productivity growth in 149 economies, 1996-2025**
> Quang-Vinh Dang, British University Vietnam (submitted to *Journal of Digital Economy*).

## Contents

| Path | Description |
|------|-------------|
| `code/panel_dose.py` | Panel-DOSE estimator (Mundlak-augmented, country-level cross-fitting, two-way within transformation, O'Sullivan-penalised B-spline sieve with leave-countries-out CV, functional-wise median aggregation across sample splits, cluster-robust and two-way clustered inference, sup-t bands and shape tests, influence diagnostics); installable with `pip install .` |
| `code/verify.py` | Checks the SHA-256 checksums of the archived data snapshot (`data`) and the headline numbers (`results`) |
| `code/download_wdi.py` | Downloads the World Development Indicators series (World Bank API); skips files that already exist |
| `code/data_prep.py` | Builds the WDI/UNDP panel (1996-2025), the PWT 11.0 (1996-2023) and PWT 10.0 (1996-2019) panels, the robustness samples and the sample-flow table |
| `code/empirical.py` | Empirical application: baseline, robustness checks, local projections, long difference, lead test, IV check |
| `code/empirical_post.py` | Two-way clustered standard errors, Holm adjustment and minimum detectable effects |
| `code/compare.py` | Heterogeneity across data sources on common country-years (`pwt`: PWT 10.0 vs 11.0; `wdi`: WDI vs PWT 11.0): component replacement, Shapley attribution, fixed-smoother variant, split-seed distribution, paired country bootstrap (copies of a country kept in one fold), leave-one-country-out refits |
| `code/diagnostics.py` | Split-seed distribution of the baseline, out-of-fold nuisance fit, sensitivity to LightGBM hyper-parameters |
| `code/make_anonymous_package.py` | Builds `submission/replication_anonymous.zip`, an anonymised copy of this package for double-blind review |
| `code/invariance_check.py` | Numerical check of Proposition 1(ii) |
| `code/plasmode.py` | Plasmode simulation on the empirical panel, for each of the three moderators |
| `code/simulation.py` | Stylised Monte Carlo study (12 estimators, 4 effect shapes, sensitivity to sample size and confounding) |
| `code/make_outputs.py` | Produces all LaTeX tables, figures and `manuscript/tables/numbers.tex` (every number quoted in the text) |
| `code/sync_tables.py` | Copies the generated tables into `manuscript/main.tex` and `manuscript/supplement.tex` |
| `code/make_springer.py` | Builds a Springer Nature (`sn-jnl`) version `manuscript/springer/main_sn.tex` |
| `data/raw/` | WDI snapshot (`wdi/`, with `SHA256SUMS`), UNDP HDR 2025 time series, Penn World Table 11.0 (`pwt110.xlsx`) and 10.0 (`pwt100.xlsx`), ITU series via Our World in Data |
| `data/processed/` | Estimation samples (`panel.csv` baseline, `panel_common.csv`, `panel_untrimmed.csv`, `panel_unscreened.csv`, `panel_pwt11.csv`, `panel_pwt.csv`), `sample_flow.csv`, `screening_log.csv`, `coverage.csv` |
| `results/` | Raw simulation draws, bootstrap draws, summaries and empirical estimates |
| `manuscript/` | `main.tex` (manuscript), `supplement.tex` (supplementary material), `main_blind.tex` and `supplement_blind.tex` (anonymised versions), `title_page.tex`, `refs.bib`, tables, figures, highlights, `build.sh` |
| `submission/` | Files for submission, written by `manuscript/build.sh`: `Digital_Dividend_Manuscript_anonymised.pdf`, `Digital_Dividend_Supplementary_Material_anonymised.pdf` and `Digital_Dividend_Title_Page.pdf` (double-blind review), `Digital_Dividend_Manuscript_with_author_details.pdf` and `Digital_Dividend_Supplementary_Material_with_author_details.pdf`, and the anonymised replication archive `replication_anonymous.zip` |
| `requirements.txt`, `requirements-lock.txt` | Minimum versions, and the exact versions used for the reported results |

## Reproducing the results from the archived snapshot

```bash
pip install -r requirements-lock.txt
python code/verify.py data          # checksums of the archived WDI snapshot
python code/data_prep.py            # data/processed/*.csv
python code/empirical.py 4          # results/empirical_summary.csv, results/empirical.json (4 = cores)
python code/empirical_post.py       # two-way SEs, Holm, MDE
python code/compare.py pwt 4 199    # results/compare_pwt.json (PWT 10.0 vs 11.0)
python code/compare.py wdi 4 199    # results/compare_wdi.json (WDI vs PWT 11.0)
python code/diagnostics.py 4        # results/diagnostics.json
python code/invariance_check.py     # results/invariance.json
python code/plasmode.py 200 4       # results/plasmode_raw.csv
python code/simulation.py 100 4     # results/simulation_raw.csv
python code/make_outputs.py         # tables, figures, numbers.tex
python code/sync_tables.py          # refresh tables inside the LaTeX sources
python code/verify.py results       # compare with the numbers reported in the paper
sh manuscript/build.sh              # PDFs (pdflatex + bibtex), copied to submission/
```

On four cores the estimation scripts take about four hours in total. Main estimates use 20
repetitions of the sample split and one cross-validated penalty pooled over the splits;
robustness checks, the bootstrap and the plasmode use ten. `code/download_wdi.py`
is needed only to rebuild the snapshot from the live World Bank API; because WDI is revised
continuously, a new download will not reproduce the archived numbers exactly.

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
tests = est.shape_tests()         # sup-t tests of constancy and linearity
```

## Data sources

* World Bank, World Development Indicators (July 2026 release), https://databank.worldbank.org
* UNDP, Human Development Report 2025, composite indices time series, https://hdr.undp.org
* Feenstra, R. C., Inklaar, R., Timmer, M. P. (2015). The next generation of the Penn World Table. *American Economic Review* 105(10), 3150-3182. Data: https://www.rug.nl/ggdc/productivity/pwt/ (PWT 11.0: doi:10.34894/FABVLR)
* International Telecommunication Union and World Bank, World Development Indicators, distributed by Our World in Data (https://ourworldindata.org).
