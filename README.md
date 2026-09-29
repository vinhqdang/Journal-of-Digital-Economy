# Panel-DOSE: heterogeneous effects of digital adoption in panel data

Replication package for the manuscript

> **Where does the digital dividend arrive? A debiased machine-learning sieve estimator for heterogeneous effects of digital adoption in panel data**
> Quang-Vinh Dang, British University Vietnam (submitted to *Journal of Digital Economy*).

## Contents

| Path | Description |
|------|-------------|
| `code/panel_dose.py` | Panel-DOSE estimator (Mundlak-augmented, country-level cross-fitting, two-way within transformation, penalised B-spline sieve, cluster-robust and sup-t inference) |
| `code/data_prep.py` | Builds the country-year panel from the raw files |
| `code/simulation.py` | Monte Carlo study (7 estimators, 4 effect shapes, sensitivity to sample size and confounding) |
| `code/empirical.py` | Empirical application: internet adoption and labour-productivity growth, 110 countries, 1996-2019 |
| `code/make_outputs.py` | Produces all LaTeX tables and figures |
| `data/raw/` | Penn World Table 10.0 and ITU/World Bank series (via Our World in Data) |
| `data/processed/panel.csv` | Estimation sample |
| `results/` | Raw simulation draws, summaries and empirical estimates |
| `manuscript/` | LaTeX source (`main.tex`, `refs.bib`), tables, figures, compiled PDF, highlights |

## Reproducing the results

```bash
pip install -r requirements.txt
python code/data_prep.py        # data/processed/panel.csv
python code/simulation.py 100   # results/simulation_raw.csv (about 1-2 h on 4 cores)
python code/empirical.py        # results/empirical_summary.csv, results/empirical.json
python code/make_outputs.py     # manuscript/tables/*.tex, manuscript/figures/*.pdf
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

* Feenstra, R. C., Inklaar, R., Timmer, M. P. (2015). The next generation of the Penn World Table. *American Economic Review* 105(10), 3150-3182. Data: https://www.rug.nl/ggdc/productivity/pwt/
* International Telecommunication Union and World Bank, World Development Indicators, distributed by Our World in Data (https://ourworldindata.org).
