"""Turn simulation and empirical results into the figures and LaTeX tables of the manuscript."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "manuscript" / "figures"
TAB = ROOT / "manuscript" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
C_MAIN, C_ALT, C_GREY, C_3 = "#1f4e79", "#c0504d", "#7f7f7f", "#9bbb59"

ORDER = ["TWFE", "TWFE-interaction", "TWFE-spline", "FE-DML (constant)",
         "FE-DML + R-learner (tuned)", "WG-DML sieve", "Pooled DML-sieve",
         "Ablation: Mundlak only", "Ablation: within only", "Panel-DOSE (unpenalised)",
         "Panel-DOSE", "Panel-DOSE (oracle nuisance)"]
SHAPES = ["constant", "linear", "threshold", "hump"]


def fmt(x, d=2):
    return "--" if x is None or pd.isna(x) else f"{x:.{d}f}"


def star(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""


def pstar(est, se):
    return star(2 * stats.norm.sf(abs(est / se))) if se and se > 0 else ""


def cov_cell(s):
    """Coverage mean with Monte Carlo standard error in parentheses."""
    s = s.dropna()
    if len(s) == 0:
        return "--"
    return f"{s.mean():.2f} ({s.std(ddof=1) / np.sqrt(len(s)):.2f})"


# ----------------------------------------------------------------------------- simulation
def sim_tables():
    raw = pd.read_csv(RES / "simulation_raw.csv")
    base = raw[(raw["N"] == 100) & (raw["rho"] == 1.0)]
    g = base.groupby(["shape", "method"])
    agg = g.agg(irmse=("irmse", "mean"), ibias=("ibias", "mean"),
                ate_rmse=("ate_err", lambda e: np.sqrt(np.mean(e ** 2))),
                contrast_rmse=("contrast_err", lambda e: np.sqrt(np.mean(e ** 2))),
                reps=("rep", "nunique")).reset_index()
    agg.to_csv(RES / "simulation_summary.csv", index=False)
    reps = int(agg["reps"].min())

    lines = []
    for mth in ORDER:
        sub = agg[agg["method"] == mth].set_index("shape")
        cells = []
        for key in ["irmse", "ibias", "contrast_rmse"]:
            for sh in SHAPES:
                v = sub.loc[sh, key]
                best = agg[agg["shape"] == sh][key].abs().min()
                c = fmt(v)
                if key == "irmse" and np.isclose(abs(v), best) and mth != "Panel-DOSE (oracle nuisance)":
                    c = r"\textbf{" + c + "}"
                cells.append(c)
        lines.append(f"{mth} & " + " & ".join(cells) + r"\\")
    (TAB / "sim_main.tex").write_text("\n".join(lines) + "\n")

    inf_methods = ["TWFE-interaction", "TWFE-spline", "FE-DML (constant)", "WG-DML sieve",
                   "Pooled DML-sieve", "Ablation: Mundlak only", "Ablation: within only",
                   "Panel-DOSE (unpenalised)", "Panel-DOSE", "Panel-DOSE (oracle nuisance)"]
    lines = []
    for sh in SHAPES:
        lines.append(r"\multicolumn{6}{l}{\textit{Design: " + sh + r" $\theta(z)$}}\\")
        for mth in inf_methods:
            s = base[(base["shape"] == sh) & (base["method"] == mth)]
            cells = [cov_cell(s[c]) if c in s else "--"
                     for c in ["cover", "ucover", "ate_cover", "contrast_cover", "blp_cover"]]
            lines.append(f"\\quad {mth} & " + " & ".join(cells) + r"\\")
        lines.append(r"\addlinespace")
    (TAB / "sim_cover.tex").write_text("\n".join(lines[:-1]) + "\n")

    sens = raw[raw["shape"] == "threshold"]
    cells = [(50, 1.0), (100, 0.25), (100, 0.5), (100, 1.0), (100, 2.0), (200, 1.0)]
    meths = ["TWFE-interaction", "TWFE-spline", "FE-DML + R-learner (tuned)",
             "Panel-DOSE (unpenalised)", "Panel-DOSE"]
    lines = [r"\multicolumn{7}{l}{\textit{IRMSE}}\\"]
    for mth in meths:
        vals = [sens[(sens["N"] == n) & (sens["rho"] == r) & (sens["method"] == mth)]["irmse"].mean()
                for n, r in cells]
        lines.append(f"\\quad {mth} & " + " & ".join(fmt(v) for v in vals) + r"\\")
    for lab, col in [("Pointwise coverage", "cover"), ("Uniform coverage", "ucover"),
                     ("Coverage of high--low contrast", "contrast_cover")]:
        lines.append(r"\addlinespace\multicolumn{7}{l}{\textit{" + lab + r"}}\\")
        for mth in ["TWFE-interaction", "Panel-DOSE (unpenalised)", "Panel-DOSE"]:
            vals = [cov_cell(sens[(sens["N"] == n) & (sens["rho"] == r) & (sens["method"] == mth)][col])
                    for n, r in cells]
            lines.append(f"\\quad {mth} & " + " & ".join(vals) + r"\\")
    (TAB / "sim_sens.tex").write_text("\n".join(lines) + "\n")

    dose = base[base["method"] == "Panel-DOSE"]
    lam = {"reps": reps, "lambda_at_max": float(np.mean(dose["lambda"] >= 1e4 - 1)),
           "lambda_zero": float(np.mean(dose["lambda"] == 0)),
           "edf_mean": float(dose["edf"].mean())}
    for sh in SHAPES:
        d = dose[dose["shape"] == sh]
        lam[f"p_const_reject_{sh}"] = float(np.mean(d["p_const"] < 0.05))
    json.dump(lam, open(RES / "simulation_lambda.json", "w"), indent=1)
    return agg


def sim_figures(agg):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8), sharey=True)
    short = {"TWFE": "TWFE", "TWFE-interaction": "TWFE-int", "TWFE-spline": "TWFE-spline",
             "FE-DML (constant)": "FE-DML", "FE-DML + R-learner (tuned)": "R-learner",
             "WG-DML sieve": "WG-sieve", "Pooled DML-sieve": "Pooled",
             "Ablation: Mundlak only": "Mundlak only", "Ablation: within only": "Within only",
             "Panel-DOSE (unpenalised)": "DOSE-u", "Panel-DOSE": "DOSE",
             "Panel-DOSE (oracle nuisance)": "DOSE-oracle"}
    for ax, sh in zip(axes, SHAPES):
        sub = agg[agg["shape"] == sh].set_index("method").loc[ORDER]
        cols = [C_MAIN if m.startswith("Panel-DOSE") else C_3 if m.startswith("Ablation") else C_GREY
                for m in ORDER]
        ax.barh([short[m] for m in ORDER], sub["irmse"], color=cols)
        ax.set_title(sh, fontsize=9)
        ax.set_xlabel("IRMSE")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "sim_irmse.pdf")
    plt.close(fig)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import simulation as sim
    from panel_dose import PanelDOSE
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=True)
    for ax, sh in zip(axes, SHAPES):
        df = sim.simulate(sh, seed=424242)
        m = PanelDOSE(sim.lgbm, n_rep=2, z_in_controls=True).fit(df, "y", "d", sim.CONTROLS,
                                                                 "unit", "t", z="z")
        e = m.effect(sim.GRID)
        tw = sim.twfe(df, "quad")["est"]
        ax.fill_between(e["grid"], e["ulo"], e["uhi"], color=C_MAIN, alpha=0.15, lw=0)
        ax.fill_between(e["grid"], e["lo"], e["hi"], color=C_MAIN, alpha=0.25, lw=0)
        ax.plot(e["grid"], sim.theta_fun(sh, e["grid"].values), "k--", lw=1, label="truth")
        ax.plot(e["grid"], e["est"], color=C_MAIN, lw=1.5, label="Panel-DOSE")
        ax.plot(sim.GRID, tw, color=C_ALT, lw=1, label="TWFE-interaction")
        ax.set_title(sh, fontsize=9)
        ax.set_xlabel("$z$")
    axes[0].set_ylabel(r"$\theta(z)$")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "sim_curves.pdf")
    plt.close(fig)


def plasmode_table():
    f = RES / "plasmode_raw.csv"
    if not f.exists():
        return
    p = pd.read_csv(f)
    p["shape"] = p["shape"].fillna("null")  # the label "null" is read as missing by pandas
    lines = []
    for sh, lab in [("null", "No effect"), ("linear", "Linear gradient"), ("threshold", "Threshold")]:
        lines.append(r"\multicolumn{8}{l}{\textit{" + lab + r"}}\\")
        for mth in ["TWFE-interaction", "TWFE-spline", "Panel-DOSE (unpenalised)", "Panel-DOSE"]:
            s = p[(p["shape"] == sh) & (p["method"] == mth)]
            cells = [fmt(s["irmse"].mean()) if "irmse" in s and s["irmse"].notna().any() else "--",
                     fmt(s["diff_err"].mean()), fmt(np.sqrt(np.mean(s["diff_err"] ** 2))),
                     cov_cell(s["diff_cover"]), cov_cell(s["reject_diff"]),
                     cov_cell(s["reject_const"]) if s["reject_const"].notna().any() else "--",
                     (f"{np.mean(s['lambda'] == 0):.2f}/{np.mean(s['lambda'] >= 1e4 - 1):.2f}"
                      if s["lambda"].notna().any() and mth == "Panel-DOSE" else "--")]
            lines.append(f"\\quad {mth} & " + " & ".join(cells) + r"\\")
        lines.append(r"\addlinespace")
    (TAB / "plasmode.tex").write_text("\n".join(lines[:-1]) + "\n")


# ----------------------------------------------------------------------------- empirical
MAIN_ROWS = {
    "A": ["Baseline", "Original specification", "Adding mobile and broadband",
          "Treatment: composite ICT index", "Random forest nuisance", "Pre-COVID sample, 1996-2019",
          "Outcome: GDP per capita growth", "Moderator: learning-adjusted schooling",
          "PWT 11.0, 1996-2023", "PWT 10.0, 1996-2019"],
    "B": ["Baseline", "Original specification", "Adding mobile and broadband",
          "Treatment: composite ICT index", "Random forest nuisance", "Pre-COVID sample, 1996-2019",
          "Moderator: distance to US frontier", "WDI data, PWT 11.0 countries, 1996-2023",
          "PWT 11.0, 1996-2023", "PWT 10.0, 1996-2019"],
    "C": ["Baseline", "Random forest nuisance", "Pre-COVID sample, 1996-2019",
          "Adding mobile and broadband"],
}
PANEL_TITLES = {"A": "Panel A. Moderator: mean years of schooling",
                "B": "Panel B. Moderator: initial log GDP per worker",
                "C": "Panel C. Moderator: years since internet take-off (share $\\geq$ 10\\%)"}


def emp_row(r):
    cells = [f"${r.fedml:.2f}{pstar(r.fedml, r.fedml_se)}$"]
    for lab in ["low", "mid", "high"]:
        cells.append(f"${r[f'gate_{lab}']:.2f}{pstar(r[f'gate_{lab}'], r[f'gate_{lab}_se'])}$")
    cells.append(f"${r['diff']:.2f}{pstar(r['diff'], r['diff_se'])}$")
    cells.append(f"${r.blp:.2f}{pstar(r.blp, r.blp_se)}$")
    cells.append(f"{r.p_const:.2f}")
    se = [f"({r.fedml_se:.2f})"] + [f"({r[f'gate_{lab}_se']:.2f})" for lab in ["low", "mid", "high"]] + \
         [f"({r.diff_se:.2f})", f"({r.blp_se:.2f})", ""]
    return cells, se


def emp_tables(summ):
    def write(fname, rows_by_panel):
        lines = []
        for P in ["A", "B", "C"]:
            names = rows_by_panel.get(P, [])
            if not names:
                continue
            lines.append(r"\multicolumn{8}{l}{\textit{" + PANEL_TITLES[P] + r"}}\\")
            sub = summ[summ["panel"] == P].set_index("spec")
            for n in names:
                if n not in sub.index:
                    continue
                cells, se = emp_row(sub.loc[n])
                lines.append(f"\\quad {n} & " + " & ".join(cells) + r"\\")
                lines.append(" & " + " & ".join(se) + r"\\")
            lines.append(r"\addlinespace")
        (TAB / fname).write_text("\n".join(lines[:-1]) + "\n")

    write("emp_main.tex", MAIN_ROWS)
    rest = {P: [s for s in summ[summ["panel"] == P]["spec"] if s not in MAIN_ROWS.get(P, [])]
            for P in ["A", "B", "C"]}
    write("emp_appendix.tex", rest)

    # diagnostics: penalty, identifying variation, support
    lines = []
    for P, n in [("A", "Baseline"), ("A", "Original specification"), ("B", "Baseline"),
                 ("B", "Original specification"), ("C", "Baseline")]:
        r = summ[(summ["panel"] == P) & (summ["spec"] == n)].iloc[0]
        lines.append(f"{P}: {n} & {r['lambda']:.3g} & {r.edf:.1f} & {r.idvar_all:.2f} & "
                     f"{r.idvar_low:.2f} / {r.idvar_mid:.2f} / {r.idvar_high:.2f} & "
                     f"{r.d_median_low:.2f} / {r.d_median_mid:.2f} / {r.d_median_high:.2f} & "
                     f"{r.d_within_sd_low:.2f} / {r.d_within_sd_mid:.2f} / {r.d_within_sd_high:.2f} & "
                     f"{int(r.n_countries_low)} / {int(r.n_countries_mid)} / {int(r.n_countries_high)}\\\\")
    (TAB / "emp_diag.tex").write_text("\n".join(lines) + "\n")


def dyn_tables(summ, ex):
    lp = summ[summ["panel"] == "LP"].copy()
    lp["h"] = lp["spec"].str[2:].astype(int)
    lp = lp.sort_values("h")
    lines = []
    for _, r in lp.iterrows():
        lines.append(f"$h={r.h}$ & ${r.fedml:.2f}{pstar(r.fedml, r.fedml_se)}$ & ({r.fedml_se:.2f}) & "
                     f"${r['diff']:.2f}{pstar(r['diff'], r['diff_se'])}$ & ({r.diff_se:.2f}) & "
                     f"{r.n_obs}\\\\")
    (TAB / "emp_lp.tex").write_text("\n".join(lines) + "\n")
    fy, ld, iv = ex["five_year"], ex["long_difference"], ex["iv"]
    cz = summ[summ["panel"] == "CZ"].set_index("spec")
    lines = [
        f"Five-year averages, TWFE & ${fy['twfe']:.2f}{pstar(fy['twfe'], fy['twfe_se'])}$ & ({fy['twfe_se']:.2f}) & {fy['n_obs']} \\\\",
        f"Five-year averages, FE-DML & ${fy['fedml']:.2f}{pstar(fy['fedml'], fy['fedml_se'])}$ & ({fy['fedml_se']:.2f}) & {fy['n_obs']} \\\\",
        f"Long difference 2000--2024, OLS & ${ld['ols']:.2f}{pstar(ld['ols'], ld['ols_se'])}$ & ({ld['ols_se']:.2f}) & {ld['n']} \\\\",
        f"Long difference 2000--2024, DML & ${ld['dml']:.2f}{pstar(ld['dml'], ld['dml_se'])}$ & ({ld['dml_se']:.2f}) & {ld['n']} \\\\",
        f"Fixed-line IV (2SLS), first-stage $F={iv['first_stage_F']:.1f}$ & ${iv['iv']:.2f}{pstar(iv['iv'], iv['iv_se'])}$ & ({iv['iv_se']:.2f}) & {iv['n_obs']} \\\\",
    ]
    for n, lab in [("High income, broadband, GDP per capita", "Broadband, GDP p.c. growth, high income"),
                   ("All countries, broadband, GDP per capita", "Broadband, GDP p.c. growth, all countries")]:
        r = cz.loc[n]
        lines.append(f"{lab} & ${r.fedml:.2f}{pstar(r.fedml, r.fedml_se)}$ & ({r.fedml_se:.2f}) & {r.n_obs} \\\\")
    (TAB / "emp_dyn.tex").write_text("\n".join(lines) + "\n")

    v = ex["vintage"]
    lines = []
    for r in v["rows"]:
        lines.append(f"{r['combo']} & ${r['gate_low']:.2f}{pstar(r['gate_low'], r['gate_low_se'])}$ & "
                     f"({r['gate_low_se']:.2f}) & ${r['diff']:.2f}{pstar(r['diff'], r['diff_se'])}$ & "
                     f"({r['diff_se']:.2f})\\\\")
    (TAB / "emp_vintage.tex").write_text("\n".join(lines) + "\n")


def band(ax, c, color, label, uniform=True, bands=None):
    """Line: estimate in c; shaded bands: from `bands` (the unpenalised sieve) if given."""
    g = np.array(c["grid"])
    b = bands if bands is not None else c
    ax.fill_between(g, b["lo"], b["hi"], color=color, alpha=0.25, lw=0)
    if uniform:
        ax.fill_between(g, b["ulo"], b["uhi"], color=color, alpha=0.12, lw=0)
    ax.plot(g, c["est"], color=color, lw=1.6, label=label)
    ax.axhline(0, color="k", lw=0.5)


def emp_figures(summ, js):
    cur, ex = js["curves"], js["extra"]
    df = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, key, z, tw, xl, tt in [
            (axes[0], "A|Baseline", "hc_l1", "twfe_curve_hc", "Mean years of schooling (lagged)",
             "(a) by schooling"),
            (axes[1], "B|Baseline", "lp_init", "twfe_curve_lp", "Initial log GDP per worker",
             "(b) by initial productivity"),
            (axes[2], "C|Baseline", "ysince_l1", None, "Years since internet take-off",
             "(c) by years since take-off")]:
        band(ax, cur[key]["pen"], C_MAIN, "Panel-DOSE")
        lo_, hi_ = np.nanpercentile(np.r_[cur[key]["pen"]["ulo"], cur[key]["pen"]["uhi"]], [3, 97])
        ax.set_ylim(min(lo_, -5), max(hi_, 5))
        if tw:
            ax.plot(ex[tw]["grid"], ex[tw]["est"], color=C_ALT, lw=1, ls="--", label="TWFE-interaction")
        lo, hi = ax.get_ylim()
        ax.plot(df[z], np.full(len(df), lo), "|", color=C_GREY, alpha=0.08, ms=6)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xl)
        ax.set_title(tt, fontsize=9)
    axes[0].set_ylabel(r"$\hat\theta$: pp growth per unit of adoption")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "emp_main.pdf")
    plt.close(fig)

    # dose response and local projections
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    if "dose" in ex:
        c = {k: v[3:] for k, v in ex["dose"]["curve"].items()}
        cu = {k: v[3:] for k, v in ex["dose"]["curve_u"].items()}
        band(axes[0], c, C_MAIN, "Panel-DOSE")
        axes[0].set_xlabel("Internet users (share, lagged)")
        axes[0].set_ylabel(r"$\hat f'(d)$")
        axes[0].set_title("(a) dose response", fontsize=9)
    lp = summ[summ["panel"] == "LP"].copy()
    lp["h"] = lp["spec"].str[2:].astype(int)
    lp = lp.sort_values("h")
    axes[1].errorbar(lp["h"], lp["fedml"] / 10, yerr=1.96 * lp["fedml_se"] / 10, fmt="o-",
                     color=C_MAIN, ms=3, capsize=2, label="Average (FE-DML)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Horizon $h$ (years after $t-1$)")
    axes[1].set_ylabel("Cumulative pp per 10 pp internet")
    axes[1].set_title("(b) local projections", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "emp_dyn.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    base = cur["A|Baseline"]["pen"]
    axes[0].fill_between(base["grid"], base["lo"], base["hi"], color=C_MAIN, alpha=0.2, lw=0)
    axes[0].plot(base["grid"], base["est"], color=C_MAIN, lw=2, label="Baseline")
    for k in cur:
        if k.startswith("A|") and k not in ("A|Baseline", "A|Moderator: learning-adjusted schooling",
                                            "A|Treatment: composite ICT index", "A|Broadband"):
            axes[0].plot(cur[k]["pen"]["grid"], cur[k]["pen"]["est"], lw=0.7, alpha=0.7)
    axes[0].set_xlabel("Mean years of schooling (lagged)")
    axes[0].set_ylabel(r"$\hat\theta(z)$")
    axes[0].set_title("(a) schooling: WDI robustness checks", fontsize=9)
    axes[0].set_ylim(-15, 12)
    axes[0].axhline(0, color="k", lw=0.5)
    for key, col, lab in [("B|Baseline", C_MAIN, "WDI (baseline)"),
                          ("B|PWT 11.0, 1996-2023", C_3, "PWT 11.0"),
                          ("B|PWT 10.0, 1996-2019", C_ALT, "PWT 10.0")]:
        c = cur[key]["pen"]
        axes[1].fill_between(c["grid"], c["lo"], c["hi"], color=col, alpha=0.12, lw=0)
        axes[1].plot(c["grid"], c["est"], color=col, lw=1.5, label=lab)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Initial log productivity")
    axes[1].set_title("(b) initial productivity: data sources", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "emp_robust.pdf")
    plt.close(fig)

    # descriptive figure
    df["grp"] = pd.qcut(df["lp_init"], 3, labels=["Low", "Middle", "High"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for lab, col in zip(["Low", "Middle", "High"], [C_ALT, C_GREY, C_MAIN]):
        g = df[df["grp"] == lab].groupby("year")
        axes[0].plot(g["internet_l1"].mean().index, 100 * g["internet_l1"].mean(), color=col,
                     label=f"{lab} initial productivity")
        s = g["dlp"].mean().rolling(3, center=True, min_periods=1).mean()
        axes[1].plot(s.index, s, color=col)
    axes[0].set_ylabel("Internet users, % (t-1)")
    axes[1].set_ylabel("Productivity growth, % (3-yr MA)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[0].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(FIG / "desc_diffusion.pdf")
    plt.close(fig)

    names = df.groupby(["grp", "country"], observed=True).size().reset_index()
    rows = []
    for lab in ["Low", "Middle", "High"]:
        cs = sorted(names.loc[names["grp"] == lab, "country"])
        rows.append(f"{lab} & " + "; ".join(c.replace("&", "\\&") for c in cs) + r"\\")
    (TAB / "countries.tex").write_text("\n".join(rows) + "\n")

    d = pd.read_csv(RES / "descriptives.csv", index_col=0)
    labels = {"dlp": "Labour-productivity growth (\\%)",
              "internet_l1": "Internet users, share ($t-1$)",
              "mobile_l1": "Mobile subscriptions per capita ($t-1$)",
              "broadband_l1": "Fixed broadband per capita ($t-1$)",
              "hc_l1": "Mean years of schooling ($t-1$)",
              "lp_init": "Initial log GDP per worker (pre-sample)",
              "csh_i_l1": "Investment share ($t-1$)",
              "csh_g_l1": "Government consumption share ($t-1$)",
              "open_l1": "Trade openness ($t-1$)",
              "dpop_l1": "Population growth, \\% ($t-1$)",
              "dep_l1": "Age-dependency ratio ($t-1$)",
              "urb_l1": "Urban population share ($t-1$)",
              "ysince_l1": "Years since internet take-off ($t-1$)",
              "dgdppc": "GDP per capita growth (\\%)"}
    lines = [f"{labels[i]} & {int(r['count'])} & {r['mean']:.3f} & {r['std']:.3f} & "
             f"{r['min']:.3f} & {r['max']:.3f}\\\\" for i, r in d.iterrows() if i in labels]
    (TAB / "descriptives.tex").write_text("\n".join(lines) + "\n")


def emp_outputs():
    summ = pd.read_csv(RES / "empirical_summary.csv")
    js = json.load(open(RES / "empirical.json"))
    emp_tables(summ)
    dyn_tables(summ, js["extra"])
    emp_figures(summ, js)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "sim"):
        agg = sim_tables()
        sim_figures(agg)
        plasmode_table()
    if what == "plasmode":
        plasmode_table()
    if what in ("all", "emp"):
        emp_outputs()
