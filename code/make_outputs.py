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
LABEL = {"Panel-DOSE (oracle nuisance)": "Latent-information oracle"}


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
                feas = agg[(agg["shape"] == sh) & (agg["method"] != "Panel-DOSE (oracle nuisance)")]
                best = feas[key].abs().min()
                c = fmt(v)
                if key == "irmse" and np.isclose(abs(v), best) and mth != "Panel-DOSE (oracle nuisance)":
                    c = r"\textbf{" + c + "}"
                cells.append(c)
        lines.append(f"{LABEL.get(mth, mth)} & " + " & ".join(cells) + r"\\")
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
            lines.append(f"\\quad {LABEL.get(mth, mth)} & " + " & ".join(cells) + r"\\")
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
    unp = base[base["method"] == "Panel-DOSE (unpenalised)"]
    for sh in SHAPES:
        d = dose[dose["shape"] == sh]
        lam[f"p_const_reject_{sh}"] = float(np.mean(d["p_const"] < 0.05))
        lam[f"p_lin_reject_{sh}"] = float(np.mean(d["p_lin"] < 0.05))
        u = unp[unp["shape"] == sh]
        lam[f"p_const_reject_u_{sh}"] = float(np.mean(u["p_const"] < 0.05))
        lam[f"p_lin_reject_u_{sh}"] = float(np.mean(u["p_lin"] < 0.05))
    json.dump(lam, open(RES / "simulation_lambda.json", "w"), indent=1)
    return agg


def sim_figures(agg):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8), sharey=True)
    short = {"TWFE": "TWFE", "TWFE-interaction": "TWFE-int", "TWFE-spline": "TWFE-spline",
             "FE-DML (constant)": "FE-DML", "FE-DML + R-learner (tuned)": "R-learner",
             "WG-DML sieve": "WG-sieve", "Pooled DML-sieve": "Pooled",
             "Ablation: Mundlak only": "Mundlak only", "Ablation: within only": "Within only",
             "Panel-DOSE (unpenalised)": "DOSE-u", "Panel-DOSE": "DOSE",
             "Panel-DOSE (oracle nuisance)": "Oracle"}
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


MODLAB = {"hc_l1": "Schooling", "lp_init": "Initial productivity",
          "ysince_l1": "Years since take-off"}
SHLAB = {"null": "no effect", "linear": "linear gradient", "threshold": "threshold"}


def plasmode_table():
    f = RES / "plasmode_raw.csv"
    if not f.exists():
        return {}
    p = pd.read_csv(f, keep_default_na=False, na_values=[""])
    main, app = [], []
    for zn, zl in MODLAB.items():
        main.append(r"\multicolumn{9}{l}{\textit{Moderator: " + zl + r"}}\\")
        app.append(r"\multicolumn{5}{l}{\textit{Moderator: " + zl + r"}}\\")
        for sh in ["null", "linear", "threshold"]:
            for mth in ["Panel-DOSE", "Panel-DOSE (unpenalised)"]:
                s = p[(p["moderator"] == zn) & (p["shape"] == sh) & (p["method"] == mth)]
                lab = SHLAB[sh] + (", unpenalised" if "unpen" in mth else "")
                cells = [fmt(s["diff_err"].mean()), fmt(np.sqrt(np.mean(s["diff_err"] ** 2))),
                         cov_cell(s["diff_cover"]), cov_cell(s["reject_diff"]),
                         cov_cell(s["reject_const"]), cov_cell(s["reject_lin"]),
                         cov_cell(s["cover"]), cov_cell(s["ucover"])]
                main.append(f"\\quad {lab} & " + " & ".join(cells) + r"\\")
            for mth in ["TWFE-interaction", "TWFE-spline"]:
                s = p[(p["moderator"] == zn) & (p["shape"] == sh) & (p["method"] == mth)]
                if len(s) == 0:
                    continue
                cells = [fmt(s["diff_err"].mean()), fmt(np.sqrt(np.mean(s["diff_err"] ** 2))),
                         cov_cell(s["diff_cover"]), cov_cell(s["reject_diff"])]
                app.append(f"\\quad {SHLAB[sh]}, {mth} & " + " & ".join(cells) + r"\\")
        main.append(r"\addlinespace")
        app.append(r"\addlinespace")
    (TAB / "plasmode.tex").write_text("\n".join(main[:-1]) + "\n")
    (TAB / "plasmode_twfe.tex").write_text("\n".join(app[:-1]) + "\n")
    return p


# ----------------------------------------------------------------------------- empirical
MAIN_ROWS = {
    "A": ["Baseline", "Common sample of the earlier version", "Adding mobile and broadband",
          "Random forest nuisance", "Region-by-year effects", "Pre-COVID sample, 1996-2019",
          "PWT 11.0, 1996-2023", "PWT 10.0, 1996-2019"],
    "B": ["Baseline", "Common sample of the earlier version", "Adding mobile and broadband",
          "Random forest nuisance", "Region-by-year effects", "Pre-COVID sample, 1996-2019",
          "WDI data, PWT 11.0 countries, 1996-2023", "PWT 11.0, 1996-2023", "PWT 10.0, 1996-2019"],
    "C": ["Baseline", "Common sample of the earlier version", "Random forest nuisance",
          "Region-by-year effects", "Predetermined moderator (zero before take-off)"],
}
PANEL_TITLES = {"A": "Panel A. Moderator: mean years of schooling",
                "B": "Panel B. Moderator: initial log GDP per worker",
                "C": "Panel C. Moderator: years since internet take-off (share $\\geq$ 10\\%)"}
SPLITS3 = "$^{\\dagger}$"


def emp_row(r):
    cells = [f"${r.fedml:.2f}{pstar(r.fedml, r.fedml_se)}$"]
    for lab in ["low", "mid", "high"]:
        cells.append(f"${r[f'gate_{lab}']:.2f}{pstar(r[f'gate_{lab}'], r[f'gate_{lab}_se'])}$")
    cells.append(f"${r['diff']:.2f}{pstar(r['diff'], r['diff_se'])}$")
    cells.append(f"${r.blp:.2f}{pstar(r.blp, r.blp_se)}$")
    cells.append("$<$0.01" if r.p_const < 0.005 else f"{r.p_const:.2f}")
    se = [f"({r.fedml_se:.2f})"] + [f"({r[f'gate_{lab}_se']:.2f})" for lab in ["low", "mid", "high"]] + \
         [f"({r.diff_se:.2f})", f"({r.blp_se:.2f})", ""]
    return cells, se


def spec_label(n):
    return n.replace("1996-", "1996--").replace("&", "\\&")


def emp_tables(summ, reps):
    def write(fname, rows_by_panel, panels=("A", "B", "C")):
        lines = []
        for P in panels:
            names = rows_by_panel.get(P, [])
            if not names:
                continue
            lines.append(r"\multicolumn{8}{l}{\textit{" + PANEL_TITLES[P] + r"}}\\")
            sub = summ[summ["panel"] == P].set_index("spec")
            for n in names:
                if n not in sub.index:
                    continue
                cells, se = emp_row(sub.loc[n])
                mark = SPLITS3 if reps.get((P, n), 5) == 3 else ""
                lines.append(f"\\quad {spec_label(n)}{mark} & " + " & ".join(cells) + r"\\")
                lines.append(" & " + " & ".join(se) + r"\\")
            lines.append(r"\addlinespace")
        (TAB / fname).write_text("\n".join(lines[:-1]) + "\n")

    write("emp_main.tex", MAIN_ROWS)
    rest = {P: [s for s in summ[summ["panel"] == P]["spec"] if s not in MAIN_ROWS.get(P, [])]
            for P in ["A", "B", "C"]}
    write("emp_appendix_a.tex", rest, ("A",))
    write("emp_appendix_bc.tex", rest, ("B", "C"))

    # diagnostics: penalty, identifying variation, support
    lines = []
    for P, n in [("A", "Baseline"), ("A", "Common sample of the earlier version"),
                 ("B", "Baseline"), ("B", "Common sample of the earlier version"),
                 ("C", "Baseline")]:
        r = summ[(summ["panel"] == P) & (summ["spec"] == n)].iloc[0]
        lab = "Baseline" if n == "Baseline" else "Common sample"
        lines.append(f"{P}: {lab} & {r['lambda']:.3g} & {r.edf:.1f} & {r.idvar_all:.2f} & "
                     f"{r.idvar_low:.2f} / {r.idvar_mid:.2f} / {r.idvar_high:.2f} & "
                     f"{r.d_median_low:.2f} / {r.d_median_mid:.2f} / {r.d_median_high:.2f} & "
                     f"{r.d_within_sd_low:.2f} / {r.d_within_sd_mid:.2f} / {r.d_within_sd_high:.2f} & "
                     f"{int(r.n_countries_low)} / {int(r.n_countries_mid)} / {int(r.n_countries_high)}\\\\")
    (TAB / "emp_diag.tex").write_text("\n".join(lines) + "\n")

    flow = pd.read_csv(ROOT / "data" / "processed" / "sample_flow.csv")
    lines = [f"{r.step} & {r.obs:,} & {r.countries}\\\\" for r in flow.itertuples()]
    (TAB / "sample_flow.tex").write_text("\n".join(lines) + "\n")


def dyn_tables(summ, ex):
    lp = summ[summ["panel"] == "LP"].copy()
    lp["h"] = lp["spec"].str[2:].astype(int)
    lpc = summ[summ["panel"] == "LPc"].copy()
    lpc["h"] = lpc["spec"].str[2:].astype(int)
    lpc = lpc.set_index("h")
    lp = lp.sort_values("h")
    lines = []
    for _, r in lp.iterrows():
        c = lpc.loc[r.h]
        lines.append(f"$h={r.h}$ ({r.h + 1}) & ${r.fedml:.2f}{pstar(r.fedml, r.fedml_se)}$ & ({r.fedml_se:.2f}) & "
                     f"{r.n_obs:,} & ${c.fedml:.2f}{pstar(c.fedml, c.fedml_se)}$ & ({c.fedml_se:.2f}) & "
                     f"${r['diff']:.2f}{pstar(r['diff'], r['diff_se'])}$ & ({r.diff_se:.2f})\\\\")
    (TAB / "emp_lp.tex").write_text("\n".join(lines) + "\n")
    fy, ld, iv, lt = ex["five_year"], ex["long_difference"], ex["iv"], ex["lead_test"]
    cz = summ[summ["panel"] == "CZ"].set_index("spec")
    lines = [
        f"Five-year averages, TWFE & ${fy['twfe']:.2f}{pstar(fy['twfe'], fy['twfe_se'])}$ & ({fy['twfe_se']:.2f}) & {fy['n_obs']:,} \\\\",
        f"Five-year averages, FE-DML & ${fy['fedml']:.2f}{pstar(fy['fedml'], fy['fedml_se'])}$ & ({fy['fedml_se']:.2f}) & {fy['n_obs']:,} \\\\",
        f"Long difference, OLS & ${ld['ols']:.2f}{pstar(ld['ols'], ld['ols_se'])}$ & ({ld['ols_se']:.2f}) & {ld['n']} \\\\",
        f"Long difference, DML & ${ld['dml']:.2f}{pstar(ld['dml'], ld['dml_se'])}$ & ({ld['dml_se']:.2f}) & {ld['n']} \\\\",
        f"Lead $D_{{t+1}}$, TWFE & ${lt['twfe_lead']:.2f}{pstar(lt['twfe_lead'], lt['twfe_lead_se'])}$ & ({lt['twfe_lead_se']:.2f}) & {lt['n_obs']:,} \\\\",
        f"Lead $D_{{t+1}}$, FE-DML & ${lt['fedml_lead']:.2f}{pstar(lt['fedml_lead'], lt['fedml_lead_se'])}$ & ({lt['fedml_lead_se']:.2f}) & {lt['n_obs']:,} \\\\",
        f"Fixed-line IV (2SLS), first-stage $F={iv['first_stage_F']:.1f}$ & ${iv['iv']:.2f}{pstar(iv['iv'], iv['iv_se'])}$ & ({iv['iv_se']:.2f}) & {iv['n_obs']:,} \\\\",
    ]
    for n, lab in [("High income, broadband, GDP per capita", "Broadband, GDP p.c. growth, high income"),
                   ("All countries, broadband, GDP per capita", "Broadband, GDP p.c. growth, all countries")]:
        r = cz.loc[n]
        lines.append(f"{lab} & ${r.fedml:.2f}{pstar(r.fedml, r.fedml_se)}$ & ({r.fedml_se:.2f}) & {r.n_obs:,} \\\\")
    (TAB / "emp_dyn.tex").write_text("\n".join(lines) + "\n")


def lam_cell(x):
    return "$10^{4}$" if x >= 1e4 - 1 else f"{x:.3g}"


def vintage_table():
    f = RES / "vintage.json"
    if not f.exists():
        return None
    v = json.load(open(f))
    name = {"10": "10.0", "11": "11.0"}
    lines = []
    for r in v["configs"]:
        lines.append(f"{name[r['y']]} & {name[r['z']]} & {name[r['x']]} & "
                     f"${r['diff']:.2f}{pstar(r['diff'], r['diff_se'])}$ & ({r['diff_se']:.2f}) & "
                     f"{lam_cell(r['lambda'])} & ${r['fixed_diff']:.2f}{pstar(r['fixed_diff'], r['fixed_diff_se'])}$ & "
                     f"({r['fixed_diff_se']:.2f})\\\\")
    (TAB / "emp_vintage.tex").write_text("\n".join(lines) + "\n")
    sf, sx = v["shapley_full"], v["shapley_fixed"]
    lines = [f"{lab} & & & ${a:.2f}$ & & & ${b:.2f}$ & \\\\" for lab, a, b in
             [("Outcome", sf["outcome"], sx["outcome"]),
              ("Moderator", sf["moderator"], sx["moderator"]),
              ("Controls", sf["controls"], sx["controls"]),
              ("Total", v["total_full"], v["total_fixed"])]]
    (TAB / "emp_shapley.tex").write_text("\n".join(lines) + "\n")
    return v
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
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, key, z, tw, xl, tt in [
            (axes[0], "A|Baseline", "hc_l1", "twfe_curve_hc", "Mean years of schooling (lagged)",
             "(a) by schooling"),
            (axes[1], "B|Baseline", "lp_init", "twfe_curve_lp", "Initial log GDP per worker",
             "(b) by initial productivity"),
            (axes[2], "C|Baseline", "ysince_l1", None, "Years since internet take-off",
             "(c) by years since take-off")]:
        band(ax, cur[key]["pen"], C_MAIN, "Panel-DOSE")
        c = cur[key]["pen"]
        lo_, hi_ = np.nanmin(c["ulo"]), np.nanmax(c["uhi"])
        pad = 0.05 * (hi_ - lo_)
        ax.set_ylim(lo_ - 2 * pad, hi_ + pad)
        if tw:
            ax.plot(ex[tw]["grid"], ex[tw]["est"], color=C_ALT, lw=1, ls="--",
                    label="TWFE-interaction")
        lo, hi = ax.get_ylim()
        ax.plot(df[z], np.full(len(df), lo + pad), "|", color=C_GREY, alpha=0.08, ms=6)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xl)
        ax.set_title(tt, fontsize=9)
    axes[0].set_ylabel(r"$\hat\theta$: pp growth per unit of adoption")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "emp_main.pdf")
    plt.close(fig)

    # dose response and local projections
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    c = {k: v[DOSE_SKIP:] for k, v in ex["dose"]["curve"].items()}
    band(axes[0], c, C_MAIN, "Panel-DOSE")
    axes[0].set_xlabel("Internet users (share, lagged)")
    axes[0].set_ylabel(r"$\hat f'(d)$")
    axes[0].set_title("(a) dose response", fontsize=9)
    for panel, col, lab, off in [("LP", C_MAIN, "All available observations", -0.08),
                                 ("LPc", C_ALT, "Common sample (all horizons observed)", 0.08)]:
        lp = summ[summ["panel"] == panel].copy()
        lp["h"] = lp["spec"].str[2:].astype(int)
        lp = lp.sort_values("h")
        axes[1].errorbar(lp["h"] + 1 + off, lp["fedml"] / 10, yerr=1.96 * lp["fedml_se"] / 10,
                         fmt="o-", color=col, ms=3, capsize=2, lw=1, label=lab)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Years from $t-1$ to $t+h$ ($h+1$)")
    axes[1].set_ylabel("pp change in log productivity\nper 10 pp internet")
    axes[1].set_title("(b) local projections", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "emp_dyn.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    base = cur["A|Baseline"]["pen"]
    axes[0].fill_between(base["grid"], base["lo"], base["hi"], color=C_MAIN, alpha=0.2, lw=0)
    axes[0].plot(base["grid"], base["est"], color=C_MAIN, lw=2, label="Baseline")
    # only WDI checks with the same moderator, treatment and outcome as the baseline
    same = [k for k in cur if k.startswith("A|") and k[2:] in ROBUST_SAME]
    hi_l = {"A|Common sample of the earlier version": ("#e08214", "Common sample"),
            "A|Random forest nuisance": ("#5e3c99", "Random-forest nuisance"),
            "A|Region-by-year effects": ("#1b7837", "Region-by-year effects")}
    for k in same:
        if k in hi_l:
            continue
        axes[0].plot(cur[k]["pen"]["grid"], cur[k]["pen"]["est"], lw=0.6, color=C_GREY, alpha=0.6)
    axes[0].plot([], [], lw=0.6, color=C_GREY, label="Other WDI checks")
    for k, (col, lab) in hi_l.items():
        axes[0].plot(cur[k]["pen"]["grid"], cur[k]["pen"]["est"], lw=1.2, color=col, label=lab)
    axes[0].set_xlabel("Mean years of schooling (lagged)")
    axes[0].set_ylabel(r"$\hat\theta(z)$")
    axes[0].set_title("(a) schooling: WDI robustness checks", fontsize=9)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend(frameon=False, fontsize=6, ncol=2, loc="lower left")
    pw = {"B|Baseline": df["lp_init"],
          "B|PWT 11.0, 1996-2023": pd.read_csv(ROOT / "data" / "processed" / "panel_pwt11.csv")["lp_init"],
          "B|PWT 10.0, 1996-2019": pd.read_csv(ROOT / "data" / "processed" / "panel_pwt.csv")["lp_init"]}
    for key, col, lab in [("B|Baseline", C_MAIN, "WDI (baseline)"),
                          ("B|PWT 11.0, 1996-2023", C_3, "PWT 11.0"),
                          ("B|PWT 10.0, 1996-2019", C_ALT, "PWT 10.0")]:
        c = cur[key]["pen"]
        zs = np.sort(pw[key].values)
        x = 100 * np.searchsorted(zs, np.array(c["grid"]), side="right") / len(zs)
        axes[1].fill_between(x, c["lo"], c["hi"], color=col, alpha=0.12, lw=0)
        axes[1].plot(x, c["est"], color=col, lw=1.5, label=lab)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Percentile of initial productivity (within sample)")
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


FIVE = {"Baseline", "Common sample of the earlier version", "Original specification",
        "Adding the investment share", "Adding mobile and broadband",
        "Treatment: composite ICT index"}
ROBUST_SAME = ["Common sample of the earlier version", "Adding the investment share",
               "Random forest nuisance", "Region-by-year effects", "Unpenalised sieve",
               "Three interior knots", "GCV penalty", "Pre-COVID sample, 1996-2019",
               "Excluding 2020-2021", "Excluding 2025", "Untrimmed outcome", "Winsorised outcome",
               "Without data screening", "Excluding resource exporters",
               "High statistical capacity (SPI)", "No Mundlak means",
               "WDI data, PWT 11.0 countries, 1996-2023"]
DOSE_SKIP = 3   # the dose-response figure starts at the fourth grid point (see caption)


def emp_outputs():
    summ = pd.read_csv(RES / "empirical_summary.csv")
    js = json.load(open(RES / "empirical.json"))
    reps = {(r.panel, r.spec): (5 if (r.spec in FIVE and not (r.panel == "C" and r.spec not in
                                                              ("Baseline", "Common sample of the earlier version")))
                                else 3) for r in summ.itertuples()}
    emp_tables(summ, reps)
    dyn_tables(summ, js["extra"])
    emp_figures(summ, js)
    return summ, js


# ----------------------------------------------------------------------------- numbers in text
def numbers():
    """Write manuscript/tables/numbers.tex: every number quoted in the text, as \nm{key}."""
    N = {}

    def put(k, v, d=2):
        if isinstance(v, str):
            N[k] = v
        elif isinstance(v, (int, np.integer)):
            N[k] = f"{int(v):,}".replace(",", "{,}")
        else:
            pval = k.startswith(("pconst", "plin", "holm", "pdose")) or (k.endswith("p") and d == 2)
            N[k] = f"{v:.3f}" if pval and abs(v) < 0.01 else f"{v:.{d}f}"

    def lamf(x):
        if x >= 1e4 - 1:
            return "10^{4}"
        return f"{x:.3g}"

    summ = pd.read_csv(RES / "empirical_summary.csv")
    js = json.load(open(RES / "empirical.json"))
    ex = js["extra"]
    S = lambda P, n: summ[(summ["panel"] == P) & (summ["spec"] == n)].iloc[0]  # noqa: E731
    smp = ex["sample"]
    put("nobs", smp["n_obs"]); put("ncty", smp["n_countries"])
    put("nobscommon", smp["common_obs"]); put("nctycommon", smp["common_countries"])
    put("nobsunt", smp["untrimmed_obs"]); put("nctyunt", smp["untrimmed_countries"])
    put("nobsuns", smp["unscreened_obs"])
    put("npwtten", smp["pwt10"][0]); put("nctypwtten", smp["pwt10"][1])
    put("npwtel", smp["pwt11"][0]); put("nctypwtel", smp["pwt11"][1])
    put("nnopre", len(smp["no_pre_sample"])); put("nopre", ", ".join(smp["no_pre_sample"]))
    cov = ex["coverage_by_income"]
    for key, lab in [("High income", "hi"), ("Upper middle income", "um"),
                     ("Lower middle income", "lm"), ("Low income", "li")]:
        put(f"cov{lab}", int(cov["sum"][key])); put(f"tot{lab}", int(cov["size"][key]))
    put("nflags", ex["screening"]["n_flags"])
    for k, v in ex["screening"]["by_series"].items():
        put(f"flags{k}", int(v))
    b = S("A", "Baseline")
    put("fedml", b.fedml); put("fedmlse", b.fedml_se)
    put("fedmlten", b.fedml / 10, 3)
    put("cilo", (b.fedml - 1.96 * b.fedml_se) / 10, 3); put("cihi", (b.fedml + 1.96 * b.fedml_se) / 10, 3)
    tw = ex["twoway"]["A"]
    put("twse", tw["fedml_se_twoway"]); put("cihitw", (b.fedml + 1.96 * tw["fedml_se_twoway"]) / 10, 3)
    put("cilotw", (b.fedml - 1.96 * tw["fedml_se_twoway"]) / 10, 3)
    put("twfe", ex["twfe"]["b"]); put("twfese", ex["twfe"]["se"])
    put("savg", b.avg_theta); put("savgse", b.avg_theta_se)
    df = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    rise = (df.groupby("iso")["internet_l1"].max() - df.groupby("iso")["internet_l1"].min()).mean()
    put("rise", 100 * rise, 0); put("riseimpl", b.fedml * rise, 1)
    put("riseimplhi", (b.fedml + 1.96 * b.fedml_se) * rise, 2)
    put("idvar", b.idvar_all); put("idvartot", 100 * b.idvar_all_total, 1)
    put("idvarlowA", b.idvar_low); put("idvarlowB", S("B", "Baseline").idvar_low)
    put("dmedlowA", b.d_median_low); put("dmedlowB", S("B", "Baseline").d_median_low)
    for key, (P, n) in {"common": ("A", "Common sample of the earlier version"),
                        "rf": ("A", "Random forest nuisance"),
                        "ry": ("A", "Region-by-year effects"),
                        "precovid": ("A", "Pre-COVID sample, 1996-2019"),
                        "nocovid": ("A", "Excluding 2020-2021"),
                        "gdppc": ("A", "Outcome: GDP per capita growth"),
                        "spi": ("A", "High statistical capacity (SPI)"),
                        "pwtel": ("A", "PWT 11.0, 1996-2023"),
                        "pwtten": ("A", "PWT 10.0, 1996-2019"),
                        "mobbb": ("A", "Adding mobile and broadband"),
                        "mobbbnf": ("A", "Adding mobile and broadband, no zero-filling"),
                        "orig": ("A", "Original specification"),
                        "ict": ("A", "Treatment: composite ICT index"),
                        "nores": ("A", "Excluding resource exporters"),
                        "popw": ("A", "Population-weighted averages"),
                        "unt": ("A", "Untrimmed outcome"),
                        "uns": ("A", "Without data screening"),
                        "nomund": ("A", "No Mundlak means"),
                        "invest": ("A", "Adding the investment share"),
                        "lagtwo": ("A", "Two-year lag of internet")}.items():
        r = S(P, n)
        put(f"{key}", r.fedml); put(f"{key}se", r.fedml_se); put(f"{key}n", int(r.n_obs))
        put(f"{key}diffA", r["diff"]); put(f"{key}diffAse", r.diff_se)
    put("popwavg", S("A", "Population-weighted averages").avg_theta)
    put("popwavgse", S("A", "Population-weighted averages").avg_theta_se)
    # heterogeneity
    for P in ["A", "B", "C"]:
        r = S(P, "Baseline")
        for k in ["diff", "diff_se", "blp", "blp_se", "gate_low", "gate_low_se", "gate_high",
                  "gate_high_se", "gate_mid", "gate_mid_se", "diff_x5", "diff_x5_se"]:
            put(f"{k.replace('_', '')}{P}", r[k])
        put(f"pconst{P}", r.p_const); put(f"plin{P}", r.p_lin); put(f"pconstu{P}", r.p_const_u)
        put(f"lam{P}", lamf(r['lambda'])); put(f"edf{P}", r.edf, 1)
        m = ex["mde"][P]
        put(f"mde{P}", m["diff_mde"] / 10); put(f"cilo{P}", m["diff_ci"][0] / 10)
        put(f"cihi{P}", m["diff_ci"][1] / 10)
        put(f"diffsetw{P}", ex["twoway"][P]["diff_se_twoway"])
        c = S(P, "Common sample of the earlier version")
        put(f"commondiff{P}", c["diff"]); put(f"commondiffse{P}", c.diff_se); put(f"commonp{P}", c.p_const)
        c = S(P, "Region-by-year effects")
        put(f"rydiff{P}", c["diff"]); put(f"rydiffse{P}", c.diff_se); put(f"ryp{P}", c.p_const)
        c = S(P, "Random forest nuisance")
        put(f"rfdiff{P}", c["diff"]); put(f"rfdiffse{P}", c.diff_se)
        c = S(P, "Adding mobile and broadband")
        put(f"mobbbdiff{P}", c["diff"]); put(f"mobbbdiffse{P}", c.diff_se)
        c = S(P, "Pre-COVID sample, 1996-2019")
        put(f"precoviddiff{P}", c["diff"]); put(f"precoviddiffse{P}", c.diff_se)
    for key, (P, n) in {"lays": ("A", "Moderator: learning-adjusted schooling"),
                        "ictB": ("B", "Treatment: composite ICT index"),
                        "distus": ("B", "Moderator: distance to US frontier"),
                        "presmp": ("B", "Countries with pre-sample productivity only"),
                        "ypre": ("C", "Predetermined moderator (zero before take-off)"),
                        "pwtelB": ("B", "PWT 11.0, 1996-2023"),
                        "pwtelnineB": ("B", "PWT 11.0, 1996-2019"),
                        "pwttenB": ("B", "PWT 10.0, 1996-2019"),
                        "wdipwtB": ("B", "WDI data, PWT 11.0 countries, 1996-2023"),
                        "pwtelnaB": ("B", "PWT 11.0, national-prices moderator"),
                        "pwttennaB": ("B", "PWT 10.0, national-prices moderator"),
                        "pwtelA": ("A", "PWT 11.0, 1996-2023"),
                        "pwttenA": ("A", "PWT 10.0, 1996-2019")}.items():
        r = S(P, n)
        put(f"{key}diff", r["diff"]); put(f"{key}diffse", r.diff_se); put(f"{key}p", r.p_const)
        put(f"{key}low", r.gate_low); put(f"{key}lowse", r.gate_low_se)
    for key, k in [("twgl", "twfe_grad_lp"), ("twgs", "twfe_grad_lp_spline"), ("twgh", "twfe_grad_hc")]:
        put(key, ex[k]["diff"]); put(key + "se", ex[k]["diff_se"])
    h = ex["holm"]
    for i, P in enumerate("ABC"):
        put(f"holm{P}", h["adjusted"][i]); put(f"holmu{P}", h["adjusted_unpen"][i])
        put(f"holmlin{P}", h["adjusted_lin"][i])
    # dose response
    put("pdose", ex["dose"]["p_linear_f"]); put("pdoseu", ex["dose"]["p_linear_f_u"])
    q0 = 5 + DOSE_SKIP * 90 / 32
    put("dosestart", q0, 0)
    # dynamics
    lp = summ[summ["panel"] == "LP"].copy()
    lp["h"] = lp["spec"].str[2:].astype(int)
    lp = lp.set_index("h")
    lpc = summ[summ["panel"] == "LPc"].copy()
    lpc["h"] = lpc["spec"].str[2:].astype(int)
    lpc = lpc.set_index("h")
    for hh in range(9):
        put(f"lp{'abcdefghi'[hh]}", lp.loc[hh, "fedml"] / 10); put(f"lpse{'abcdefghi'[hh]}", lp.loc[hh, "fedml_se"] / 10)
        put(f"lpc{'abcdefghi'[hh]}", lpc.loc[hh, "fedml"] / 10); put(f"lpcse{'abcdefghi'[hh]}", lpc.loc[hh, "fedml_se"] / 10)
        put(f"lpn{'abcdefghi'[hh]}", int(lp.loc[hh, "n_obs"]))
    put("lpcn", int(lpc.loc[0, "n_obs"]))
    put("lpfedmlsplit", lp.loc[0, "fedml"])
    fy, ld, iv, lt = ex["five_year"], ex["long_difference"], ex["iv"], ex["lead_test"]
    put("fytwfe", fy["twfe"]); put("fytwfese", fy["twfe_se"]); put("fyfedml", fy["fedml"])
    put("fyfedmlse", fy["fedml_se"]); put("fyn", fy["n_obs"])
    put("ldols", ld["ols"]); put("ldolsse", ld["ols_se"]); put("lddml", ld["dml"]); put("lddmlse", ld["dml_se"])
    put("ldn", ld["n"]); put("ldolsten", ld["ols"] / 10); put("lddmlten", ld["dml"] / 10)
    put("ivf", iv["first_stage_F"], 1); put("iv", iv["iv"]); put("ivse", iv["iv_se"])
    put("leadtw", lt["twfe_lead"]); put("leadtwse", lt["twfe_lead_se"])
    put("leaddml", lt["fedml_lead"]); put("leaddmlse", lt["fedml_lead_se"])
    put("leadd", lt["twfe_d"]); put("leaddse", lt["twfe_d_se"])
    cz = summ[summ["panel"] == "CZ"].set_index("spec").loc["High income, broadband, GDP per capita"]
    put("cz", cz.fedml); put("czse", cz.fedml_se)
    put("czlo", (cz.fedml - 1.96 * cz.fedml_se) / 10); put("czhi", (cz.fedml + 1.96 * cz.fedml_se) / 10)
    # vintages
    f = RES / "vintage.json"
    if f.exists():
        v = json.load(open(f))
        put("vn", v["n_common"]); put("vcty", v["n_countries"])
        cf = {(r["y"], r["z"], r["x"]): r for r in v["configs"]}
        a0, a1 = cf[("10", "10", "10")], cf[("11", "11", "11")]
        put("vten", a0["diff"]); put("vtense", a0["diff_se"]); put("vel", a1["diff"]); put("velse", a1["diff_se"])
        put("vtenlow", a0["gate_low"]); put("vtenlowse", a0["gate_low_se"])
        put("vy", cf[("11", "10", "10")]["diff"]); put("vz", cf[("10", "11", "10")]["diff"])
        put("vx", cf[("10", "10", "11")]["diff"])
        put("vtenfx", a0["fixed_diff"]); put("velfx", a1["fixed_diff"])
        put("vlamten", lamf(a0["lambda"])); put("vlamel", lamf(a1["lambda"]))
        put("vlamy", lamf(cf[("11", "10", "10")]["lambda"])); put("vlamfx", lamf(v["fixed_lambda"]))
        d = v["difference"]
        put("vdiff", d["est"]); put("vdiffbse", d["boot_se"]); put("vdiffbmean", d["boot_mean"])
        put("vdifflo", d["boot_ci"][0]); put("vdiffhi", d["boot_ci"][1]); put("vnboot", d["n_boot"])
        put("vdiffp", d["boot_p_le0"], 3)
        put("vtenbse", d["diff10_boot_se"]); put("velbse", d["diff11_boot_se"])
        put("vtenbmean", d["diff10_boot_mean"]); put("velbmean", d["diff11_boot_mean"])
        put("vshrink", 100 * (1 - a1["diff"] / a0["diff"]), 0)
        for k in ["outcome", "moderator", "controls"]:
            put(f"shf{k}", v["shapley_full"][k]); put(f"shx{k}", v["shapley_fixed"][k])
            put(f"shfshare{k}", 100 * v["shapley_full"][k] / v["total_full"], 0)
            put(f"shxshare{k}", 100 * v["shapley_fixed"][k] / v["total_fixed"], 0)
        put("shftotal", v["total_full"]); put("shxtotal", v["total_fixed"])
        put("vswitch", 100 * v["tercile_switch_share"], 1); put("vswitchn", v["countries_switching"])
        put("vrankcorr", v["rank_corr"], 3); put("vgcorr", v["growth_corr_annual"])
        put("vgcorrm", v["growth_corr_country_means"])
        lo = v["loco"]
        cty = pd.read_csv(ROOT / "data" / "processed" / "panel_pwt11.csv").drop_duplicates("iso").set_index("iso")["country"]
        for k, lab in [("ch_dd", "dd"), ("ch_low10", "low"), ("ch_diff10", "dten"), ("ch_diff11", "del")]:
            put(f"loco{lab}", lo[k]["change"]); put(f"loco{lab}cty", cty.get(lo[k]["iso"], lo[k]["iso"]))
        put("locoddmin", lo["dd_range"][0]); put("locoddmax", lo["dd_range"][1])
    f = RES / "invariance.json"
    if f.exists():
        iv_ = json.load(open(f))
        put("invcA", iv_["A"]["contrast"]); put("invtA", iv_["A"]["treatment_contrast"])
        put("invseA", iv_["A"]["contrast_se"]); put("invsd", iv_["A"]["sd_treatment_residual"], 3)
        put("invshiftse", abs(iv_["A"]["treatment_contrast"] - iv_["A"]["contrast"]) / iv_["A"]["contrast_se"], 2)
    # plasmode
    f = RES / "plasmode_raw.csv"
    if f.exists() and "moderator" in pd.read_csv(f, nrows=1).columns:
        p = pd.read_csv(f, keep_default_na=False, na_values=[""])
        for zn, P in [("hc_l1", "A"), ("lp_init", "B"), ("ysince_l1", "C")]:
            for sh, sl in [("null", "n"), ("linear", "l"), ("threshold", "t")]:
                for mth, ml in [("Panel-DOSE", ""), ("Panel-DOSE (unpenalised)", "u")]:
                    q = p[(p["moderator"] == zn) & (p["shape"] == sh) & (p["method"] == mth)]
                    tag = f"{sl}{ml}{P}"
                    put(f"prc{tag}", q["reject_const"].mean()); put(f"prl{tag}", q["reject_lin"].mean())
                    put(f"pcov{tag}", q["diff_cover"].mean()); put(f"pbias{tag}", q["diff_err"].mean())
                    put(f"ppw{tag}", q["cover"].mean()); put(f"psim{tag}", q["ucover"].mean())
                    put(f"prd{tag}", q["reject_diff"].mean())
                    put(f"preps{tag}", int(len(q)))
            put(f"ptarget{P}", p.loc[p["moderator"] == zn, "target"].iloc[0])
            for mth, ml in [("TWFE-interaction", "twi"), ("TWFE-spline", "tws")]:
                q = p[(p["moderator"] == zn) & (p["shape"] == "null") & (p["method"] == mth)]
                if len(q):
                    put(f"p{ml}rej{P}", q["reject_diff"].mean()); put(f"p{ml}bias{P}", q["diff_err"].mean())
            n0 = p[(p["moderator"] == zn) & (p["shape"] == "null") & (p["method"] == "Panel-DOSE")]
            put(f"pmcse{P}", np.sqrt(0.05 * 0.95 / len(n0)) * 100, 1)
    # simulation
    f = RES / "simulation_raw.csv"
    if f.exists() and "p_lin" in pd.read_csv(f, nrows=1).columns:
        raw = pd.read_csv(f)
        base = raw[(raw["N"] == 100) & (raw["rho"] == 1.0)]
        g = base.groupby(["shape", "method"])["irmse"].mean()
        for sh, t in [("constant", "c"), ("linear", "l"), ("threshold", "t"), ("hump", "h")]:
            put(f"sdose{t}", g[(sh, "Panel-DOSE")]); put(f"sdoseu{t}", g[(sh, "Panel-DOSE (unpenalised)")])
            put(f"sspl{t}", g[(sh, "TWFE-spline")]); put(f"srl{t}", g[(sh, "FE-DML + R-learner (tuned)")])
            put(f"sorc{t}", g[(sh, "Panel-DOSE (oracle nuisance)")])
            for mth, ml in [("Panel-DOSE", ""), ("Panel-DOSE (unpenalised)", "u")]:
                q = base[(base["shape"] == sh) & (base["method"] == mth)]
                put(f"scov{ml}{t}", q["cover"].mean()); put(f"sucov{ml}{t}", q["ucover"].mean())
                put(f"srejc{ml}{t}", np.mean(q["p_const"] < 0.05)); put(f"srejl{ml}{t}", np.mean(q["p_lin"] < 0.05))
        het = ["linear", "threshold", "hump"]
        dose_i = np.array([g[(sh, "Panel-DOSE")] for sh in het])
        rng_ = lambda arr: (100 * arr.min(), 100 * arr.max())  # noqa: E731
        for key, mth in [("spl", "TWFE-spline"), ("rl", "FE-DML + R-learner (tuned)")]:
            other = np.array([g[(sh, mth)] for sh in het])
            lo_, hi_ = rng_(1 - dose_i / other)
            put(f"sbelow{key}lo", lo_, 0); put(f"sbelow{key}hi", hi_, 0)
        mo = np.array([g[(sh, "Ablation: Mundlak only")] for sh in het])
        wo = np.array([g[(sh, "Ablation: within only")] for sh in het])
        wg = np.array([g[(sh, "WG-DML sieve")] for sh in het])
        lo_, hi_ = rng_(mo / dose_i - 1)
        put("snowithinlo", lo_, 0); put("snowithinhi", hi_, 0)
        put("snowithinirlo", mo.min()); put("snowithinirhi", mo.max())
        put("snomundlo", (wo / dose_i).min(), 1); put("snomundhi", (wo / dose_i).max(), 1)
        put("snomundirlo", wo.min()); put("snomundirhi", wo.max())
        put("swglo", wg.min()); put("swghi", wg.max())
        orc = np.array([g[(sh, "Panel-DOSE (oracle nuisance)")] for sh in het])
        lo_, hi_ = rng_(1 - orc / dose_i)
        put("sorclo", lo_, 0); put("sorchi", hi_, 0)
        ib = base.groupby(["shape", "method"])["ibias"].mean()
        put("ssplbias", ib[("linear", "TWFE-spline")])
        cov = base.groupby(["shape", "method"])
        for col, key in [("ate_cover", "ate"), ("contrast_cover", "con"), ("blp_cover", "blp")]:
            vals = np.array([cov[col].mean()[(sh, "Panel-DOSE")] for sh in SHAPES])
            put(f"scov{key}lo", vals.min()); put(f"scov{key}hi", vals.max())
        vals = np.array([cov["ate_cover"].mean()[(sh, "Panel-DOSE (oracle nuisance)")] for sh in SHAPES])
        put("scovorclo", vals.min()); put("scovorchi", vals.max())
        pw = np.array([cov["cover"].mean()[(sh, "Panel-DOSE")] for sh in SHAPES])
        uw = np.array([cov["ucover"].mean()[(sh, "Panel-DOSE")] for sh in SHAPES])
        pwu = np.array([cov["cover"].mean()[(sh, "Panel-DOSE (unpenalised)")] for sh in SHAPES])
        uwu = np.array([cov["ucover"].mean()[(sh, "Panel-DOSE (unpenalised)")] for sh in SHAPES])
        for key, arr in [("pw", pw), ("uw", uw), ("pwu", pwu), ("uwu", uwu)]:
            put(f"s{key}lo", arr.min()); put(f"s{key}hi", arr.max())
        tic = np.array([cov["contrast_cover"].mean()[(sh, "TWFE-interaction")] for sh in SHAPES])
        put("stwicovlo", tic.min()); put("stwicovhi", tic.max())
        sens_all = raw[raw["shape"] == "threshold"]
        for (n_, r_), key in [((100, 0.25), "rq"), ((100, 0.5), "rh")]:
            q = sens_all[(sens_all["N"] == n_) & (sens_all["rho"] == r_)]
            put(f"ssens{key}spl", q[q["method"] == "TWFE-spline"]["irmse"].mean())
            put(f"ssens{key}dose", q[q["method"] == "Panel-DOSE"]["irmse"].mean())
        q = sens_all[(sens_all["N"] == 100) & (sens_all["rho"] == 0.25) & (sens_all["method"] == "TWFE-spline")]
        put("ssensrqsplcov", q["cover"].mean())
        sens = raw[(raw["shape"] == "threshold") & (raw["method"] == "Panel-DOSE")]
        put("sucovrhotwo", sens[(sens["N"] == 100) & (sens["rho"] == 2.0)]["ucover"].mean())
        dose = base[base["method"] == "Panel-DOSE"]
        put("slammax", 100 * np.mean(dose["lambda"] >= 1e4 - 1), 0)
        put("slamzero", 100 * np.mean(dose["lambda"] == 0), 0)
    lines = [r"\makeatletter", r"\newcommand{\nm}[1]{\ifcsname nm:#1\endcsname\csname nm:#1\endcsname"
             r"\else\PackageError{numbers}{Unknown number #1}{}\fi}"]
    for k, v in N.items():
        lines.append(r"\expandafter\def\csname nm:" + k + r"\endcsname{" + v + "}")
    lines.append(r"\makeatother")
    (TAB / "numbers.tex").write_text("\n".join(lines) + "\n")
    return N


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "sim"):
        agg = sim_tables()
        sim_figures(agg)
    if what in ("all", "sim", "plasmode"):
        plasmode_table()
    if what in ("all", "emp"):
        emp_outputs()
        vintage_table()
    numbers()
