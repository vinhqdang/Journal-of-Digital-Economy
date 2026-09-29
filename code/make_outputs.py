"""Turn simulation and empirical results into the figures and LaTeX tables of the manuscript."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulation as sim  # noqa: E402
from panel_dose import PanelDOSE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "manuscript" / "figures"
TAB = ROOT / "manuscript" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
C_MAIN, C_ALT, C_GREY = "#1f4e79", "#c0504d", "#7f7f7f"

ORDER = ["TWFE", "TWFE-interaction", "FE-DML (constant)", "FE-DML + R-learner",
         "Pooled DML-sieve", "Panel-DOSE (unpenalised)", "Panel-DOSE"]


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


# ----------------------------------------------------------------------------- simulation
def sim_tables():
    raw = pd.read_csv(RES / "simulation_raw.csv")
    base = raw[(raw["N"] == 100) & (raw["rho"] == 1.0)]
    agg = base.groupby(["shape", "method"]).agg(
        irmse=("irmse", "mean"), ibias=("ibias", "mean"),
        ate_bias=("ate_err", "mean"), ate_rmse=("ate_err", lambda e: np.sqrt(np.mean(e ** 2))),
        cover=("cover", "mean"), ucover=("ucover", "mean"), reps=("rep", "nunique")).reset_index()
    agg.to_csv(RES / "simulation_summary.csv", index=False)
    shapes = ["constant", "linear", "threshold", "hump"]
    lines = []
    for sh in shapes:
        lines.append(r"\multicolumn{7}{l}{\textit{Design: " + sh + r" $\theta(z)$}}\\")
        sub = agg[agg["shape"] == sh].set_index("method")
        best = sub["irmse"].min()
        for mth in ORDER:
            r = sub.loc[mth]
            ir = fmt(r.irmse)
            if np.isclose(r.irmse, best):
                ir = r"\textbf{" + ir + "}"
            lines.append(f"\\quad {mth} & {ir} & {fmt(r.ibias)} & {fmt(r.ate_bias)} & "
                         f"{fmt(r.ate_rmse)} & {fmt(r.cover, 2)} & {fmt(r.ucover, 2)}\\\\")
        lines.append(r"\addlinespace")
    (TAB / "sim_main.tex").write_text("\n".join(lines[:-1]) + "\n")

    sens = raw[(raw["shape"] == "threshold")]
    rows = []
    for (n, rho), g in sens.groupby(["N", "rho"]):
        if g["rep"].nunique() < 10:
            continue
        a = g.groupby("method").agg(irmse=("irmse", "mean"), cover=("cover", "mean"),
                                    ate_rmse=("ate_err", lambda e: np.sqrt(np.mean(e ** 2))))
        for mth in ["TWFE-interaction", "FE-DML + R-learner", "Pooled DML-sieve", "Panel-DOSE"]:
            rows.append({"N": n, "rho": rho, "method": mth, **a.loc[mth].to_dict()})
    sdf = pd.DataFrame(rows)
    sdf.to_csv(RES / "simulation_sensitivity.csv", index=False)
    piv = sdf.pivot_table(index="method", columns=["N", "rho"], values="irmse").loc[
        ["TWFE-interaction", "FE-DML + R-learner", "Pooled DML-sieve", "Panel-DOSE"]]
    cols = [(50, 1.0), (100, 0.5), (100, 1.0), (100, 2.0), (200, 1.0)]
    cols = [c for c in cols if c in piv.columns]
    lines = [f"{m} & " + " & ".join(fmt(piv.loc[m, c]) for c in cols) + r"\\" for m in piv.index]
    (TAB / "sim_sens.tex").write_text("\n".join(lines) + "\n")
    return agg, sdf


def sim_figures(agg):
    shapes = ["constant", "linear", "threshold", "hump"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=False)
    short = {"TWFE": "TWFE", "TWFE-interaction": "TWFE-int", "FE-DML (constant)": "FE-DML",
             "FE-DML + R-learner": "R-learner", "Pooled DML-sieve": "Pooled",
             "Panel-DOSE (unpenalised)": "DOSE-u", "Panel-DOSE": "DOSE"}
    for ax, sh in zip(axes, shapes):
        sub = agg[agg["shape"] == sh].set_index("method").loc[ORDER]
        cols = [C_MAIN if "DOSE" in m else C_GREY for m in ORDER]
        ax.barh([short[m] for m in ORDER], sub["irmse"], color=cols)
        ax.set_title(sh, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.2)
        if sh != "constant":
            ax.set_yticklabels([])
        ax.set_xlabel("IRMSE")
    fig.tight_layout()
    fig.savefig(FIG / "sim_irmse.pdf")
    plt.close(fig)

    # one illustrative replication per design
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=True)
    for ax, sh in zip(axes, shapes):
        df = sim.simulate(sh, seed=424242)
        m = PanelDOSE(sim.lgbm, n_rep=2, z_in_controls=True).fit(df, "y", "d", sim.CONTROLS,
                                                                 "unit", "t", z="z")
        e = m.effect(sim.GRID)
        tw, _, _ = sim.twfe(df, True)
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


# ----------------------------------------------------------------------------- empirical
def emp_outputs():
    js = json.load(open(RES / "empirical.json"))
    summ = pd.read_csv(RES / "empirical_summary.csv")
    cur = js["curves"]

    def band(ax, c, color, label, uniform=True):
        g = np.array(c["grid"])
        ax.fill_between(g, c["lo"], c["hi"], color=color, alpha=0.25, lw=0)
        if uniform:
            ax.fill_between(g, c["ulo"], c["uhi"], color=color, alpha=0.12, lw=0)
        ax.plot(g, c["est"], color=color, lw=1.6, label=label)
        ax.axhline(0, color="k", lw=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    band(axes[0], cur["dose_hc"], C_MAIN, "Panel-DOSE")
    tw = cur["twfe_hc"]
    g = np.array(tw["grid"])
    axes[0].plot(g, tw["est"], color=C_ALT, lw=1, ls="--", label="TWFE-interaction")
    axes[0].set_xlabel("Mean years of schooling (lagged)")
    axes[0].set_ylabel(r"$\hat\theta$: pp growth per unit internet")
    axes[0].legend(frameon=False, fontsize=7)
    band(axes[1], cur["dose_lp"], C_MAIN, "Panel-DOSE")
    axes[1].set_xlabel("Log GDP per worker (lagged)")
    # the lowest adoption quantiles contain very few observations; start at the 20th percentile
    band(axes[2], {k: v[3:] for k, v in cur["dose_net"].items()}, C_MAIN, "Panel-DOSE")
    axes[2].set_xlabel("Internet users (share, lagged)")
    axes[2].set_ylabel(r"$\hat f'(d)$")
    for ax, t in zip(axes, ["(a) by schooling", "(b) by development level",
                            "(c) dose response"]):
        ax.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "emp_main.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    base = cur["dose_hc"]
    ax.fill_between(base["grid"], base["lo"], base["hi"], color=C_MAIN, alpha=0.2, lw=0)
    ax.plot(base["grid"], base["est"], color=C_MAIN, lw=2, label="Baseline")
    for k in cur:
        if k.startswith("rob_") and k not in ("rob_Broadband as treatment",):
            ax.plot(cur[k]["grid"], cur[k]["est"], lw=0.8, alpha=0.8, label=k[4:])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Mean years of schooling (lagged)")
    ax.set_ylabel(r"$\hat\theta(z)$")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "emp_robust.pdf")
    plt.close(fig)

    def star(p):
        return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""

    from scipy import stats
    rename = {"Baseline (LightGBM, Z = human capital)": "Baseline",
              "Z = initial labour productivity": "Baseline"}
    panel_a = [s for s in summ["spec"] if not s.startswith("Z =") and not s.startswith("Dose")]
    panel_b = [s for s in summ["spec"] if s.startswith("Z =")]
    lines = []

    def row(r):
        p_avg = 2 * stats.norm.sf(abs(r.avg / r.avg_se))
        cells = [f"${r.avg:.2f}{star(p_avg)}$", f"({r.avg_se:.2f})"]
        for lab in ["low", "mid", "high"]:
            p = 2 * stats.norm.sf(abs(r[f"gate_{lab}"] / r[f"gate_{lab}_se"]))
            cells.append(f"${r[f'gate_{lab}']:.2f}{star(p)}$")
        cells.append(f"${r.gate_diff:.2f}{star(r.gate_diff_p)}$")
        cells.append(f"${r.blp_slope:.2f}{star(r.blp_p)}$")
        name = rename.get(r["spec"], r["spec"].replace("Z = productivity: ", ""))
        name = name[0].upper() + name[1:]
        name = name.replace("Outcome TFP", "Outcome: TFP")
        lines.append("\\quad " + name + " & " + " & ".join(cells) + r"\\")
        se = ["", ""] + [f"({r[f'gate_{lab}_se']:.2f})" for lab in ["low", "mid", "high"]] + \
             [f"({r.gate_diff_se:.2f})", f"({r.blp_slope_se:.2f})"]
        lines.append(" & " + " & ".join(se) + r"\\")

    lines.append(r"\multicolumn{8}{l}{\textit{Panel A. Moderator: mean years of schooling}}\\")
    for s in panel_a:
        row(summ.set_index("spec").loc[s].rename(None).to_frame().T.assign(spec=s).iloc[0])
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{8}{l}{\textit{Panel B. Moderator: lagged log GDP per worker}}\\")
    for s in panel_b:
        row(summ.set_index("spec").loc[s].rename(None).to_frame().T.assign(spec=s).iloc[0])
    (TAB / "emp_main.tex").write_text("\n".join(lines) + "\n")

    # descriptive figure: diffusion of internet use by productivity tercile
    panel = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    first = panel.sort_values("year").groupby("iso")["lp_l1"].transform("first")
    panel["grp"] = pd.qcut(first, 3, labels=["Low", "Middle", "High"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for lab, col in zip(["Low", "Middle", "High"], ["#c0504d", "#7f7f7f", C_MAIN]):
        g = panel[panel["grp"] == lab].groupby("year")
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

    # appendix: country list
    names = panel.groupby(["grp", "country"], observed=True).size().reset_index()
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
              "lp_l1": "Log GDP per worker ($t-1$)",
              "csh_i_l1": "Investment share ($t-1$)",
              "csh_g_l1": "Government consumption share ($t-1$)",
              "open_l1": "Trade openness ($t-1$)",
              "dpop_l1": "Population growth, \\% ($t-1$)",
              "dep_l1": "Age-dependency ratio ($t-1$)",
              "urb_l1": "Urban population share ($t-1$)"}
    lines = [f"{labels[i]} & {int(r['count'])} & {r['mean']:.3f} & {r['std']:.3f} & "
             f"{r['min']:.3f} & {r['max']:.3f}\\\\" for i, r in d.iterrows()]
    (TAB / "descriptives.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "sim"):
        agg, _ = sim_tables()
        sim_figures(agg)
    if what in ("all", "emp"):
        emp_outputs()
