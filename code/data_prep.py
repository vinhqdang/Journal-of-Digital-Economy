"""Build the country-year panels used in the empirical application.

Main panel (data/processed/panel.csv), 1996-2025
* World Bank World Development Indicators (July 2026 release, downloaded by
  download_wdi.py): GDP per person employed (constant 2021 PPP $), ITU digital-adoption
  series, expenditure shares, demography.
* UNDP Human Development Report 2025: mean years of schooling (1990-2023).

PWT panels, used for robustness
* data/processed/panel_pwt11.csv: Penn World Table 11.0, 1996-2023
* data/processed/panel_pwt.csv:   Penn World Table 10.0, 1996-2019
  (Feenstra, Inklaar and Timmer, 2015): output, employment, capital, human capital,
  TFP and expenditure shares.
* ITU / World Bank digital-adoption series distributed by Our World in Data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START, END = 1996, 2019          # PWT 10.0 ends in 2019
END_PWT11 = 2023                 # PWT 11.0 ends in 2023
WDI_END = 2025


def load_owid(name, col):
    df = pd.read_csv(RAW / f"{name}.csv")
    df = df[df["Code"].notna() & (df["Code"].str.len() == 3)]
    return df.rename(columns={"Code": "iso", "Year": "year", df.columns[3]: col})[["iso", "year", col]]


def fill_pre_rollout(df):
    """Structural zeros: before a country's first reported broadband figure the service is taken
    not to exist and the missing values are set to zero.  Countries that never report broadband
    are left missing (unknown, not zero).  The unfilled series is kept as broadband_nf."""
    df["broadband_nf"] = df["broadband"]
    first = df[df["broadband"].notna()].groupby("iso")["year"].min()
    pre = df["year"] < df["iso"].map(first)
    df.loc[pre & df["broadband"].isna(), "broadband"] = 0.0


def initial_level(df, col):
    """Initial (pre-sample) level: mean over 1991-1995 of the available values; for countries
    without any value in that window, the first available later value.  Returns the level and a
    flag for countries with a pre-sample value."""
    early = df[df["year"].between(1991, 1995) & df[col].notna()].groupby("iso")[col].mean()
    first_any = df[df[col].notna()].sort_values("year").groupby("iso")[col].first()
    return (df["iso"].map(early).fillna(df["iso"].map(first_any)), df["iso"].isin(early.index))


def finish(base, required, trim=None, min_years=15):
    """Trim the outcome at its 1st/99th percentiles (or at given cutoffs) and keep countries with
    at least min_years usable observations.  Returns the panel and the cutoffs."""
    base = base.dropna(subset=required)
    lo, hi = base["dlp"].quantile([0.01, 0.99]) if trim is None else trim
    panel = base[(base["dlp"] >= lo) & (base["dlp"] <= hi)]
    panel = panel[panel.groupby("iso")["year"].transform("size") >= min_years]
    return panel.reset_index(drop=True), (lo, hi)


def build_pwt(fname="pwt100.xlsx", end=END, out="panel_pwt.csv"):
    pwt = pd.read_excel(RAW / fname, sheet_name="Data")
    pwt = pwt.rename(columns={"countrycode": "iso"})
    keep = ["iso", "country", "year", "rgdpna", "cgdpo", "emp", "pop", "hc", "rnna", "rtfpna",
            "csh_i", "csh_g", "csh_x", "csh_m", "labsh"]
    pwt = pwt[keep].sort_values(["iso", "year"])

    net = load_owid("share-of-individuals-using-the-internet", "internet")
    mob = load_owid("mobile-cellular-subscriptions-per-100-people", "mobile")
    bb = load_owid("fixed-broadband-subscriptions-per-100-people", "broadband")

    df = pwt.merge(net, on=["iso", "year"], how="left")
    df = df.merge(mob, on=["iso", "year"], how="left").merge(bb, on=["iso", "year"], how="left")
    df = df.sort_values(["iso", "year"]).reset_index(drop=True)
    fill_pre_rollout(df)

    g = df.groupby("iso", group_keys=False)
    df["lp"] = np.log(df["rgdpna"] / df["emp"])
    df["kl"] = np.log(df["rnna"] / df["emp"])
    df["tfp"] = np.log(df["rtfpna"])
    # Outcome: annual labour-productivity growth in percent.
    df["dlp"] = 100 * g["lp"].diff()
    df["dtfp"] = 100 * g["tfp"].diff()
    df["dkl"] = 100 * g["kl"].diff()
    df["dpop"] = 100 * g["pop"].apply(lambda s: np.log(s).diff())
    df["open"] = df["csh_x"] - df["csh_m"]
    # Digital adoption enters with a one-year lag; shares are rescaled to [0, 1].
    for c in ["internet", "mobile", "broadband"]:
        df[f"{c}_l1"] = g[c].shift(1) / 100.0
    df["internet_l2"] = g["internet"].shift(2) / 100.0
    for c in ["lp", "hc", "csh_i", "csh_g", "open", "dpop", "labsh", "kl"]:
        df[f"{c}_l1"] = g[c].shift(1)
    df["dlp_l1"] = g["dlp"].shift(1)

    # Initial productivity for cross-country comparison: output-side real GDP at current PPPs
    # (cgdpo) per person engaged, the PWT measure of relative productive capacity at a point in
    # time; growth rates use rgdpna.  The national-prices level is kept for a sensitivity check.
    df["lp_ppp"] = np.log(df["cgdpo"] / df["emp"])
    df["lp_init"], df["lp_init_pre"] = initial_level(df, "lp_ppp")
    df["lp_init_na"], _ = initial_level(df, "lp")
    cols = ["iso", "country", "year", "dlp", "dtfp", "dkl", "internet_l1", "internet_l2",
            "mobile_l1", "broadband_l1", "lp_l1", "kl_l1", "hc_l1", "csh_i_l1", "csh_g_l1",
            "open_l1", "dpop_l1", "labsh_l1", "dlp_l1", "lp_init", "lp_init_na", "lp_init_pre"]
    # Sample: only the outcome, the treatment and the baseline controls must be observed.
    required = ["dlp", "internet_l1", "hc_l1", "csh_g_l1", "open_l1", "dpop_l1", "labsh_l1",
                "lp_init"]
    base = df[(df["year"] >= START) & (df["year"] <= end)][cols]
    panel, _ = finish(base, required)
    panel.to_csv(OUT / out, index=False)
    return panel


def load_mys():
    """UNDP mean years of schooling, long format; 2024 extrapolated from the 2019-2023 trend."""
    h = pd.read_csv(RAW / "HDR25_Composite_indices_complete_time_series.csv", encoding="latin1")
    cols = [c for c in h.columns if c.startswith("mys_") and c[4:].isdigit()]
    m = h[["iso3"] + cols].melt(id_vars="iso3", var_name="year", value_name="mys")
    m["year"] = m["year"].str[4:].astype(int)
    m = m.rename(columns={"iso3": "iso"}).dropna()
    last = m[m["year"].between(2019, 2023)]
    slope = last.groupby("iso").apply(
        lambda g: np.polyfit(g["year"], g["mys"], 1)[0] if len(g) >= 3 else 0.0,
        include_groups=False)
    base = m[m["year"] == 2023].set_index("iso")["mys"]
    ext = (base + slope.reindex(base.index).fillna(0.0)).rename("mys").reset_index()
    ext["year"] = 2024
    return pd.concat([m, ext], ignore_index=True)


def screen_adoption(df):
    """Flag implausible values and jumps in the ITU adoption series (set to missing).

    Rules: internet share outside [0, 100] or a year-on-year change above 25 points; mobile
    subscriptions above 250 per 100 people or a change above 100 per 100; broadband above 60 per
    100 or a change above 20 per 100.  The flagged country-years are written to a log.
    """
    rules = {"internet": (100, 25), "mobile": (250, 100), "broadband": (60, 20)}
    log = []
    g = df.groupby("iso", group_keys=False)
    for c, (cap, jump) in rules.items():
        d = g[c].diff().abs()
        bad = (df[c] > cap) | (df[c] < 0) | (d > jump)
        for _, r in df[bad].iterrows():
            log.append({"iso": r["iso"], "year": int(r["year"]), "series": c, "value": r[c]})
        df.loc[bad, c + "_scr"] = np.nan
        df.loc[~bad, c + "_scr"] = df.loc[~bad, c]
    pd.DataFrame(log).to_csv(OUT / "screening_log.csv", index=False)
    return df


def build_wdi():
    w = RAW / "wdi"
    meta = pd.read_csv(w / "countries.csv")
    df = None
    for f in ["gdp_per_worker", "internet", "mobile", "broadband", "invest", "govcons", "exports",
              "imports", "popgrowth", "urban", "dependency", "gdp_pc", "population",
              "fixed_lines"]:
        x = pd.read_csv(w / f"{f}.csv")
        df = x if df is None else df.merge(x, on=["iso", "year"], how="outer")
    df = df.merge(load_mys(), on=["iso", "year"], how="left")
    df = df.merge(meta[["iso", "country", "income"]], on="iso", how="left")
    df = df.merge(meta[["iso", "region"]], on="iso", how="left")
    df = df.sort_values(["iso", "year"]).reset_index(drop=True)
    fill_pre_rollout(df)
    df = screen_adoption(df)
    df["broadband_nf_scr"] = df["broadband_scr"].where(df["broadband_nf"].notna())

    g = df.groupby("iso", group_keys=False)
    df["lp"] = np.log(df["gdp_per_worker"])
    df["dlp"] = 100 * g["lp"].diff()
    df["dgdppc"] = 100 * g["gdp_pc"].apply(lambda s: np.log(s).diff())
    df["open"] = (df["exports"] + df["imports"]) / 100
    df["csh_i"] = df["invest"] / 100
    df["csh_g"] = df["govcons"] / 100
    df["hc"] = df["mys"]
    df["dpop"] = df["popgrowth"]
    df["dep"] = df["dependency"] / 100
    df["urb"] = df["urban"] / 100
    for c in ["internet", "mobile", "broadband"]:
        df[f"{c}_l1"] = g[c + "_scr"].shift(1) / 100.0
        df[f"{c}_raw_l1"] = g[c].shift(1) / 100.0
    df["broadband_nf_l1"] = g["broadband_nf_scr"].shift(1) / 100.0
    df["internet_l2"] = g["internet_scr"].shift(2) / 100.0
    # leads of the treatment D_t = internet_{t-1} (strict-exogeneity test): D_{t+1} = internet_t
    df["internet_f1"] = df["internet_scr"] / 100.0
    df["internet_f2"] = g["internet_scr"].shift(-1) / 100.0
    for c in ["lp", "hc", "csh_i", "csh_g", "open", "dpop", "dep", "urb", "dlp"]:
        df[f"{c}_l1"] = g[c].shift(1)
    # cumulative growth from t-1 to t+h (local projections), h = 0 is the baseline outcome
    for h in range(0, 9):
        df[f"cum{h}"] = 100 * (g["lp"].shift(-h) - df["lp_l1"])

    # predetermined moderators and time-invariant country characteristics
    df["lp_init"], df["lp_init_pre"] = initial_level(df, "lp")
    us = df[df["iso"] == "USA"].set_index("year")["lp"]
    df["dist_us_l1"] = df["lp_l1"] - (df["year"] - 1).map(us)
    take = df[df["internet_scr"] >= 10].groupby("iso")["year"].min()
    df["ysince_l1"] = (df["year"] - 1) - df["iso"].map(take)
    # predetermined version: zero until take-off (known at t-1), years since take-off after it;
    # countries that never reach 10% are at zero throughout
    df["ysince_pre_l1"] = df["ysince_l1"].clip(lower=0).fillna(0.0)
    df["ryear"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    df["region_code"] = pd.factorize(df["region"])[0]
    fl90 = df[df["year"] == 1990].set_index("iso")["fixed_lines"]
    df["fixed90"] = df["iso"].map(fl90) / 100.0
    for f, col in [("spi", "spi"), ("lays", "lays")]:
        x = pd.read_csv(w / f"{f}.csv").groupby("iso")[col].mean()
        df[col + "_mean"] = df["iso"].map(x)
    fuel = pd.read_csv(w / "fuel_exports.csv")
    ores = pd.read_csv(w / "ores_exports.csv")
    fo = (fuel[fuel["year"] >= 1996].groupby("iso")["fuel_exports"].mean().add(
        ores[ores["year"] >= 1996].groupby("iso")["ores_exports"].mean(), fill_value=0))
    df["resource_share"] = df["iso"].map(fo)

    cols = ["iso", "country", "income", "region", "year", "dlp", "internet_l1", "internet_l2",
            "internet_f1", "internet_f2", "mobile_l1", "broadband_l1", "broadband_nf_l1", "lp_l1",
            "hc_l1", "csh_i_l1", "csh_g_l1", "open_l1", "dpop_l1", "dep_l1", "urb_l1", "dlp_l1"]
    extra = ["dgdppc", "population", "internet_raw_l1", "mobile_raw_l1", "broadband_raw_l1",
             "lp_init", "lp_init_pre", "dist_us_l1", "ysince_l1", "ysince_pre_l1", "ryear",
             "region_code", "fixed90", "spi_mean", "lays_mean", "resource_share"] + \
        [f"cum{h}" for h in range(9)]
    # Baseline sample: only the outcome, the treatment and the baseline controls must be observed
    # (variables used only in robustness checks do not restrict the sample).
    required = ["dlp", "internet_l1", "hc_l1", "csh_g_l1", "open_l1", "dpop_l1", "dep_l1",
                "urb_l1", "lp_init"]
    base = df[(df["year"] >= START) & (df["year"] <= WDI_END)][cols + extra]
    panel, cut = finish(base, required)
    panel.to_csv(OUT / "panel.csv", index=False)
    keep = set(panel["iso"])
    # common sample of the earlier version: every ICT, investment and lagged-outcome variable
    # observed as well (matched-sample robustness check)
    common_req = ["dlp", "internet_l1", "internet_l2", "mobile_l1", "broadband_l1", "lp_l1",
                  "hc_l1", "csh_i_l1", "csh_g_l1", "open_l1", "dpop_l1", "dep_l1", "urb_l1",
                  "dlp_l1", "lp_init"]
    common, _ = finish(base, common_req)
    common.to_csv(OUT / "panel_common.csv", index=False)
    # untrimmed outcome, same countries as the baseline (trimming check)
    unt = base.dropna(subset=required)
    unt = unt[unt["iso"].isin(keep)].reset_index(drop=True)
    unt.to_csv(OUT / "panel_untrimmed.csv", index=False)
    # unscreened adoption series, baseline trimming cutoffs and countries (screening check)
    raw = df.copy()
    raw["internet_l1"], raw["mobile_l1"] = raw["internet_raw_l1"], raw["mobile_raw_l1"]
    raw["broadband_l1"] = raw["broadband_raw_l1"]
    raw["internet_l2"] = g["internet"].shift(2) / 100.0
    rb = raw[(raw["year"] >= START) & (raw["year"] <= WDI_END)][cols + extra]
    rb, _ = finish(rb, required, trim=cut)
    rb[rb["iso"].isin(keep)].reset_index(drop=True).to_csv(OUT / "panel_unscreened.csv",
                                                           index=False)
    # sample flow: observations lost at each step
    yrs = df[(df["year"] >= START) & (df["year"] <= WDI_END)]
    flow = [("Country-years 1996-2025 (WDI economies)", len(yrs), yrs["iso"].nunique())]
    step = yrs
    for lab, c in [("Outcome observed", "dlp"), ("Treatment observed", "internet_l1"),
                   ("Schooling observed", "hc_l1"), ("Government share observed", "csh_g_l1"),
                   ("Openness observed", "open_l1"), ("Population growth observed", "dpop_l1"),
                   ("Dependency ratio observed", "dep_l1"), ("Urban share observed", "urb_l1"),
                   ("Initial productivity observed", "lp_init")]:
        step = step[step[c].notna()]
        flow.append((lab, len(step), step["iso"].nunique()))
    step = step[(step["dlp"] >= cut[0]) & (step["dlp"] <= cut[1])]
    flow.append(("Outcome within 1st-99th percentiles", len(step), step["iso"].nunique()))
    flow.append(("Countries with at least 15 years (baseline)", len(panel), panel["iso"].nunique()))
    flow.append(("Common sample of the earlier version", len(common), common["iso"].nunique()))
    pd.DataFrame(flow, columns=["step", "obs", "countries"]).to_csv(OUT / "sample_flow.csv",
                                                                    index=False)
    # WDI economies absent from the sample, by income group (coverage diagnostic)
    meta[["iso", "country", "income"]].assign(in_sample=meta["iso"].isin(panel["iso"])).to_csv(
        OUT / "coverage.csv", index=False)
    return panel


if __name__ == "__main__":
    q = build_pwt()
    print("PWT 10.0 panel", q.shape, q["iso"].nunique(), "countries", q["year"].min(), q["year"].max())
    q = build_pwt("pwt110.xlsx", END_PWT11, "panel_pwt11.csv")
    print("PWT 11.0 panel", q.shape, q["iso"].nunique(), "countries", q["year"].min(), q["year"].max())
    p = build_wdi()
    print(p.shape, p["iso"].nunique(), "countries", p["year"].min(), p["year"].max())
    print(p.describe().T.round(3))
