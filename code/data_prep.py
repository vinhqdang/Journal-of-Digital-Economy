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
    """Before a country's first reported broadband figure the service did not exist: set to zero."""
    first = df[df["broadband"].notna()].groupby("iso")["year"].min()
    pre = df["year"] < df["iso"].map(first).fillna(9999)
    df.loc[pre & df["broadband"].isna(), "broadband"] = 0.0


def build_pwt(fname="pwt100.xlsx", end=END, out="panel_pwt.csv"):
    pwt = pd.read_excel(RAW / fname, sheet_name="Data")
    pwt = pwt.rename(columns={"countrycode": "iso"})
    keep = ["iso", "country", "year", "rgdpna", "emp", "pop", "hc", "rnna", "rtfpna",
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

    cols = ["iso", "country", "year", "dlp", "dtfp", "dkl", "internet_l1", "internet_l2",
            "mobile_l1", "broadband_l1", "lp_l1", "kl_l1", "hc_l1", "csh_i_l1", "csh_g_l1",
            "open_l1", "dpop_l1", "labsh_l1", "dlp_l1"]
    panel = df[(df["year"] >= START) & (df["year"] <= end)][cols].dropna(
        subset=[c for c in cols if c != "dtfp"])
    # Trim extreme growth episodes (wars, commodity collapses) at the 1st/99th percentiles.
    lo, hi = panel["dlp"].quantile([0.01, 0.99])
    panel = panel[(panel["dlp"] >= lo) & (panel["dlp"] <= hi)]
    # Keep countries with at least 15 usable years so fixed effects are well identified.
    n = panel.groupby("iso")["year"].transform("size")
    panel = panel[n >= 15].reset_index(drop=True)
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


def build_wdi():
    w = RAW / "wdi"
    meta = pd.read_csv(w / "countries.csv")
    df = None
    for f in ["gdp_per_worker", "internet", "mobile", "broadband", "invest", "govcons", "exports",
              "imports", "popgrowth", "urban", "dependency"]:
        x = pd.read_csv(w / f"{f}.csv")
        df = x if df is None else df.merge(x, on=["iso", "year"], how="outer")
    df = df.merge(load_mys(), on=["iso", "year"], how="left")
    df = df.merge(meta[["iso", "country", "income"]], on="iso", how="left")
    df = df.sort_values(["iso", "year"]).reset_index(drop=True)
    fill_pre_rollout(df)

    g = df.groupby("iso", group_keys=False)
    df["lp"] = np.log(df["gdp_per_worker"])
    df["dlp"] = 100 * g["lp"].diff()
    df["open"] = (df["exports"] + df["imports"]) / 100
    df["csh_i"] = df["invest"] / 100
    df["csh_g"] = df["govcons"] / 100
    df["hc"] = df["mys"]
    df["dpop"] = df["popgrowth"]
    df["dep"] = df["dependency"] / 100
    df["urb"] = df["urban"] / 100
    for c in ["internet", "mobile", "broadband"]:
        df[f"{c}_l1"] = g[c].shift(1) / 100.0
    df["internet_l2"] = g["internet"].shift(2) / 100.0
    for c in ["lp", "hc", "csh_i", "csh_g", "open", "dpop", "dep", "urb", "dlp"]:
        df[f"{c}_l1"] = g[c].shift(1)

    cols = ["iso", "country", "income", "year", "dlp", "internet_l1", "internet_l2", "mobile_l1",
            "broadband_l1", "lp_l1", "hc_l1", "csh_i_l1", "csh_g_l1", "open_l1", "dpop_l1",
            "dep_l1", "urb_l1", "dlp_l1"]
    panel = df[(df["year"] >= START) & (df["year"] <= WDI_END)][cols].dropna()
    lo, hi = panel["dlp"].quantile([0.01, 0.99])
    panel = panel[(panel["dlp"] >= lo) & (panel["dlp"] <= hi)]
    n = panel.groupby("iso")["year"].transform("size")
    panel = panel[n >= 15].reset_index(drop=True)
    panel.to_csv(OUT / "panel.csv", index=False)
    return panel


if __name__ == "__main__":
    q = build_pwt()
    print("PWT 10.0 panel", q.shape, q["iso"].nunique(), "countries", q["year"].min(), q["year"].max())
    q = build_pwt("pwt110.xlsx", END_PWT11, "panel_pwt11.csv")
    print("PWT 11.0 panel", q.shape, q["iso"].nunique(), "countries", q["year"].min(), q["year"].max())
    p = build_wdi()
    print(p.shape, p["iso"].nunique(), "countries", p["year"].min(), p["year"].max())
    print(p.describe().T.round(3))
