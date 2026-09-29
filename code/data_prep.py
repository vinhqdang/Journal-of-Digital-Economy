"""Build the country-year panel used in the empirical application.

Sources
-------
* Penn World Table 10.0 (Feenstra, Inklaar and Timmer, 2015): output, employment,
  capital, human capital, TFP and expenditure shares.
* ITU / World Bank WDI digital-adoption series, distributed by Our World in Data:
  internet users (% of population), mobile cellular and fixed broadband
  subscriptions per 100 people.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START, END = 1996, 2019


def load_owid(name, col):
    df = pd.read_csv(RAW / f"{name}.csv")
    df = df[df["Code"].notna() & (df["Code"].str.len() == 3)]
    return df.rename(columns={"Code": "iso", "Year": "year", df.columns[3]: col})[["iso", "year", col]]


def build():
    pwt = pd.read_excel(RAW / "pwt100.xlsx", sheet_name="Data")
    pwt = pwt.rename(columns={"countrycode": "iso"})
    keep = ["iso", "country", "year", "rgdpna", "emp", "pop", "hc", "rnna", "rtfpna",
            "csh_i", "csh_g", "csh_x", "csh_m", "labsh"]
    pwt = pwt[keep].sort_values(["iso", "year"])

    net = load_owid("share-of-individuals-using-the-internet", "internet")
    mob = load_owid("mobile-cellular-subscriptions-per-100-people", "mobile")
    bb = load_owid("fixed-broadband-subscriptions-per-100-people", "broadband")

    df = pwt.merge(net, on=["iso", "year"], how="left")
    df = df.merge(mob, on=["iso", "year"], how="left").merge(bb, on=["iso", "year"], how="left")
    # Before broadband roll-out the ITU series is missing rather than zero.
    df.loc[df["year"] < 2000, "broadband"] = df.loc[df["year"] < 2000, "broadband"].fillna(0.0)

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
    panel = df[(df["year"] >= START) & (df["year"] <= END)][cols].dropna(
        subset=[c for c in cols if c != "dtfp"])
    # Trim extreme growth episodes (wars, commodity collapses) at the 1st/99th percentiles.
    lo, hi = panel["dlp"].quantile([0.01, 0.99])
    panel = panel[(panel["dlp"] >= lo) & (panel["dlp"] <= hi)]
    # Keep countries with at least 15 usable years so fixed effects are well identified.
    n = panel.groupby("iso")["year"].transform("size")
    panel = panel[n >= 15].reset_index(drop=True)
    panel.to_csv(OUT / "panel.csv", index=False)
    return panel


if __name__ == "__main__":
    p = build()
    print(p.shape, p["iso"].nunique(), "countries", p["year"].min(), p["year"].max())
    print(p.describe().T.round(3))
