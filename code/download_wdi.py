"""Download the World Development Indicators series used in the 1996-2025 panel.

Series are saved as long CSV files in data/raw/wdi/ so that the analysis can be rerun offline.
Files that already exist are not downloaded again, so the archived snapshot in data/raw/wdi/
(retrieved from the World Bank API in September 2026, July 2026 WDI release) is used by default;
data/raw/wdi/SHA256SUMS lists its checksums.
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "wdi"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.worldbank.org/v2"

INDICATORS = {
    "SL.GDP.PCAP.EM.KD": "gdp_per_worker",     # GDP per person employed, constant 2021 PPP $
    "IT.NET.USER.ZS": "internet",              # individuals using the internet, % population
    "IT.CEL.SETS.P2": "mobile",                # mobile subscriptions per 100 people
    "IT.NET.BBND.P2": "broadband",             # fixed broadband subscriptions per 100 people
    "NE.GDI.TOTL.ZS": "invest",                # gross capital formation, % GDP
    "NE.CON.GOVT.ZS": "govcons",               # government final consumption, % GDP
    "NE.EXP.GNFS.ZS": "exports",               # exports, % GDP
    "NE.IMP.GNFS.ZS": "imports",               # imports, % GDP
    "SP.POP.GROW": "popgrowth",                # population growth, %
    "SP.URB.TOTL.IN.ZS": "urban",              # urban population, % total
    "SP.POP.DPND": "dependency",               # age dependency ratio, % working-age population
    # robustness and diagnostic series
    "NY.GDP.PCAP.KD": "gdp_pc",                # GDP per capita, constant 2015 US$
    "SP.POP.TOTL": "population",               # total population
    "IQ.SPI.OVRL": "spi",                      # Statistical Performance Indicators, overall score
    "TX.VAL.FUEL.ZS.UN": "fuel_exports",       # fuel exports, % of merchandise exports
    "TX.VAL.MMTL.ZS.UN": "ores_exports",       # ores and metals exports, % of merchandise exports
    "IT.MLT.MAIN.P2": "fixed_lines",           # fixed telephone subscriptions per 100 people
    "HD.HCI.LAYS": "lays",                     # learning-adjusted years of schooling
}


def get(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # network hiccups: back off and retry
            print("retry", k + 1, url[:90], e)
            time.sleep(2 ** k)
    raise RuntimeError(url)


def countries():
    d = get(f"{API}/country?format=json&per_page=400")
    rows = [(c["id"], c["name"], c["region"]["value"], c["incomeLevel"]["value"]) for c in d[1]
            if c["region"]["value"] != "Aggregates"]
    return pd.DataFrame(rows, columns=["iso", "country", "region", "income"])


def main():
    meta = countries()
    meta.to_csv(OUT / "countries.csv", index=False)
    for code, name in INDICATORS.items():
        f = OUT / f"{name}.csv"
        if f.exists():
            continue
        d = get(f"{API}/country/all/indicator/{code}?format=json&per_page=20000&date=1985:2025")
        rows = [(r["countryiso3code"], int(r["date"]), r["value"]) for r in d[1]
                if r["value"] is not None and r["countryiso3code"] in set(meta["iso"])]
        pd.DataFrame(rows, columns=["iso", "year", name]).to_csv(f, index=False)
        print(code, len(rows))
    write_checksums()


def write_checksums():
    import hashlib
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}"
             for f in sorted(OUT.glob("*.csv"))]
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
