"""
Construye el dataset final analizado y lo guarda en parquet.

Parte de los .dta de ICPSR 34562, limpia los códigos de missing del NLSY97,
une DS0001 con las agresiones a los 24-25 de DS0002 y agrega todas las
variables de ingeniería usadas en los modelos (target de por vida, target
adulto 18-23, conducta temprana 14-17 y agregados de por vida).
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path("/home/cord2108/ITAM/Semestre_9/proyecto_final_polpub")
OUT = BASE / "data/nlsy97_analisis_final.parquet"


def clean_nlsy(df):
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].where(df[num] >= 0, np.nan)
    return df


ds1 = clean_nlsy(pd.read_stata(BASE / "data/ICPSR_34562/DS0001/34562-0001-Data.dta", convert_categoricals=False))
ds2 = clean_nlsy(pd.read_stata(BASE / "data/ICPSR_34562/DS0002/34562-0002-Data.dta", convert_categoricals=False))

# ── Unión: DS0001 + agresión a los 24-25 desde DS0002 ─────────────────────────
df = ds1.merge(ds2[["PUBID", "ASSAULT_CY24", "ASSAULT_CY25"]], on="PUBID", how="left")

AGES = range(14, 26)
EARLY = range(14, 18)
LATE = range(18, 24)


def cols(prefix, ages=AGES):
    return [f"{prefix}{a}" for a in ages if f"{prefix}{a}" in df.columns]


def ssum(d, c):
    return d[c].sum(axis=1, min_count=1)


def ever(d, c):
    sub = d[c]
    out = (sub.max(axis=1) > 0).astype(float)
    out[sub.notna().sum(axis=1) == 0] = np.nan
    return out


assault = cols("ASSAULT_CY")

# ── Targets ───────────────────────────────────────────────────────────────────
df["TOTAL_ASSAULTS"] = df[assault].sum(axis=1, min_count=1)
df["EVER_ASSAULT"] = (df["TOTAL_ASSAULTS"] > 0).astype(float)
df.loc[df["TOTAL_ASSAULTS"].isna(), "EVER_ASSAULT"] = np.nan
df["ASSAULT_18_23"] = ever(df, cols("ASSAULT_CY", LATE))
df["EARLY_ASSAULT"] = ever(df, cols("ASSAULT_CY", EARLY))

# ── Conducta temprana 14-17 (ever) ────────────────────────────────────────────
df["EARLY_DESTROY"] = ever(df, cols("DESTROY_CY", EARLY))
df["EARLY_STEAL_LO"] = ever(df, cols("STEAL_UNDER_50_CY", EARLY))
df["EARLY_STEAL_HI"] = ever(df, cols("STEAL_OVER_50_CY", EARLY))
df["EARLY_SELL_DRUGS"] = ever(df, cols("SELL_DRUGS_CY", EARLY))
df["EARLY_CARR_GUN"] = ever(df, cols("CARR_GUN_CY", EARLY))
df["EARLY_GANG"] = ever(df, cols("GANGS_CY", EARLY))
df["EARLY_ALCOHOL"] = ever(df, cols("ALCOHOL_CY", EARLY))
df["EARLY_MARIJUANA"] = ever(df, cols("MARIJUANA_CY", EARLY))
df["EARLY_HARD_DRUGS"] = ever(df, cols("HARD_DRUGS_CY", EARLY))
df["EARLY_SMOKING"] = ever(df, cols("SMOKING_CY", EARLY))
df["ARR_BY17"] = (df["EVER_ARR_CY17"] > 0).astype(float)
df["INCARC_BY17"] = (df["EVER_INCARC_CY17"] > 0).astype(float)
df["HHINC_EARLY"] = df[cols("HHINC_CY", EARLY)].clip(lower=0).mean(axis=1)

# ── Agregados de por vida (modelo de asociación contemporánea) ────────────────
df["TOTAL_DESTROY"] = ssum(df, cols("DESTROY_CY"))
df["TOTAL_STEAL_LO"] = ssum(df, cols("STEAL_UNDER_50_CY"))
df["TOTAL_STEAL_HI"] = ssum(df, cols("STEAL_OVER_50_CY"))
df["TOTAL_SELL_DRUGS"] = ssum(df, cols("SELL_DRUGS_CY"))
df["TOTAL_CARR_GUN"] = ssum(df, cols("CARR_GUN_CY"))
df["TOTAL_GANG"] = ssum(df, cols("GANGS_CY"))
df["TOTAL_ALCOHOL"] = ssum(df, cols("ALCOHOL_CY"))
df["TOTAL_MARIJUANA"] = ssum(df, cols("MARIJUANA_CY"))
df["TOTAL_HARD_DRUGS"] = ssum(df, cols("HARD_DRUGS_CY"))
df["TOTAL_SMOKING"] = ssum(df, cols("SMOKING_CY"))
df["AVG_HHINC"] = df[cols("HHINC_CY")].clip(lower=0).mean(axis=1)
df["AVG_RINC"] = df[cols("RINC_CY")].clip(lower=0).mean(axis=1)

df.to_parquet(OUT, index=False, engine="pyarrow", compression="snappy")
size_mb = OUT.stat().st_size / 1e6
print(f"Parquet escrito: {OUT}")
print(f"  filas x columnas: {df.shape[0]} x {df.shape[1]}")
print(f"  tamaño: {size_mb:.2f} MB")
print(f"  columnas de ingeniería añadidas: TOTAL_ASSAULTS, EVER_ASSAULT, "
      f"ASSAULT_18_23, EARLY_*, ARR_BY17, INCARC_BY17, HHINC_EARLY, TOTAL_*, AVG_*")
# Validación rápida
print(f"  prevalencia EVER_ASSAULT: {df['EVER_ASSAULT'].mean()*100:.1f}%")
print(f"  prevalencia ASSAULT_18_23: {df['ASSAULT_18_23'].mean()*100:.1f}%")
print(f"  re-lectura OK: {pd.read_parquet(OUT).shape}")
