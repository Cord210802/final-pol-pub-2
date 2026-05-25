"""
Modelos de INICIO vs PERSISTENCIA de la violencia (ICPSR 34562, NLSY97).

Parte del modelo predictivo 14-17 -> 18-23 y lo divide en dos submuestras
según la agresión temprana (EARLY_ASSAULT), excluyéndola como predictor:

  - INICIO:       individuos SIN agresión a los 14-17. ¿Qué predice que la
                  inicien a los 18-23?
  - PERSISTENCIA: individuos CON agresión a los 14-17. ¿Qué predice que la
                  continúen a los 18-23?

Esto separa los factores de inicio de los de continuidad de la conducta.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings("ignore")
sns.set_palette("muted")
plt.style.use("seaborn-v0_8-whitegrid")

BASE = Path("/home/cord2108/ITAM/Semestre_9/proyecto_final_polpub")
FIG = BASE / "analisis/figuras"
RES = {}


def clean_nlsy(df):
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].where(df[num] >= 0, np.nan)
    return df


ds1 = clean_nlsy(pd.read_stata(BASE / "data/ICPSR_34562/DS0001/34562-0001-Data.dta", convert_categoricals=False))
EARLY = range(14, 18)
LATE = range(18, 24)


def ecols(prefix, ages):
    return [f"{prefix}{a}" for a in ages if f"{prefix}{a}" in ds1.columns]


def ever(d, cols):
    sub = d[cols]
    out = (sub.max(axis=1) > 0).astype(float)
    out[sub.notna().sum(axis=1) == 0] = np.nan
    return out


def mean_pos(d, cols):
    return d[cols].clip(lower=0).mean(axis=1)


late_assault = ecols("ASSAULT_CY", LATE)
ds = ds1[ds1[late_assault].notna().any(axis=1)].copy()
ds["ASSAULT_18_23"] = ever(ds, late_assault)
ds = ds[ds["ASSAULT_18_23"].notna()].copy()
ds["EARLY_ASSAULT"] = ever(ds, ecols("ASSAULT_CY", EARLY))
ds = ds[ds["EARLY_ASSAULT"].notna()].copy()

# Predictores (SIN EARLY_ASSAULT)
X = pd.DataFrame(index=ds.index)
X["EARLY_DESTROY"] = ever(ds, ecols("DESTROY_CY", EARLY))
X["EARLY_STEAL_LO"] = ever(ds, ecols("STEAL_UNDER_50_CY", EARLY))
X["EARLY_STEAL_HI"] = ever(ds, ecols("STEAL_OVER_50_CY", EARLY))
X["EARLY_SELL_DRUGS"] = ever(ds, ecols("SELL_DRUGS_CY", EARLY))
X["EARLY_CARR_GUN"] = ever(ds, ecols("CARR_GUN_CY", EARLY))
X["EARLY_GANG"] = ever(ds, ecols("GANGS_CY", EARLY))
X["EARLY_ALCOHOL"] = ever(ds, ecols("ALCOHOL_CY", EARLY))
X["EARLY_MARIJUANA"] = ever(ds, ecols("MARIJUANA_CY", EARLY))
X["EARLY_HARD_DRUGS"] = ever(ds, ecols("HARD_DRUGS_CY", EARLY))
X["EARLY_SMOKING"] = ever(ds, ecols("SMOKING_CY", EARLY))
X["ARR_BY17"] = (ds["EVER_ARR_CY17"] > 0).astype(float)
X["INCARC_BY17"] = (ds["EVER_INCARC_CY17"] > 0).astype(float)
X["HHINC_EARLY"] = mean_pos(ds, ecols("HHINC_CY", EARLY))
X["ASVAB_SCORE"] = ds["ASVAB_SCORE"]
X["POVRATIO97"] = ds["POVRATIO97"]
X["NUMSIBS"] = ds["NUMSIBS"]
X["BIOMOMAGE"] = ds["BIOMOMAGE"]
X["SEX"] = ds["SEX"]
X["RACE_ETHNICITY"] = ds["RACE_ETHNICITY"]
X["FMSTRC97"] = ds["FMSTRC97"]
X["RESPARED"] = ds["RESPARED"]
X["MOMEMP97"] = ds["MOMEMP97"]
X["DADEMP97"] = ds["DADEMP97"]
feat = list(X.columns)

imp = SimpleImputer(strategy="median")
Xi = pd.DataFrame(imp.fit_transform(X), columns=feat, index=X.index)
y_all = ds["ASSAULT_18_23"].astype(int)
early = ds["EARLY_ASSAULT"].astype(int)


def run_model(mask, nombre):
    Xm, ym = Xi[mask], y_all[mask]
    info = {"n": int(mask.sum()), "prevalencia_%": round(float(ym.mean() * 100), 2),
            "n_pos": int(ym.sum()), "n_neg": int((ym == 0).sum())}
    Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.25, random_state=42, stratify=ym)
    rf = RandomForestClassifier(n_estimators=400, max_depth=9, min_samples_leaf=20,
                                max_features="sqrt", class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    prob = rf.predict_proba(Xte)[:, 1]
    info["auc_test"] = round(float(roc_auc_score(yte, prob)), 4)
    cv = cross_val_score(rf, Xm, ym, cv=5, scoring="roc_auc", n_jobs=-1)
    info["auc_cv_mean"] = round(float(cv.mean()), 4)
    info["auc_cv_std"] = round(float(cv.std()), 4)
    gini = pd.Series(rf.feature_importances_, index=feat).sort_values(ascending=False)
    perm = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=42, scoring="roc_auc", n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=feat).sort_values(ascending=False)
    out = {"info": info,
           "gini": {k: round(float(v), 4) for k, v in gini.items()},
           "perm": {k: round(float(v), 5) for k, v in perm_s.items()}}
    print(f"\n===== {nombre} =====")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("  GINI (top 10):")
    for k, v in list(out["gini"].items())[:10]:
        print(f"    {k:18s} {v}")
    print("  PERMUTACION (top 10):")
    for k, v in list(out["perm"].items())[:10]:
        print(f"    {k:18s} {v}")
    # figura
    fig, ax = plt.subplots(figsize=(8, 7))
    g = gini.head(15)[::-1]
    ax.barh(range(len(g)), g.values, color="#dd8452" if "INICIO" in nombre else "#c44e52")
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=9)
    ax.set_xlabel("Importancia de Gini"); ax.set_title(f"{nombre} (RF, Gini)")
    fig.tight_layout()
    fig.savefig(FIG / f"modelo_{nombre.lower().split()[0]}.png", dpi=130)
    plt.close(fig)
    return out


RES["INICIO"] = run_model(early == 0, "INICIO (sin agresion temprana)")
RES["PERSISTENCIA"] = run_model(early == 1, "PERSISTENCIA (con agresion temprana)")

with open(BASE / "analisis/resultados_inicio_persistencia.json", "w") as f:
    json.dump(RES, f, indent=2, ensure_ascii=False)
print("\nOK -> analisis/resultados_inicio_persistencia.json")
