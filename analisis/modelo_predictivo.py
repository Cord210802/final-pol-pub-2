"""
Diseño PREDICTIVO de riesgo temprano sobre ICPSR 34562 (NLSY97).

Orden temporal estricto (sin circularidad):
  - Predictores: perfil base 1997 + conducta/justicia en la adolescencia (14-17).
  - Target:      ASSAULT_18_23 = autoreportó >=1 agresión entre los 18 y 23 años.

Métodos: Random Forest (importancia de Gini) + permutación (robustez) + PCA.
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

EARLY = range(14, 18)   # 14,15,16,17
LATE = range(18, 24)    # 18..23


def ecols(prefix, ages):
    return [f"{prefix}{a}" for a in ages if f"{prefix}{a}" in ds1.columns]


def ever(d, cols):
    """1 si hizo la conducta en algún año de la ventana; NaN si nunca observado."""
    sub = d[cols]
    out = (sub.max(axis=1) > 0).astype(float)
    out[sub.notna().sum(axis=1) == 0] = np.nan
    return out


def mean_pos(d, cols):
    return d[cols].clip(lower=0).mean(axis=1)


# ── Target: agresión a los 18-23 ──────────────────────────────────────────────
late_assault = ecols("ASSAULT_CY", LATE)
obs_late = ds1[late_assault].notna().any(axis=1)
ds = ds1[obs_late].copy()
ds["ASSAULT_18_23"] = ever(ds, late_assault)
ds = ds[ds["ASSAULT_18_23"].notna()].copy()

RES["target"] = {
    "ventana_predictores": "14-17",
    "ventana_target": "18-23",
    "n_con_obs_target": int(obs_late.sum()),
    "n_modelable": int(len(ds)),
    "prevalencia_%": round(float(ds["ASSAULT_18_23"].mean() * 100), 2),
    "n_pos": int(ds["ASSAULT_18_23"].sum()),
    "n_neg": int((ds["ASSAULT_18_23"] == 0).sum()),
}

# ── Predictores (14-17 + base 1997) ───────────────────────────────────────────
X = pd.DataFrame(index=ds.index)

# Conducta delictiva temprana (ever 14-17)
X["EARLY_ASSAULT"] = ever(ds, ecols("ASSAULT_CY", EARLY))
X["EARLY_DESTROY"] = ever(ds, ecols("DESTROY_CY", EARLY))
X["EARLY_STEAL_LO"] = ever(ds, ecols("STEAL_UNDER_50_CY", EARLY))
X["EARLY_STEAL_HI"] = ever(ds, ecols("STEAL_OVER_50_CY", EARLY))
X["EARLY_SELL_DRUGS"] = ever(ds, ecols("SELL_DRUGS_CY", EARLY))
X["EARLY_CARR_GUN"] = ever(ds, ecols("CARR_GUN_CY", EARLY))
X["EARLY_GANG"] = ever(ds, ecols("GANGS_CY", EARLY))
# Uso de sustancias temprano
X["EARLY_ALCOHOL"] = ever(ds, ecols("ALCOHOL_CY", EARLY))
X["EARLY_MARIJUANA"] = ever(ds, ecols("MARIJUANA_CY", EARLY))
X["EARLY_HARD_DRUGS"] = ever(ds, ecols("HARD_DRUGS_CY", EARLY))
X["EARLY_SMOKING"] = ever(ds, ecols("SMOKING_CY", EARLY))
# Contacto temprano con la justicia (acumulado a los 17)
X["ARR_BY17"] = (ds["EVER_ARR_CY17"] > 0).astype(float) if "EVER_ARR_CY17" in ds else np.nan
X["INCARC_BY17"] = (ds["EVER_INCARC_CY17"] > 0).astype(float) if "EVER_INCARC_CY17" in ds else np.nan
# Económico temprano
X["HHINC_EARLY"] = mean_pos(ds, ecols("HHINC_CY", EARLY))
# Perfil base 1997 (estático)
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

y = ds["ASSAULT_18_23"].astype(int)
feature_cols = list(X.columns)

RES["cobertura_predictores_%"] = {
    c: round(float(X[c].notna().mean() * 100), 1) for c in feature_cols
}

imp = SimpleImputer(strategy="median")
Xi = pd.DataFrame(imp.fit_transform(X), columns=feature_cols, index=X.index)

X_tr, X_te, y_tr, y_te = train_test_split(Xi, y, test_size=0.2, random_state=42, stratify=y)

# ── Random Forest ─────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(X_tr, y_tr)
y_prob = rf.predict_proba(X_te)[:, 1]
y_pred = rf.predict(X_te)
auc = roc_auc_score(y_te, y_prob)
cv = cross_val_score(rf, Xi, y, cv=5, scoring="roc_auc", n_jobs=-1)
RES["modelo"] = {
    "n_train": len(X_tr), "n_test": len(X_te), "n_features": len(feature_cols),
    "auc_test": round(float(auc), 4),
    "auc_cv_mean": round(float(cv.mean()), 4), "auc_cv_std": round(float(cv.std()), 4),
}
RES["classification_report"] = classification_report(
    y_te, y_pred, target_names=["Sin asalto", "Con asalto"], output_dict=True)

gini = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
RES["gini"] = {k: round(float(v), 4) for k, v in gini.items()}

perm = permutation_importance(rf, X_te, y_te, n_repeats=30, random_state=42, scoring="roc_auc", n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
RES["perm"] = {k: round(float(v), 5) for k, v in perm_s.items()}

# ── PCA (solo variables continuas/ordinales tempranas) ────────────────────────
pca_vars = [
    "EARLY_ASSAULT", "EARLY_DESTROY", "EARLY_STEAL_LO", "EARLY_STEAL_HI",
    "EARLY_SELL_DRUGS", "EARLY_CARR_GUN", "EARLY_GANG", "EARLY_ALCOHOL",
    "EARLY_MARIJUANA", "EARLY_HARD_DRUGS", "EARLY_SMOKING", "ARR_BY17",
    "INCARC_BY17", "HHINC_EARLY", "ASVAB_SCORE", "POVRATIO97", "NUMSIBS", "BIOMOMAGE",
]
Xp = StandardScaler().fit_transform(Xi[pca_vars])
pca = PCA().fit(Xp)
evr = pca.explained_variance_ratio_
RES["pca_evr"] = [round(float(v), 4) for v in evr[:10]]
RES["pca_cum"] = [round(float(v), 4) for v in np.cumsum(evr)[:10]]
load = pd.DataFrame(pca.components_[:4].T, index=pca_vars, columns=["PC1", "PC2", "PC3", "PC4"])
RES["pca_loadings"] = load.round(3).to_dict()
scores = pca.transform(Xp)
RES["pca_corr_target"] = {f"PC{i+1}": round(float(np.corrcoef(scores[:, i], y)[0, 1]), 4) for i in range(4)}

# ── Contrastes bivariados ─────────────────────────────────────────────────────
def rate_by(col, labels):
    r = ds.groupby(col)["ASSAULT_18_23"].mean() * 100
    return {labels.get(k, str(k)): round(float(v), 1) for k, v in r.items()}

RES["tasa_sexo"] = rate_by("SEX", {1: "Hombre", 2: "Mujer"})
RES["tasa_raza"] = rate_by("RACE_ETHNICITY", {1: "Negro", 2: "Hispano", 3: "Mixto", 4: "Blanco/Otro"})
RES["tasa_familia"] = rate_by("FMSTRC97", {1: "Ambos biologicos", 2: "Un padre", 3: "Bio+padrastro", 4: "Adoptivos", 5: "Otro"})
# Tasa de asalto adulto según asalto temprano (persistencia)
RES["tasa_por_early_assault"] = {
    "sin asalto temprano": round(float(ds.loc[X["EARLY_ASSAULT"] == 0, "ASSAULT_18_23"].mean() * 100), 1),
    "con asalto temprano": round(float(ds.loc[X["EARLY_ASSAULT"] == 1, "ASSAULT_18_23"].mean() * 100), 1),
}

# ── Figuras ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
g = gini.head(20)[::-1]
ax.barh(range(len(g)), g.values, color="#4c72b0")
ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=9)
ax.set_xlabel("Importancia de Gini"); ax.set_title("Importancia de variables (RF, Gini) — predictivo 14-17 → 18-23")
fig.tight_layout(); fig.savefig(FIG / "pred_gini.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(evr) + 1), evr * 100, color="#55a868", alpha=0.7, label="Individual")
ax.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100, marker="o", color="#c44e52", label="Acumulada")
ax.set_xlabel("Componente"); ax.set_ylabel("% varianza"); ax.set_title("PCA: varianza explicada"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "pred_pca_scree.png", dpi=130); plt.close(fig)

with open(BASE / "analisis/resultados_predictivo.json", "w") as f:
    json.dump(RES, f, indent=2, ensure_ascii=False)

# ── Print ─────────────────────────────────────────────────────────────────────
print("TARGET"); [print(f"  {k}: {v}") for k, v in RES["target"].items()]
print("\nMODELO"); [print(f"  {k}: {v}") for k, v in RES["modelo"].items()]
cr = RES["classification_report"]["Con asalto"]
print(f"  recall(asalto)={cr['recall']:.3f}  precision={cr['precision']:.3f}")
print("\nGINI (top 14)"); [print(f"  {k:18s} {v}") for k, v in list(RES["gini"].items())[:14]]
print("\nPERMUTACION (top 14)"); [print(f"  {k:18s} {v}") for k, v in list(RES["perm"].items())[:14]]
print("\nPCA evr:", RES["pca_evr"][:6], "| cum:", RES["pca_cum"][:6])
print("PCA corr target:", RES["pca_corr_target"])
print("PC1 loadings:");
for k, v in load["PC1"].sort_values(ascending=False).items(): print(f"  {k:18s} {v:+.3f}")
print("\nTASAS  sexo:", RES["tasa_sexo"]); print("  raza:", RES["tasa_raza"]); print("  familia:", RES["tasa_familia"])
print("  persistencia (asalto temprano):", RES["tasa_por_early_assault"])
print("\nCOBERTURA predictores (%):"); [print(f"  {k:18s} {v}") for k, v in RES["cobertura_predictores_%"].items()]
print("\nOK -> analisis/resultados_predictivo.json")
