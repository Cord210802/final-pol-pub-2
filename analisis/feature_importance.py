"""
Análisis de Feature Importance + PCA sobre ICPSR 34562 (Recidivism NLSY97).
Target: EVER_ASSAULT (1 = autoreportó >=1 agresión entre los 14 y 25 años).
Genera números y figuras para revisar antes de redactar el paper en Quarto.
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
DS1_PATH = BASE / "data/ICPSR_34562/DS0001/34562-0001-Data.dta"
DS2_PATH = BASE / "data/ICPSR_34562/DS0002/34562-0002-Data.dta"
FIG = BASE / "analisis/figuras"
RESULTS = {}


def clean_nlsy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].where(df[num] >= 0, np.nan)
    return df


# ── Carga ────────────────────────────────────────────────────────────────────
ds1 = clean_nlsy(pd.read_stata(DS1_PATH, convert_categoricals=False))
ds2 = clean_nlsy(pd.read_stata(DS2_PATH, convert_categoricals=False))
df = ds1.merge(ds2[["PUBID", "ASSAULT_CY24", "ASSAULT_CY25"]], on="PUBID", how="left")

AGES = list(range(14, 26))


def cols(prefix):
    return [f"{prefix}{a}" for a in AGES if f"{prefix}{a}" in df.columns]


assault_cols = cols("ASSAULT_CY")
destroy_cols = cols("DESTROY_CY")
steal_lo_cols = cols("STEAL_UNDER_50_CY")
steal_hi_cols = cols("STEAL_OVER_50_CY")
drugs_sell_cols = cols("SELL_DRUGS_CY")
gun_cols = cols("CARR_GUN_CY")
gang_cols = cols("GANGS_CY")
alcohol_cols = cols("ALCOHOL_CY")
marijuana_cols = cols("MARIJUANA_CY")
hard_cols = cols("HARD_DRUGS_CY")
smoking_cols = cols("SMOKING_CY")
hhinc_cols = cols("HHINC_CY")
rinc_cols = cols("RINC_CY")

# ── Target ────────────────────────────────────────────────────────────────────
df["TOTAL_ASSAULTS"] = df[assault_cols].sum(axis=1, min_count=1)
df["EVER_ASSAULT"] = (df["TOTAL_ASSAULTS"] > 0).astype(float)
df.loc[df["TOTAL_ASSAULTS"].isna(), "EVER_ASSAULT"] = np.nan

has_obs = df[assault_cols].notna().any(axis=1)
dfa = df[has_obs].copy()

n_total = len(df)
n_obs = int(has_obs.sum())
vc = dfa["EVER_ASSAULT"].value_counts()
RESULTS["target"] = {
    "n_total": n_total,
    "n_con_observacion_assault": n_obs,
    "n_sin_asalto": int(vc.get(0.0, 0)),
    "n_con_asalto": int(vc.get(1.0, 0)),
    "prevalencia_asalto_%": round(dfa["EVER_ASSAULT"].mean() * 100, 2),
}

# ── Features agregados (sin leakage de assault) ───────────────────────────────
def safe_sum(d, c):
    return d[c].sum(axis=1, min_count=1)


agg = pd.DataFrame(index=dfa.index)
agg["TOTAL_DESTROY"] = safe_sum(dfa, destroy_cols)
agg["TOTAL_STEAL_LO"] = safe_sum(dfa, steal_lo_cols)
agg["TOTAL_STEAL_HI"] = safe_sum(dfa, steal_hi_cols)
agg["TOTAL_SELL_DRUGS"] = safe_sum(dfa, drugs_sell_cols)
agg["TOTAL_CARR_GUN"] = safe_sum(dfa, gun_cols)
agg["TOTAL_GANG"] = safe_sum(dfa, gang_cols)
agg["TOTAL_ALCOHOL"] = safe_sum(dfa, alcohol_cols)
agg["TOTAL_MARIJUANA"] = safe_sum(dfa, marijuana_cols)
agg["TOTAL_HARD_DRUGS"] = safe_sum(dfa, hard_cols)
agg["TOTAL_SMOKING"] = safe_sum(dfa, smoking_cols)
agg["TOTAL_ARRESTS"] = dfa["TOTARRESTS"]
agg["AGE_FIRST_ARREST"] = dfa["FIRSTARREST"]
agg["TOTAL_INCARC"] = dfa["TOTINCARC"]
agg["AVG_HHINC"] = dfa[hhinc_cols].clip(lower=0).mean(axis=1)
agg["AVG_RINC"] = dfa[rinc_cols].clip(lower=0).mean(axis=1)
agg["ASVAB_SCORE"] = dfa["ASVAB_SCORE"]
agg["POVRATIO97"] = dfa["POVRATIO97"]
agg["NUMSIBS"] = dfa["NUMSIBS"]
agg["BIOMOMAGE"] = dfa["BIOMOMAGE"]
agg["SEX"] = dfa["SEX"]
agg["RACE_ETHNICITY"] = dfa["RACE_ETHNICITY"]
agg["FMSTRC97"] = dfa["FMSTRC97"]
agg["RESPARED"] = dfa["RESPARED"]
agg["MOMEMP97"] = dfa["MOMEMP97"]
agg["DADEMP97"] = dfa["DADEMP97"]

feature_cols = list(agg.columns)
y = dfa["EVER_ASSAULT"]
mask = y.notna()
X_raw = agg[mask]
y = y[mask].astype(int)

imp = SimpleImputer(strategy="median")
X = pd.DataFrame(imp.fit_transform(X_raw), columns=feature_cols, index=X_raw.index)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Random Forest ─────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=15,
    max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(X_tr, y_tr)
y_prob = rf.predict_proba(X_te)[:, 1]
y_pred = rf.predict(X_te)
auc = roc_auc_score(y_te, y_prob)
cv_auc = cross_val_score(rf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)

RESULTS["modelo"] = {
    "n_train": len(X_tr), "n_test": len(X_te), "n_features": len(feature_cols),
    "auc_test": round(float(auc), 4),
    "auc_cv_mean": round(float(cv_auc.mean()), 4),
    "auc_cv_std": round(float(cv_auc.std()), 4),
}
RESULTS["classification_report"] = classification_report(
    y_te, y_pred, target_names=["Sin asalto", "Con asalto"], output_dict=True
)

# ── Gini importance ───────────────────────────────────────────────────────────
gini = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
RESULTS["gini_importance"] = {k: round(float(v), 4) for k, v in gini.items()}

# ── Permutation importance (robustez) ─────────────────────────────────────────
perm = permutation_importance(
    rf, X_te, y_te, n_repeats=20, random_state=42, scoring="roc_auc", n_jobs=-1
)
perm_s = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
RESULTS["perm_importance"] = {k: round(float(v), 5) for k, v in perm_s.items()}

# ── PCA ───────────────────────────────────────────────────────────────────────
num_for_pca = [
    "TOTAL_DESTROY", "TOTAL_STEAL_LO", "TOTAL_STEAL_HI", "TOTAL_SELL_DRUGS",
    "TOTAL_CARR_GUN", "TOTAL_GANG", "TOTAL_ALCOHOL", "TOTAL_MARIJUANA",
    "TOTAL_HARD_DRUGS", "TOTAL_SMOKING", "TOTAL_ARRESTS", "TOTAL_INCARC",
    "AVG_HHINC", "AVG_RINC", "ASVAB_SCORE", "POVRATIO97", "NUMSIBS", "BIOMOMAGE",
]
Xp = StandardScaler().fit_transform(X[num_for_pca])
pca = PCA().fit(Xp)
evr = pca.explained_variance_ratio_
RESULTS["pca_explained_variance_ratio"] = [round(float(v), 4) for v in evr[:10]]
RESULTS["pca_cumulative"] = [round(float(v), 4) for v in np.cumsum(evr)[:10]]

loadings = pd.DataFrame(
    pca.components_[:4].T, index=num_for_pca,
    columns=["PC1", "PC2", "PC3", "PC4"],
)
RESULTS["pca_loadings"] = loadings.round(3).to_dict()

# Correlación de las primeras PCs con el target
scores = pca.transform(Xp)
pc_corr = {}
for i in range(4):
    pc_corr[f"PC{i+1}"] = round(float(np.corrcoef(scores[:, i], y)[0, 1]), 4)
RESULTS["pca_corr_target"] = pc_corr

# ── Contrastes bivariados (tasa de asalto por grupo) ──────────────────────────
def rate_by(col, labels):
    r = dfa.groupby(col)["EVER_ASSAULT"].mean() * 100
    return {labels.get(k, str(k)): round(float(v), 1) for k, v in r.items()}


RESULTS["tasa_por_sexo"] = rate_by("SEX", {1: "Hombre", 2: "Mujer"})
RESULTS["tasa_por_raza"] = rate_by(
    "RACE_ETHNICITY", {1: "Negro", 2: "Hispano", 3: "Mixto", 4: "Blanco/Otro"}
)
RESULTS["tasa_por_familia"] = rate_by(
    "FMSTRC97", {1: "Ambos biologicos", 2: "Un padre soltero", 3: "Uno bio + padrastro",
                 4: "Dos adoptivos", 5: "Otro"}
)

# Medias de variables continuas por grupo de target
cont = ["ASVAB_SCORE", "POVRATIO97", "NUMSIBS", "BIOMOMAGE", "TOTARRESTS"]
means = dfa.groupby("EVER_ASSAULT")[cont].mean()
RESULTS["medias_por_target"] = means.round(2).to_dict()

# ── Figuras ───────────────────────────────────────────────────────────────────
# Gini
fig, ax = plt.subplots(figsize=(9, 8))
g = gini.head(20)[::-1]
ax.barh(range(len(g)), g.values, color="#4c72b0")
ax.set_yticks(range(len(g)))
ax.set_yticklabels(g.index, fontsize=9)
ax.set_xlabel("Importancia de Gini")
ax.set_title("Importancia de variables (Random Forest, Gini)")
fig.tight_layout()
fig.savefig(FIG / "gini_importance.png", dpi=130)
plt.close(fig)

# PCA scree
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(evr) + 1), evr * 100, color="#55a868", alpha=0.7, label="Individual")
ax.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100, marker="o", color="#c44e52", label="Acumulada")
ax.set_xlabel("Componente principal")
ax.set_ylabel("% varianza explicada")
ax.set_title("PCA: varianza explicada")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "pca_scree.png", dpi=130)
plt.close(fig)

# PCA biplot de loadings PC1 vs PC2
fig, ax = plt.subplots(figsize=(9, 8))
for i, v in enumerate(num_for_pca):
    ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
             head_width=0.02, color="#8172b2", alpha=0.7)
    ax.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1, v, fontsize=8)
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
ax.set_title("PCA: cargas de las variables (PC1 vs PC2)")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
fig.tight_layout()
fig.savefig(FIG / "pca_biplot.png", dpi=130)
plt.close(fig)

# ── Guardar resultados ────────────────────────────────────────────────────────
out = BASE / "analisis/resultados.json"
with open(out, "w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)

# ── Print legible ─────────────────────────────────────────────────────────────
print("=" * 70)
print("TARGET")
for k, v in RESULTS["target"].items():
    print(f"  {k}: {v}")
print("\nMODELO RF")
for k, v in RESULTS["modelo"].items():
    print(f"  {k}: {v}")
cr = RESULTS["classification_report"]
print(f"  recall(con asalto): {cr['Con asalto']['recall']:.3f} | precision: {cr['Con asalto']['precision']:.3f}")
print("\nGINI IMPORTANCE (top 12)")
for k, v in list(RESULTS["gini_importance"].items())[:12]:
    print(f"  {k:20s} {v}")
print("\nPERMUTATION IMPORTANCE (top 12)")
for k, v in list(RESULTS["perm_importance"].items())[:12]:
    print(f"  {k:20s} {v}")
print("\nPCA varianza explicada (primeras 6):", RESULTS["pca_explained_variance_ratio"][:6])
print("PCA acumulada (primeras 6):", RESULTS["pca_cumulative"][:6])
print("PCA corr con target:", RESULTS["pca_corr_target"])
print("\nPC1 loadings (orden desc):")
pc1 = loadings["PC1"].sort_values(ascending=False)
for k, v in pc1.items():
    print(f"  {k:20s} {v:+.3f}")
print("\nTASA DE ASALTO POR GRUPO")
print("  sexo:", RESULTS["tasa_por_sexo"])
print("  raza:", RESULTS["tasa_por_raza"])
print("  familia:", RESULTS["tasa_por_familia"])
print("\nMEDIAS POR TARGET:", json.dumps(RESULTS["medias_por_target"], ensure_ascii=False))
print("=" * 70)
print(f"Resultados guardados en {out}")
