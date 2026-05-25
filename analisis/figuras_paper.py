"""Figuras descriptivas adicionales para el paper."""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_palette("muted")
plt.style.use("seaborn-v0_8-whitegrid")
BASE = Path("/home/cord2108/ITAM/Semestre_9/proyecto_final_polpub")
FIG = BASE / "analisis/figuras"


def clean(d):
    d = d.copy(); n = d.select_dtypes("number").columns; d[n] = d[n].where(d[n] >= 0, np.nan); return d


ds1 = clean(pd.read_stata(BASE / "data/ICPSR_34562/DS0001/34562-0001-Data.dta", convert_categoricals=False))

# ── 1. Curva edad-crimen (prevalencia de asalto autoreportado por edad) ───────
ages = range(14, 24)
prev, ns = [], []
for a in ages:
    c = f"ASSAULT_CY{a}"
    s = ds1[c].dropna()
    prev.append((s > 0).mean() * 100)
    ns.append(len(s))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(ages), prev, marker="o", color="#c44e52", linewidth=2)
ax.set_xlabel("Edad")
ax.set_ylabel("% que reporta haber agredido ese año")
ax.set_title("Curva edad-agresión (NLSY97, autoreporte)")
ax.set_xticks(list(ages))
fig.tight_layout()
fig.savefig(FIG / "curva_edad_crimen.png", dpi=130)
plt.close(fig)

# ── 2. Biplot PCA del modelo predictivo (variables 14-17) ─────────────────────
EARLY = range(14, 18)


def ecols(p, ages):
    return [f"{p}{a}" for a in ages if f"{p}{a}" in ds1.columns]


def ever(d, cols):
    sub = d[cols]; out = (sub.max(axis=1) > 0).astype(float)
    out[sub.notna().sum(axis=1) == 0] = np.nan; return out


LATE = range(18, 24)
ds = ds1[ds1[ecols("ASSAULT_CY", LATE)].notna().any(axis=1)].copy()
feats = {
    "EARLY_ASSAULT": ever(ds, ecols("ASSAULT_CY", EARLY)),
    "EARLY_DESTROY": ever(ds, ecols("DESTROY_CY", EARLY)),
    "EARLY_STEAL_LO": ever(ds, ecols("STEAL_UNDER_50_CY", EARLY)),
    "EARLY_STEAL_HI": ever(ds, ecols("STEAL_OVER_50_CY", EARLY)),
    "EARLY_SELL_DRUGS": ever(ds, ecols("SELL_DRUGS_CY", EARLY)),
    "EARLY_CARR_GUN": ever(ds, ecols("CARR_GUN_CY", EARLY)),
    "EARLY_GANG": ever(ds, ecols("GANGS_CY", EARLY)),
    "EARLY_ALCOHOL": ever(ds, ecols("ALCOHOL_CY", EARLY)),
    "EARLY_MARIJUANA": ever(ds, ecols("MARIJUANA_CY", EARLY)),
    "EARLY_HARD_DRUGS": ever(ds, ecols("HARD_DRUGS_CY", EARLY)),
    "EARLY_SMOKING": ever(ds, ecols("SMOKING_CY", EARLY)),
    "ARR_BY17": (ds["EVER_ARR_CY17"] > 0).astype(float),
    "INCARC_BY17": (ds["EVER_INCARC_CY17"] > 0).astype(float),
    "HHINC_EARLY": ds[ecols("HHINC_CY", EARLY)].clip(lower=0).mean(axis=1),
    "ASVAB_SCORE": ds["ASVAB_SCORE"], "POVRATIO97": ds["POVRATIO97"],
    "NUMSIBS": ds["NUMSIBS"], "BIOMOMAGE": ds["BIOMOMAGE"],
}
Xp = pd.DataFrame(feats)
Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xp), columns=Xp.columns)
Z = StandardScaler().fit_transform(Xi)
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(9, 8))
for i, v in enumerate(Xp.columns):
    ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.02, color="#8172b2", alpha=0.7)
    ax.text(pca.components_[0, i] * 1.08, pca.components_[1, i] * 1.08, v, fontsize=8)
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
ax.set_title("PCA de factores de riesgo (14-17): cargas PC1 vs PC2")
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
fig.tight_layout()
fig.savefig(FIG / "pred_pca_biplot.png", dpi=130)
plt.close(fig)

print("Figuras generadas: curva_edad_crimen.png, pred_pca_biplot.png")
