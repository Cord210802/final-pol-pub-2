"""
Análisis jerárquico de la agresión adulta (18-23) sobre la NLSY97.

Diseño:
  - 7 regresiones logísticas anidadas, ordenadas de más exógeno a más proximal:
      B1 Demográficas, B2 Familia, B3 Recursos económicos, B4 Cognitivo,
      B5 Conducta antisocial 14-17, B6 Sustancias 14-17, B7 Justicia 14-17.
  - Comparación entre modelos: deviance / Likelihood Ratio test (ANOVA logística).
  - Random Forest:
      * Importancia por permutación a nivel INDIVIDUAL.
      * Importancia por permutación a nivel BLOQUE (permuta todas las
        variables del bloque a la vez para neutralizar la multicolinealidad
        intra-bloque).

Salidas:
  analisis/resultados_jerarquico.json
  analisis/figuras/jer_coef_stability.png
  analisis/figuras/jer_perm_individual.png
  analisis/figuras/jer_perm_bloques.png
  analisis/figuras/jer_ranking_logit_vs_rf.png
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

import statsmodels.api as sm
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sns.set_palette("muted")
plt.style.use("seaborn-v0_8-whitegrid")

BASE = Path("/Users/jeronimo.deli/Desktop/other/Vs/Topicos/final-pol-pub-2")
FIG = BASE / "analisis/figuras"
FIG.mkdir(parents=True, exist_ok=True)
RES: dict = {}


# ── Lectura y limpieza ───────────────────────────────────────────────────────
def clean_nlsy(df):
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].where(df[num] >= 0, np.nan)
    return df


print("Leyendo NLSY97...")
ds1 = clean_nlsy(pd.read_stata(
    BASE / "data/ICPSR_34562/DS0001/34562-0001-Data.dta",
    convert_categoricals=False))

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


# ── Target ───────────────────────────────────────────────────────────────────
late_assault = ecols("ASSAULT_CY", LATE)
obs_late = ds1[late_assault].notna().any(axis=1)
ds = ds1[obs_late].copy()
ds["ASSAULT_18_23"] = ever(ds, late_assault)
ds = ds[ds["ASSAULT_18_23"].notna()].copy()
y = ds["ASSAULT_18_23"].astype(int)

RES["target"] = {
    "n_modelable": int(len(ds)),
    "prevalencia_%": round(float(y.mean() * 100), 2),
    "n_pos": int(y.sum()),
    "n_neg": int((y == 0).sum()),
}
print(f"  n = {len(ds)}, prevalencia = {y.mean()*100:.2f}%")


# ── Construcción de features y bloques ────────────────────────────────────────
def q_dummies(s, name, q=4):
    """Cuartiles a dummies. Q4 (más alto) = referencia (no se incluye)."""
    s = s.copy()
    edges = s.quantile(np.linspace(0, 1, q + 1)).values
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = [f"{name}_Q{i}" for i in range(1, q + 1)]
    cats = pd.cut(s, bins=edges, labels=labels, include_lowest=True)
    cats = cats.astype(object)  # evita categorical en join
    out = pd.get_dummies(cats).astype(float)
    out = out.drop(columns=[f"{name}_Q{q}"], errors="ignore")
    # columnas como strings simples (no Index categorical)
    out.columns = [str(c) for c in out.columns]
    return out


def cat_dummies(s, name, ref):
    """Dummies de categórica, ref = nivel de referencia (se elimina)."""
    s = s.copy().astype(object)
    out = pd.get_dummies(s, prefix=name).astype(float)
    refcol = f"{name}_{ref}"
    if refcol in out.columns:
        out = out.drop(columns=[refcol])
    out.columns = [str(c) for c in out.columns]
    return out


X = pd.DataFrame(index=ds.index)

# B1 - Demográficas
X["SEX"] = (ds["SEX"] == 1).astype(float)
race = cat_dummies(ds["RACE_ETHNICITY"], "RACE", ref=4.0)  # ref blanco/otro = 4
X = X.join(race)
X["BIOMOMAGE"] = ds["BIOMOMAGE"]
X["NUMSIBS"] = ds["NUMSIBS"]
B1 = ["SEX"] + list(race.columns) + ["BIOMOMAGE", "NUMSIBS"]

# B2 - Familia y capital social
fmstrc = cat_dummies(ds["FMSTRC97"], "FMSTRC", ref=1.0)
respared = cat_dummies(ds["RESPARED"], "RESPARED", ref=1.0)
X = X.join(fmstrc).join(respared)
B2 = list(fmstrc.columns) + list(respared.columns)

# B3 - Recursos económicos
pov_q = q_dummies(ds["POVRATIO97"], "POVRATIO")
ds["HHINC_EARLY"] = mean_pos(ds, ecols("HHINC_CY", EARLY))
hh_q = q_dummies(ds["HHINC_EARLY"], "HHINC_EARLY")
X = X.join(pov_q).join(hh_q)
B3 = list(pov_q.columns) + list(hh_q.columns)

# B4 - Cognitivo
asvab_q = q_dummies(ds["ASVAB_SCORE"], "ASVAB")
X = X.join(asvab_q)
B4 = list(asvab_q.columns)

# B5 - Conducta antisocial 14-17
X["EARLY_ASSAULT"] = ever(ds, ecols("ASSAULT_CY", EARLY))
X["EARLY_DESTROY"] = ever(ds, ecols("DESTROY_CY", EARLY))
X["EARLY_STEAL_LO"] = ever(ds, ecols("STEAL_UNDER_50_CY", EARLY))
X["EARLY_STEAL_HI"] = ever(ds, ecols("STEAL_OVER_50_CY", EARLY))
X["EARLY_SELL_DRUGS"] = ever(ds, ecols("SELL_DRUGS_CY", EARLY))
B5 = ["EARLY_ASSAULT", "EARLY_DESTROY", "EARLY_STEAL_LO",
      "EARLY_STEAL_HI", "EARLY_SELL_DRUGS"]

# B6 - Sustancias y riesgo 14-17
X["EARLY_ALCOHOL"] = ever(ds, ecols("ALCOHOL_CY", EARLY))
X["EARLY_MARIJUANA"] = ever(ds, ecols("MARIJUANA_CY", EARLY))
X["EARLY_HARD_DRUGS"] = ever(ds, ecols("HARD_DRUGS_CY", EARLY))
X["EARLY_SMOKING"] = ever(ds, ecols("SMOKING_CY", EARLY))
X["EARLY_CARR_GUN"] = ever(ds, ecols("CARR_GUN_CY", EARLY))
X["EARLY_GANG"] = ever(ds, ecols("GANGS_CY", EARLY))
B6 = ["EARLY_ALCOHOL", "EARLY_MARIJUANA", "EARLY_HARD_DRUGS",
      "EARLY_SMOKING", "EARLY_CARR_GUN", "EARLY_GANG"]

# B7 - Sistema de justicia 14-17
X["ARR_BY17"] = (ds["EVER_ARR_CY17"] > 0).astype(float)
X["INCARC_BY17"] = (ds["EVER_INCARC_CY17"] > 0).astype(float)
B7 = ["ARR_BY17", "INCARC_BY17"]

BLOCKS = {
    "B1 Demograficas": B1,
    "B2 Familia": B2,
    "B3 Recursos economicos": B3,
    "B4 Cognitivo": B4,
    "B5 Conducta antisocial 14-17": B5,
    "B6 Sustancias 14-17": B6,
    "B7 Justicia 14-17": B7,
}

all_features = sum(BLOCKS.values(), [])
X = X[all_features].copy()
print(f"  Features totales: {X.shape[1]}")
print(f"  Tamaños por bloque: " +
      ", ".join(f"{k.split()[0]}={len(v)}" for k, v in BLOCKS.items()))

# Imputación mediana
imp = SimpleImputer(strategy="median")
Xi = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

RES["bloques"] = {k: v for k, v in BLOCKS.items()}
RES["n_features_total"] = int(Xi.shape[1])


# ── 7 regresiones logísticas anidadas ────────────────────────────────────────
print("\nEstimando 7 regresiones logísticas anidadas...")
models = {}
coefs_by_m = {}
cumulative = []
ll_null = None
for i, (name, cols) in enumerate(BLOCKS.items(), start=1):
    cumulative += cols
    Xm = sm.add_constant(Xi[cumulative].astype(float))
    res = sm.Logit(y, Xm).fit(disp=0, maxiter=200)
    if ll_null is None:
        ll_null = sm.Logit(y, sm.add_constant(pd.Series(1, index=y.index))).fit(disp=0).llf
    models[f"M{i}"] = res
    coefs_by_m[f"M{i}"] = res.params.to_dict()
    print(f"  M{i} ({name}): k={int(res.df_model)}, "
          f"logL={res.llf:.1f}, AIC={res.aic:.1f}")

# Tabla ANOVA / Deviance
print("\nANOVA / Likelihood Ratio Tests entre modelos consecutivos:")
anova_rows = []
prev = None
for i, (name, _) in enumerate(BLOCKS.items(), start=1):
    m = models[f"M{i}"]
    dev = -2 * m.llf
    aic = m.aic
    bic = m.bic
    r2_mcfadden = 1 - m.llf / ll_null
    r2_nagel = (1 - np.exp((2 / len(y)) * (ll_null - m.llf))) / \
               (1 - np.exp(ll_null * 2 / len(y)))
    if prev is None:
        lr = np.nan; dfd = np.nan; p = np.nan; ddev = np.nan
    else:
        lr = 2 * (m.llf - prev.llf)
        dfd = int(m.df_model - prev.df_model)
        p = 1 - stats.chi2.cdf(lr, dfd) if dfd > 0 else np.nan
        ddev = -lr  # cambio en deviance
    anova_rows.append({
        "Modelo": f"M{i}: + {name}",
        "k": int(m.df_model),
        "Deviance": round(float(dev), 1),
        "LR_chi2": round(float(lr), 2) if not np.isnan(lr) else None,
        "df_change": dfd if not np.isnan(lr) else None,
        "p_value": round(float(p), 5) if not np.isnan(p) else None,
        "PseudoR2_McFadden": round(float(r2_mcfadden), 4),
        "PseudoR2_Nagelkerke": round(float(r2_nagel), 4),
        "AIC": round(float(aic), 1),
        "BIC": round(float(bic), 1),
    })
    print(f"  M{i}: dev={dev:.1f}, AIC={aic:.1f}, "
          f"R2_McF={r2_mcfadden:.3f}, R2_Nag={r2_nagel:.3f}" +
          (f"  | LR vs M{i-1} = {lr:.1f} (df={dfd}, p={p:.4g})"
           if prev is not None else ""))
    prev = m

RES["anova"] = anova_rows

# Aporte incremental al pseudo-R² Nagelkerke por bloque
r2_seq = [0.0] + [r["PseudoR2_Nagelkerke"] for r in anova_rows]
delta_r2 = [r2_seq[i + 1] - r2_seq[i] for i in range(len(r2_seq) - 1)]
RES["delta_r2_nagelkerke"] = {
    list(BLOCKS.keys())[i]: round(float(delta_r2[i]), 4)
    for i in range(len(delta_r2))
}

# Coefs del modelo final M7 con SE robustos
m7 = models["M7"]
m7_robust = sm.Logit(y, sm.add_constant(Xi[sum(BLOCKS.values(), [])].astype(float))).fit(
    disp=0, cov_type="HC1", maxiter=200)
coef_table = pd.DataFrame({
    "coef": m7_robust.params,
    "se": m7_robust.bse,
    "z": m7_robust.tvalues,
    "p": m7_robust.pvalues,
    "or": np.exp(m7_robust.params),
}).round(4)
RES["coefs_M7"] = coef_table.to_dict(orient="index")


# ── Random Forest ────────────────────────────────────────────────────────────
print("\nEntrenando Random Forest...")
X_tr, X_te, y_tr, y_te = train_test_split(
    Xi, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
auc_te = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
cv = cross_val_score(rf, Xi, y, cv=5, scoring="roc_auc", n_jobs=-1)
RES["rf"] = {
    "auc_test": round(float(auc_te), 4),
    "auc_cv_mean": round(float(cv.mean()), 4),
    "auc_cv_std": round(float(cv.std()), 4),
}
print(f"  AUC test = {auc_te:.3f}, AUC CV = {cv.mean():.3f} (±{cv.std():.3f})")


# Permutation importance INDIVIDUAL
print("\nPermutation importance INDIVIDUAL (n_repeats=30) ...")
rng = np.random.RandomState(42)
auc_base = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
perm_ind = {}
for col in X_te.columns:
    diffs = []
    for _ in range(30):
        Xp = X_te.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        diffs.append(auc_base - roc_auc_score(y_te, rf.predict_proba(Xp)[:, 1]))
    perm_ind[col] = {
        "mean": float(np.mean(diffs)),
        "std": float(np.std(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
RES["perm_individual"] = {k: {kk: round(vv, 5) for kk, vv in v.items()}
                          for k, v in perm_ind.items()}


# Permutation importance POR BLOQUE
print("\nPermutation importance POR BLOQUE (n_repeats=30) ...")
perm_blk = {}
for bname, cols in BLOCKS.items():
    diffs = []
    for _ in range(30):
        Xp = X_te.copy()
        for col in cols:
            Xp[col] = rng.permutation(Xp[col].values)
        diffs.append(auc_base - roc_auc_score(y_te, rf.predict_proba(Xp)[:, 1]))
    perm_blk[bname] = {
        "mean": float(np.mean(diffs)),
        "std": float(np.std(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
    print(f"  {bname}: ΔAUC = {perm_blk[bname]['mean']:.4f} "
          f"[{perm_blk[bname]['ci_low']:.4f}, {perm_blk[bname]['ci_high']:.4f}]")
RES["perm_bloques"] = {k: {kk: round(vv, 5) for kk, vv in v.items()}
                      for k, v in perm_blk.items()}


# Comparación de rankings: ΔR² (logit) vs ΔAUC (RF)
print("\nComparación de rankings entre métodos:")
logit_delta = {list(BLOCKS.keys())[i]: delta_r2[i] for i in range(len(delta_r2))}
rf_delta = {k: v["mean"] for k, v in perm_blk.items()}
logit_rank = pd.Series(logit_delta).rank(ascending=False).astype(int).to_dict()
rf_rank = pd.Series(rf_delta).rank(ascending=False).astype(int).to_dict()
RES["ranking_logit"] = logit_rank
RES["ranking_rf"] = rf_rank
for b in BLOCKS:
    print(f"  {b:35s}  ΔR²={logit_delta[b]:.4f} (rk {logit_rank[b]})  "
          f"ΔAUC={rf_delta[b]:.4f} (rk {rf_rank[b]})")


# ── Figuras ──────────────────────────────────────────────────────────────────
print("\nGenerando figuras...")

# Fig 1: Coefficient stability plot (top variables del M7)
top_vars = (coef_table["coef"].abs().sort_values(ascending=False)
            .index.drop("const", errors="ignore").tolist()[:8])
stab = {v: [] for v in top_vars}
for i in range(1, 8):
    p = models[f"M{i}"].params
    for v in top_vars:
        stab[v].append(p.get(v, np.nan))

fig, ax = plt.subplots(figsize=(9, 6))
for v in top_vars:
    ax.plot(range(1, 8), stab[v], marker="o", label=v, linewidth=1.5)
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_xlabel("Modelo (acumulado)")
ax.set_ylabel("Coeficiente (log-odds)")
ax.set_title("Estabilidad de coeficientes top al agregar bloques")
ax.set_xticks(range(1, 8))
ax.set_xticklabels([f"M{i}" for i in range(1, 8)])
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "jer_coef_stability.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# Fig 2: Permutation importance individual (top 15)
ind_df = pd.DataFrame(perm_ind).T.sort_values("mean", ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(range(len(ind_df)), ind_df["mean"],
        xerr=[ind_df["mean"] - ind_df["ci_low"], ind_df["ci_high"] - ind_df["mean"]],
        color="#4c72b0", alpha=0.85, ecolor="gray", capsize=2)
ax.set_yticks(range(len(ind_df)))
ax.set_yticklabels(ind_df.index, fontsize=9)
ax.set_xlabel(r"$\Delta$ AUC al permutar la variable")
ax.set_title("Random Forest — importancia por permutación (individual, top 15)")
fig.tight_layout()
fig.savefig(FIG / "jer_perm_individual.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# Fig 3: Permutation importance por bloque
blk_df = pd.DataFrame(perm_blk).T.sort_values("mean", ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(len(blk_df)), blk_df["mean"],
        xerr=[blk_df["mean"] - blk_df["ci_low"], blk_df["ci_high"] - blk_df["mean"]],
        color="#c44e52", alpha=0.85, ecolor="gray", capsize=3)
ax.set_yticks(range(len(blk_df)))
ax.set_yticklabels(blk_df.index, fontsize=10)
ax.set_xlabel(r"$\Delta$ AUC al permutar todas las variables del bloque")
ax.set_title("Random Forest — importancia por permutación (por bloque)")
fig.tight_layout()
fig.savefig(FIG / "jer_perm_bloques.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# Fig 4: Ranking comparison (slope chart)
blks = list(BLOCKS.keys())
fig, ax = plt.subplots(figsize=(9, 5))
for b in blks:
    ax.plot([0, 1], [logit_rank[b], rf_rank[b]], marker="o", linewidth=2)
    ax.text(-0.04, logit_rank[b], b, ha="right", va="center", fontsize=9)
    ax.text(1.04, rf_rank[b], b, ha="left", va="center", fontsize=9)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Logística jerárquica\n(rank ΔR² Nagelkerke)",
                    "Random Forest\n(rank ΔAUC perm. bloque)"])
ax.invert_yaxis()
ax.set_yticks(range(1, len(blks) + 1))
ax.set_ylabel("Ranking de aporte del bloque (1 = mayor)")
ax.set_title("Triangulación: ranking de bloques en ambos métodos")
ax.set_xlim(-0.6, 1.6)
fig.tight_layout()
fig.savefig(FIG / "jer_ranking_logit_vs_rf.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ── Guardar JSON ─────────────────────────────────────────────────────────────
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if not np.isnan(o) else None
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


out = BASE / "analisis/resultados_jerarquico.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(_clean(RES), f, indent=2, ensure_ascii=False)

print(f"\nOK. Resultados en {out}")
print("Figuras:")
for f in ["jer_coef_stability.png", "jer_perm_individual.png",
          "jer_perm_bloques.png", "jer_ranking_logit_vs_rf.png"]:
    print(f"  {FIG / f}")
