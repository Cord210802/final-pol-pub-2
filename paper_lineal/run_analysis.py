"""
Análisis lineal jerárquico (LPM) de EVER_ASSAULT con bloques incrementales.
Genera tablas (JSON/CSV) y figuras para el paper.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path("/Users/jeronimo.deli/Desktop/other/Vs/Topicos/final-pol-pub-2")
DATA = ROOT / "data/ICPSR_34562/DS0001/34562-0001-Data.dta"
OUT = ROOT / "paper_lineal"
FIGS = OUT / "figuras"
RES = OUT / "resultados"

SEED = 42
np.random.seed(SEED)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ------------------------------------------------------------------
# 1. Cargar y construir features
# ------------------------------------------------------------------
print("Loading data...")
df = pd.read_stata(DATA, convert_categoricals=True)
print(f"   shape raw: {df.shape}")


def to_yes_no_numeric(series):
    """Map 'Yes'->1, 'No'->0 (categorical or string); other -> NaN."""
    if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        out = pd.Series(np.nan, index=series.index)
        out[s.eq("yes")] = 1.0
        out[s.eq("no")] = 0.0
        return out
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s >= 0)


def clean_num(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s >= 0)


# Target ---------------------------------------------------------------
assault_cols = [c for c in df.columns if c.startswith("ASSAULT_CY")]
assault = pd.concat([to_yes_no_numeric(df[c]) for c in assault_cols], axis=1)
df["EVER_ASSAULT"] = (assault.sum(axis=1, min_count=1) > 0).astype("float")
# Keep only rows with at least one valid assault observation
df = df[assault.notna().any(axis=1)].copy()
df["EVER_ASSAULT"] = df["EVER_ASSAULT"].astype(int)


# Demographic block ---------------------------------------------------
df["FEMALE"] = (df["SEX"].astype(str).str.lower() == "female").astype(int)

race = df["RACE_ETHNICITY"].astype(str).str.strip().str.lower()
df["RACE_BLACK"] = (race == "black").astype(int)
df["RACE_HISPANIC"] = (race == "hispanic").astype(int)
df["RACE_MIXED"] = (race == "mixed race (non-hispanic)").astype(int)
# baseline = "non-black / non-hispanic"

df["BIOMOMAGE"] = clean_num(df["BIOMOMAGE"])
df["NUMSIBS"] = clean_num(df["NUMSIBS"])
df["BDATE_Y"] = clean_num(df["BDATE_Y"])

# Family / SES block --------------------------------------------------
fmstrc = df["FMSTRC97"].astype(str).str.strip().str.lower()
df["FAM_TWO_BIO"] = (fmstrc == "two biological parents").astype(int)  # ref
df["FAM_ONE_PARENT"] = (fmstrc == "one single biologic/adoptive parent").astype(int)
df["FAM_BIO_STEP"] = (fmstrc == "one biological- and one step-parent").astype(int)
# Colapsamos "other family type" y "two adoptive parent(s)" (raros) en FAM_OTHER
df["FAM_OTHER"] = fmstrc.isin(["other family type", "two adoptive parent(s)"]).astype(int)

resp_raw = df["RESPARED"].astype(str).str.strip().str.lower()
# baseline = "Zero through 11th grade of high school" (<HS).
LT_HS = "zero through 11th grade of high school"
HS = "graduated from hs"
SC = "some college"
COL = "college degree or higher"
valid_resp = resp_raw.isin([LT_HS, HS, SC, COL])
df["PAR_ED_HS"] = np.where(valid_resp, (resp_raw == HS).astype(int), np.nan)
df["PAR_ED_SOME_COL"] = np.where(valid_resp, (resp_raw == SC).astype(int), np.nan)
df["PAR_ED_COL"] = np.where(valid_resp, (resp_raw == COL).astype(int), np.nan)

df["POVRATIO97"] = clean_num(df["POVRATIO97"])

# Cognitive block -----------------------------------------------------
df["ASVAB_SCORE"] = clean_num(df["ASVAB_SCORE"])

# Past behavior block -------------------------------------------------
def sum_binary(prefix, ages):
    cols = [f"{prefix}_CY{a}" for a in ages if f"{prefix}_CY{a}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    mat = pd.concat([to_yes_no_numeric(df[c]) for c in cols], axis=1)
    return mat.sum(axis=1, min_count=1)


early_ages = list(range(14, 18))  # 14-17 = adolescencia temprana

df["EARLY_DESTROY"] = sum_binary("DESTROY", early_ages)
df["EARLY_STEAL_UNDER"] = sum_binary("STEAL_UNDER_50", early_ages)
df["EARLY_STEAL_OVER"] = sum_binary("STEAL_OVER_50", early_ages)
df["EARLY_OTHER_PROP"] = sum_binary("OTHER_PROPERTY", early_ages)
df["EARLY_GANG"] = sum_binary("GANGS", early_ages)
df["EARLY_GUN"] = sum_binary("CARR_GUN", early_ages)
df["EARLY_RUNAWAY"] = sum_binary("RUNAWAY", early_ages)

# Justice / Substances block -----------------------------------------
df["EARLY_SMOKING"] = sum_binary("SMOKING", early_ages)
df["EARLY_ALCOHOL"] = sum_binary("ALCOHOL", early_ages)
df["EARLY_MARIJUANA"] = sum_binary("MARIJUANA", early_ages)
df["EARLY_HARD_DRUGS"] = sum_binary("HARD_DRUGS", early_ages)

# Arrested before 18 (any year 14-17)
arr_cols = [f"EVER_ARR_CY{a}" for a in early_ages if f"EVER_ARR_CY{a}" in df.columns]
arr_mat = pd.concat([to_yes_no_numeric(df[c]) for c in arr_cols], axis=1)
df["EVER_ARR_BY17"] = (arr_mat.max(axis=1) > 0).astype(float)


# Build the analytical sample ----------------------------------------
# Requiere variables clave no nulas para que los 6 modelos corran sobre la
# misma muestra (comparabilidad).
core_vars = [
    "EVER_ASSAULT",
    "FEMALE", "RACE_BLACK", "RACE_HISPANIC", "RACE_MIXED",
    "BIOMOMAGE", "NUMSIBS",
    "FAM_ONE_PARENT", "FAM_BIO_STEP", "FAM_OTHER",
    "PAR_ED_HS", "PAR_ED_SOME_COL", "PAR_ED_COL",
    "POVRATIO97",
    "ASVAB_SCORE",
    "EARLY_DESTROY", "EARLY_STEAL_UNDER", "EARLY_STEAL_OVER",
    "EARLY_OTHER_PROP", "EARLY_GANG", "EARLY_GUN", "EARLY_RUNAWAY",
    "EARLY_SMOKING", "EARLY_ALCOHOL", "EARLY_MARIJUANA", "EARLY_HARD_DRUGS",
    "EVER_ARR_BY17",
]
analytic = df[core_vars].dropna().copy()
print(f"   analytic sample: {analytic.shape}")
analytic.to_csv(RES / "analytic_sample.csv", index=False)


# ------------------------------------------------------------------
# 2. Definición de bloques de modelos
# ------------------------------------------------------------------
BLOCKS = {
    "demographic": ["FEMALE", "RACE_BLACK", "RACE_HISPANIC", "RACE_MIXED",
                    "BIOMOMAGE", "NUMSIBS"],
    "family_ses": ["FAM_ONE_PARENT", "FAM_BIO_STEP", "FAM_OTHER",
                   "PAR_ED_HS", "PAR_ED_SOME_COL", "PAR_ED_COL",
                   "POVRATIO97"],
    "cognitive": ["ASVAB_SCORE"],
    "past_behavior": ["EARLY_DESTROY", "EARLY_STEAL_UNDER", "EARLY_STEAL_OVER",
                      "EARLY_OTHER_PROP", "EARLY_GANG", "EARLY_GUN",
                      "EARLY_RUNAWAY"],
    "justice_substance": ["EARLY_SMOKING", "EARLY_ALCOHOL", "EARLY_MARIJUANA",
                          "EARLY_HARD_DRUGS", "EVER_ARR_BY17"],
}

MODEL_SPECS = {
    "Modelo 1: Demográfico":
        BLOCKS["demographic"],
    "Modelo 2: + Familia y SES":
        BLOCKS["demographic"] + BLOCKS["family_ses"],
    "Modelo 3: + Cognitivo":
        BLOCKS["demographic"] + BLOCKS["family_ses"] + BLOCKS["cognitive"],
    "Modelo 4: + Conducta pasada":
        BLOCKS["demographic"] + BLOCKS["family_ses"] +
        BLOCKS["cognitive"] + BLOCKS["past_behavior"],
    "Modelo 5: + Justicia/Sustancias":
        BLOCKS["demographic"] + BLOCKS["family_ses"] +
        BLOCKS["cognitive"] + BLOCKS["past_behavior"] +
        BLOCKS["justice_substance"],
}


def fit_ols(y, X):
    X_ = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_).fit(cov_type="HC1")


y = analytic["EVER_ASSAULT"].astype(float)


# ------------------------------------------------------------------
# 3. Ajustar los 5 modelos jerárquicos + Modelo 6 (interacción)
# ------------------------------------------------------------------
fitted = {}
for name, cols in MODEL_SPECS.items():
    fitted[name] = fit_ols(y, analytic[cols])

# Modelo 6: añade interacción FEMALE × RACE_BLACK y FEMALE × RACE_HISPANIC
X6 = analytic[MODEL_SPECS["Modelo 5: + Justicia/Sustancias"]].copy()
X6["FEMALE_x_BLACK"] = X6["FEMALE"] * X6["RACE_BLACK"]
X6["FEMALE_x_HISPANIC"] = X6["FEMALE"] * X6["RACE_HISPANIC"]
fitted["Modelo 6: + Interacción sexo×raza"] = fit_ols(y, X6)


# ------------------------------------------------------------------
# 4. Exportar tabla de coeficientes y tabla de ajuste
# ------------------------------------------------------------------
def model_summary(res):
    return {
        "n": int(res.nobs),
        "k": int(res.df_model),
        "r2": float(res.rsquared),
        "r2_adj": float(res.rsquared_adj),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "ll": float(res.llf),
        "fstat": float(res.fvalue) if res.fvalue is not None else None,
        "f_pvalue": float(res.f_pvalue) if res.f_pvalue is not None else None,
    }


fit_table = {name: model_summary(res) for name, res in fitted.items()}
with open(RES / "fit_table.json", "w") as f:
    json.dump(fit_table, f, indent=2)


def coef_frame(res):
    out = pd.DataFrame({
        "coef": res.params,
        "se": res.bse,
        "t": res.tvalues,
        "p": res.pvalues,
        "ci_low": res.conf_int()[0],
        "ci_high": res.conf_int()[1],
    })
    return out


coef_tables = {}
for name, res in fitted.items():
    cf = coef_frame(res)
    cf.to_csv(RES / f"coefs__{name.split(':')[0].replace(' ', '_')}.csv")
    coef_tables[name] = cf

# Wide table with all 6 models side-by-side
all_terms = []
for cf in coef_tables.values():
    for t in cf.index:
        if t not in all_terms:
            all_terms.append(t)
wide = pd.DataFrame(index=all_terms)
for name, cf in coef_tables.items():
    short = name.split(":")[0]
    wide[f"{short}_coef"] = cf["coef"].reindex(all_terms)
    wide[f"{short}_se"] = cf["se"].reindex(all_terms)
    wide[f"{short}_p"] = cf["p"].reindex(all_terms)
wide.to_csv(RES / "coefs_wide.csv")


# ------------------------------------------------------------------
# 5. ANOVA entre modelos anidados (cambio en R^2, F incremental)
# ------------------------------------------------------------------
def incremental_f(res_small, res_big):
    rss_s = float(np.sum(res_small.resid ** 2))
    rss_b = float(np.sum(res_big.resid ** 2))
    df_diff = int(res_big.df_model - res_small.df_model)
    df_b = int(res_big.df_resid)
    num = (rss_s - rss_b) / df_diff
    den = rss_b / df_b
    f = num / den
    p = 1 - stats.f.cdf(f, df_diff, df_b)
    return {"df_diff": df_diff, "df_resid": df_b, "F": float(f), "p": float(p),
            "delta_r2": float(res_big.rsquared - res_small.rsquared),
            "rss_small": rss_s, "rss_big": rss_b}


anova_seq = []
names = list(fitted.keys())
for i in range(1, len(names)):
    info = incremental_f(fitted[names[i - 1]], fitted[names[i]])
    info["from"] = names[i - 1]
    info["to"] = names[i]
    anova_seq.append(info)
with open(RES / "anova_sequence.json", "w") as f:
    json.dump(anova_seq, f, indent=2)


# Clásico ANOVA univariado de EVER_ASSAULT por raza y por educación de los padres
def anova_oneway(group_var, label):
    groups = []
    labels = []
    for lvl, sub in analytic.groupby(group_var):
        groups.append(sub["EVER_ASSAULT"].values)
        labels.append(str(lvl))
    f, p = stats.f_oneway(*groups)
    means = {lab: float(g.mean()) for lab, g in zip(labels, groups)}
    return {"label": label, "F": float(f), "p": float(p),
            "means": means, "n_groups": len(groups)}


# Reconstruir variables categóricas en formato legible para los ANOVA
analytic["_race"] = np.where(
    analytic["RACE_BLACK"] == 1, "Black",
    np.where(analytic["RACE_HISPANIC"] == 1, "Hispanic",
             np.where(analytic["RACE_MIXED"] == 1, "Mixed",
                      "Non-Black/Non-Hispanic")))
analytic["_par_ed"] = np.where(
    analytic["PAR_ED_HS"] == 1, "HS",
    np.where(analytic["PAR_ED_SOME_COL"] == 1, "Some College",
             np.where(analytic["PAR_ED_COL"] == 1, "College+", "<HS")))
analytic["_sex"] = np.where(analytic["FEMALE"] == 1, "Female", "Male")

anova_univariate = [
    anova_oneway("_sex", "Sexo"),
    anova_oneway("_race", "Raza/etnicidad"),
    anova_oneway("_par_ed", "Educación de los padres"),
]
with open(RES / "anova_univariate.json", "w") as f:
    json.dump(anova_univariate, f, indent=2)


# ------------------------------------------------------------------
# 6. Validación fuera de muestra (85/15) sobre el Modelo 5 (final)
# ------------------------------------------------------------------
X_full = analytic[MODEL_SPECS["Modelo 5: + Justicia/Sustancias"]].copy()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y, test_size=0.15, random_state=SEED, stratify=y)

train_fit = sm.OLS(y_tr, sm.add_constant(X_tr)).fit(cov_type="HC1")
pred_tr = train_fit.predict(sm.add_constant(X_tr))
pred_te = train_fit.predict(sm.add_constant(X_te))


def metrics(y_true, y_pred):
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    # Para LPM: precisión clasificando con umbral 0.5
    y_hat_bin = (y_pred >= 0.5).astype(int)
    acc = float((y_hat_bin == y_true.astype(int)).mean())
    # AUC manual con Mann-Whitney
    pos = y_pred[y_true.astype(int) == 1].values
    neg = y_pred[y_true.astype(int) == 0].values
    if len(pos) > 0 and len(neg) > 0:
        u, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
        auc = float(u / (len(pos) * len(neg)))
    else:
        auc = None
    return {"r2": float(r2), "rmse": rmse, "mae": mae,
            "accuracy_0.5": acc, "auc": auc}


val = {
    "n_train": int(len(y_tr)),
    "n_test": int(len(y_te)),
    "train": metrics(y_tr, pred_tr),
    "test": metrics(y_te, pred_te),
}
val["train"]["r2_adj"] = float(train_fit.rsquared_adj)
with open(RES / "validation.json", "w") as f:
    json.dump(val, f, indent=2)


# ------------------------------------------------------------------
# 7. Diagnóstico del modelo final (Modelo 5 — preferido por AIC/BIC)
# ------------------------------------------------------------------
res_final = fitted["Modelo 5: + Justicia/Sustancias"]
X_final = analytic[MODEL_SPECS["Modelo 5: + Justicia/Sustancias"]]
resid_f = res_final.resid
fitted_f = res_final.fittedvalues

# Breusch-Pagan
bp = sm.stats.diagnostic.het_breuschpagan(
    resid_f, sm.add_constant(X_final))
bp_dict = {"lm": float(bp[0]), "lm_p": float(bp[1]),
           "f": float(bp[2]), "f_p": float(bp[3])}

# Leverage and Cook
infl = res_final.get_influence()
leverage = infl.hat_matrix_diag
cooks = infl.cooks_distance[0]
n_f = int(res_final.nobs)
p_f = int(res_final.df_model + 1)
lev_thr = 2 * p_f / n_f
cook_thr = 4 / n_f
diag = {
    "model": "Modelo 5",
    "n": n_f, "k": p_f,
    "lev_threshold": float(lev_thr),
    "cook_threshold": float(cook_thr),
    "n_high_lev": int(np.sum(leverage > lev_thr)),
    "pct_high_lev": float(np.mean(leverage > lev_thr) * 100),
    "n_high_cook": int(np.sum(cooks > cook_thr)),
    "pct_high_cook": float(np.mean(cooks > cook_thr) * 100),
    "bp": bp_dict,
}
with open(RES / "diagnostics.json", "w") as f:
    json.dump(diag, f, indent=2)


# ------------------------------------------------------------------
# 8. Figuras
# ------------------------------------------------------------------

# Fig 1: distribución de la target
fig, ax = plt.subplots(figsize=(5.5, 3.5))
counts = analytic["EVER_ASSAULT"].value_counts().sort_index()
bars = ax.bar(["No agresión\n(0)", "Reportó agresión\n(1)"],
              counts.values, color=["#4C78A8", "#E45756"], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + max(counts.values) * 0.01,
            f"{v:,}\n({v / counts.sum() * 100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Número de respondentes")
ax.set_title("Distribución de EVER_ASSAULT en la muestra analítica")
ax.set_ylim(0, max(counts.values) * 1.18)
plt.tight_layout()
plt.savefig(FIGS / "01_target_distribution.png")
plt.close()

# Fig 2: prevalencia por sexo y raza
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
prev_sex = analytic.groupby("_sex")["EVER_ASSAULT"].mean()
axes[0].bar(prev_sex.index, prev_sex.values, color=["#54A24B", "#E45756"], width=0.5)
axes[0].set_ylabel("Prob. de haber reportado agresión")
axes[0].set_title("Prevalencia por sexo")
for i, v in enumerate(prev_sex.values):
    axes[0].text(i, v + 0.01, f"{v:.2%}", ha="center", fontsize=9)
axes[0].set_ylim(0, prev_sex.max() * 1.25)

prev_race = analytic.groupby("_race")["EVER_ASSAULT"].mean().sort_values()
axes[1].barh(prev_race.index, prev_race.values, color="#4C78A8")
axes[1].set_xlabel("Prob. de haber reportado agresión")
axes[1].set_title("Prevalencia por raza/etnicidad")
for i, v in enumerate(prev_race.values):
    axes[1].text(v + 0.005, i, f"{v:.2%}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "02_prevalence_sex_race.png")
plt.close()

# Fig 3: prevalencia por habilidad cognitiva (deciles de ASVAB)
fig, ax = plt.subplots(figsize=(6.5, 3.7))
analytic["_asvab_dec"] = pd.qcut(analytic["ASVAB_SCORE"], 10, labels=False)
g = analytic.groupby("_asvab_dec")["EVER_ASSAULT"].mean()
ax.plot(g.index + 1, g.values, marker="o", color="#4C78A8")
ax.set_xlabel("Decil de ASVAB (1=bajo, 10=alto)")
ax.set_ylabel("Prob. de haber reportado agresión")
ax.set_title("Prevalencia de agresión por habilidad cognitiva (ASVAB)")
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig(FIGS / "03_prevalence_by_asvab.png")
plt.close()

# Fig 4: prevalencia por conductas tempranas (gang, gun, drogas duras)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, var, title in zip(
        axes,
        ["EARLY_GANG", "EARLY_GUN", "EARLY_HARD_DRUGS"],
        ["Pandilla (14-17)", "Portó arma (14-17)", "Drogas duras (14-17)"]):
    df_g = analytic.copy()
    df_g["_bin"] = (df_g[var] > 0).astype(int)
    g = df_g.groupby("_bin")["EVER_ASSAULT"].mean()
    ax.bar(["No", "Sí"], g.values, color=["#4C78A8", "#E45756"], width=0.55)
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.01, f"{v:.2%}", ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylim(0, max(g.values) * 1.2)
axes[0].set_ylabel("Prob. de agresión")
plt.tight_layout()
plt.savefig(FIGS / "04_prevalence_risk_behaviors.png")
plt.close()

# Fig 5: R^2 ajustada por modelo
fig, ax = plt.subplots(figsize=(7, 3.6))
labels = list(fit_table.keys())
short_labels = [l.split(":")[0] for l in labels]
r2adj = [fit_table[l]["r2_adj"] for l in labels]
ax.bar(short_labels, r2adj, color="#4C78A8")
for i, v in enumerate(r2adj):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("R² ajustada")
ax.set_title("Capacidad explicativa por modelo (R² ajustada)")
ax.set_ylim(0, max(r2adj) * 1.15)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGS / "05_r2_by_model.png")
plt.close()

# Fig 6: diagnóstico del modelo final (Modelo 5)
fig, axes = plt.subplots(2, 2, figsize=(9, 7))

# Resid vs fitted
axes[0, 0].scatter(fitted_f, resid_f, s=4, alpha=0.25, color="#4C78A8")
axes[0, 0].axhline(0, color="red", lw=0.8)
axes[0, 0].set_xlabel("Valor ajustado")
axes[0, 0].set_ylabel("Residuo")
axes[0, 0].set_title("Residuos vs ajustados")

# QQ
sm.qqplot(resid_f, line="45", fit=True, ax=axes[0, 1],
          markerfacecolor="#4C78A8", markeredgecolor="#4C78A8",
          markersize=3, alpha=0.5)
axes[0, 1].set_title("Q-Q plot de residuos")

# Scale-location
axes[1, 0].scatter(fitted_f, np.sqrt(np.abs(resid_f)),
                   s=4, alpha=0.25, color="#4C78A8")
axes[1, 0].set_xlabel("Valor ajustado")
axes[1, 0].set_ylabel("√|residuo|")
axes[1, 0].set_title("Scale-Location")

# Leverage
axes[1, 1].scatter(leverage, resid_f, s=4, alpha=0.25, color="#4C78A8")
axes[1, 1].axvline(lev_thr, color="red", lw=0.6, ls="--",
                   label=f"2p/n={lev_thr:.4f}")
axes[1, 1].set_xlabel("Leverage")
axes[1, 1].set_ylabel("Residuo")
axes[1, 1].set_title("Residuos vs leverage")
axes[1, 1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "06_diagnostics_m5.png")
plt.close()

# Fig 7: pred vs observado en test (jitter porque y es binaria)
fig, ax = plt.subplots(figsize=(6, 4))
jitter = np.random.uniform(-0.03, 0.03, size=len(y_te))
ax.scatter(pred_te, y_te.values + jitter,
           s=6, alpha=0.25, color="#4C78A8")
ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Umbral 0.5")
ax.set_xlabel("Probabilidad predicha (Modelo 5)")
ax.set_ylabel("EVER_ASSAULT observado (con jitter)")
ax.set_title(
    f"Test (n={val['n_test']:,}) — R²={val['test']['r2']:.3f}, "
    f"AUC={val['test']['auc']:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "07_pred_vs_obs_test.png")
plt.close()

print("All figures saved to", FIGS)
print("Sample size:", len(analytic))
print("Fit table:")
for k, v in fit_table.items():
    print(f"  {k:45s} R²={v['r2']:.4f}  R²adj={v['r2_adj']:.4f}"
          f"  AIC={v['aic']:.1f}  BIC={v['bic']:.1f}")
print("Validation:", val)
print("Diagnostics:", diag)
print("ANOVA sequence (incremental F):")
for a in anova_seq:
    print(f"  {a['from']} -> {a['to']}: F={a['F']:.2f} "
          f"df=({a['df_diff']},{a['df_resid']}) p={a['p']:.3g} "
          f"ΔR²={a['delta_r2']:.4f}")
