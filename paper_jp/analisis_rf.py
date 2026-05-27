# ============================================================
# Random Forest — predicción de agresión en la adultez
# con importancia de bloque por permutación
#
# Datos: NLSY97 (ICPSR 34562)
# Target: ASSAULT_ADULT — reportó agresión en algún año 18-23
# Predictores: conductas e información de edades 14-17
#
# Este script entrena un Random Forest sobre los mismos datos
# que analisis_lineal.py y evalúa la importancia de cada BLOQUE
# de variables mediante block permutation importance.
#
# Estructura de celdas (misma lógica que analisis_lineal.py):
#   [1-2]   Imports y funciones de limpieza
#   [3-4]   Cargar datos y construir target
#   [5-9]   Construir los 5 bloques de predictores
#   [10-11] Muestra analítica y split 85/15
#   [12]    Definición de bloques
#   [13]    Entrenamiento del Random Forest
#   [14]    Evaluación en test (AUC, ROC)
#   [15]    Marco teórico: permutation & block permutation importance
#   [16]    Block permutation importance
#   [17-20] Figuras
#   [21]    Resumen final
# ============================================================

# %% [1] ── IMPORTS Y CONFIGURACIÓN GLOBAL ──────────────────────────────────

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # cambiar a %matplotlib inline en Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/cord2108/ITAM/Semestre_9/proyecto_final_polpub")
DATA = ROOT / "data/ICPSR_34562/DS0001/34562-0001-Data.dta"
OUT  = ROOT / "paper_jp"
FIGS = OUT / "figuras_rf"
RES  = OUT / "resultados_rf"

FIGS.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

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

print("✓ Configuración lista")


# %% [2] ── FUNCIONES DE LIMPIEZA ──────────────────────────────────────────
# Idénticas a analisis_lineal.py. Ver comentarios allá para detalle.

def to_yes_no_numeric(series):
    if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        out = pd.Series(np.nan, index=series.index)
        out[s.eq("yes")] = 1.0
        out[s.eq("no")]  = 0.0
        return out
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s >= 0)


def clean_num(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s >= 0)


def sum_binary(df, prefix, ages):
    cols = [f"{prefix}_CY{a}" for a in ages if f"{prefix}_CY{a}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    mat = pd.concat([to_yes_no_numeric(df[c]) for c in cols], axis=1)
    return mat.sum(axis=1, min_count=1)


print("✓ Funciones de limpieza definidas")


# %% [3] ── CARGAR DATOS CRUDOS ────────────────────────────────────────────

print("Cargando datos...")
df = pd.read_stata(DATA, convert_categoricals=True)
print(f"   Shape crudo: {df.shape}")


# %% [4] ── CONSTRUIR EL TARGET: ASSAULT_ADULT ─────────────────────────────
# Target limpio (sin leakage): agresión reportada en algún año entre los
# 18 y 23 años. Los predictores cubren edades 14-17, por lo que no hay
# traslape temporal.

assault_all  = sorted([c for c in df.columns if c.startswith("ASSAULT_CY")])
adult_cols   = [c for c in assault_all if int(c.replace("ASSAULT_CY", "")) >= 18]

assault_adult_mat = pd.concat(
    [to_yes_no_numeric(df[c]) for c in adult_cols], axis=1
)
valid_adult = assault_adult_mat.notna().any(axis=1)
df = df[valid_adult].copy()
assault_adult_mat = assault_adult_mat[valid_adult]

df["ASSAULT_ADULT"] = (
    assault_adult_mat.sum(axis=1, min_count=1) > 0
).astype(int)

print(f"   Filas con obs. adulta válida: {len(df):,}")
print(f"   Prevalencia ASSAULT_ADULT: {df['ASSAULT_ADULT'].mean():.3f} "
      f"({df['ASSAULT_ADULT'].sum():,} personas)")


# %% [5] ── PREDICTORES: BLOQUE DEMOGRÁFICO ────────────────────────────────

df["FEMALE"]        = (df["SEX"].astype(str).str.lower() == "female").astype(int)
race = df["RACE_ETHNICITY"].astype(str).str.strip().str.lower()
df["RACE_BLACK"]    = (race == "black").astype(int)
df["RACE_HISPANIC"] = (race == "hispanic").astype(int)
df["RACE_MIXED"]    = (race == "mixed race (non-hispanic)").astype(int)
df["BIOMOMAGE"]     = clean_num(df["BIOMOMAGE"])
df["NUMSIBS"]       = clean_num(df["NUMSIBS"])

print("✓ Bloque demográfico")


# %% [6] ── PREDICTORES: BLOQUE FAMILIA Y SES ─────────────────────────────

fmstrc = df["FMSTRC97"].astype(str).str.strip().str.lower()
df["FAM_ONE_PARENT"] = (fmstrc == "one single biologic/adoptive parent").astype(int)
df["FAM_BIO_STEP"]   = (fmstrc == "one biological- and one step-parent").astype(int)
df["FAM_OTHER"]      = fmstrc.isin(
    ["other family type", "two adoptive parent(s)"]
).astype(int)

LT_HS = "zero through 11th grade of high school"
HS, SC, COL = "graduated from hs", "some college", "college degree or higher"
resp_raw   = df["RESPARED"].astype(str).str.strip().str.lower()
valid_resp = resp_raw.isin([LT_HS, HS, SC, COL])
df["PAR_ED_HS"]       = np.where(valid_resp, (resp_raw == HS ).astype(int), np.nan)
df["PAR_ED_SOME_COL"] = np.where(valid_resp, (resp_raw == SC ).astype(int), np.nan)
df["PAR_ED_COL"]      = np.where(valid_resp, (resp_raw == COL).astype(int), np.nan)

df["POVRATIO97"] = clean_num(df["POVRATIO97"])

print("✓ Bloque familia y SES")


# %% [7] ── PREDICTORES: BLOQUE COGNITIVO ─────────────────────────────────

df["ASVAB_SCORE"] = clean_num(df["ASVAB_SCORE"])

print(f"✓ ASVAB_SCORE  mean={df['ASVAB_SCORE'].mean():.1f}  sd={df['ASVAB_SCORE'].std():.1f}")


# %% [8] ── PREDICTORES: BLOQUE CONDUCTA PASADA (14-17) ───────────────────

early_ages = list(range(14, 18))

df["EARLY_DESTROY"]     = sum_binary(df, "DESTROY",        early_ages)
df["EARLY_STEAL_UNDER"] = sum_binary(df, "STEAL_UNDER_50", early_ages)
df["EARLY_STEAL_OVER"]  = sum_binary(df, "STEAL_OVER_50",  early_ages)
df["EARLY_OTHER_PROP"]  = sum_binary(df, "OTHER_PROPERTY", early_ages)
df["EARLY_GANG"]        = sum_binary(df, "GANGS",          early_ages)
df["EARLY_GUN"]         = sum_binary(df, "CARR_GUN",       early_ages)
df["EARLY_RUNAWAY"]     = sum_binary(df, "RUNAWAY",        early_ages)

print("✓ Bloque conducta pasada (14-17)")


# %% [9] ── PREDICTORES: BLOQUE JUSTICIA Y SUSTANCIAS (14-17) ─────────────

df["EARLY_SMOKING"]    = sum_binary(df, "SMOKING",    early_ages)
df["EARLY_ALCOHOL"]    = sum_binary(df, "ALCOHOL",    early_ages)
df["EARLY_MARIJUANA"]  = sum_binary(df, "MARIJUANA",  early_ages)
df["EARLY_HARD_DRUGS"] = sum_binary(df, "HARD_DRUGS", early_ages)

arr_cols = [f"EVER_ARR_CY{a}" for a in early_ages if f"EVER_ARR_CY{a}" in df.columns]
arr_mat  = pd.concat([to_yes_no_numeric(df[c]) for c in arr_cols], axis=1)
df["EVER_ARR_BY17"] = (arr_mat.max(axis=1) > 0).astype(float)

print("✓ Bloque justicia y sustancias (14-17)")


# %% [10] ── MUESTRA ANALÍTICA FINAL ──────────────────────────────────────
# Exactamente las mismas 3,737 personas que en analisis_lineal.py.
# (mismo listwise deletion sobre las mismas variables)

ALL_VARS = [
    "ASSAULT_ADULT",
    "FEMALE", "RACE_BLACK", "RACE_HISPANIC", "RACE_MIXED", "BIOMOMAGE", "NUMSIBS",
    "FAM_ONE_PARENT", "FAM_BIO_STEP", "FAM_OTHER",
    "PAR_ED_HS", "PAR_ED_SOME_COL", "PAR_ED_COL", "POVRATIO97",
    "ASVAB_SCORE",
    "EARLY_DESTROY", "EARLY_STEAL_UNDER", "EARLY_STEAL_OVER",
    "EARLY_OTHER_PROP", "EARLY_GANG", "EARLY_GUN", "EARLY_RUNAWAY",
    "EARLY_SMOKING", "EARLY_ALCOHOL", "EARLY_MARIJUANA", "EARLY_HARD_DRUGS",
    "EVER_ARR_BY17",
]

analytic = df[ALL_VARS].dropna().copy().reset_index(drop=True)
print(f"Muestra analítica: {len(analytic):,} personas × {len(ALL_VARS)-1} variables")
print(f"Prevalencia ASSAULT_ADULT: {analytic['ASSAULT_ADULT'].mean():.3f}")


# %% [11] ── SPLIT 85 / 15 ────────────────────────────────────────────────
# Misma semilla y misma proporción que en analisis_lineal.py para que las
# métricas de test sean comparables entre el LPM y el RF.

FEATURE_COLS = [v for v in ALL_VARS if v != "ASSAULT_ADULT"]

X = analytic[FEATURE_COLS]
y = analytic["ASSAULT_ADULT"].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

print(f"Train: {len(y_tr):,} personas  |  Test: {len(y_te):,} personas")
print(f"Prevalencia train: {y_tr.mean():.3f}  |  test: {y_te.mean():.3f}")


# %% [12] ── DEFINICIÓN DE BLOQUES ────────────────────────────────────────
# Los mismos 5 bloques que en el LPM. Esta estructura es central para la
# block permutation importance de la Celda [16].

BLOCKS = {
    "B1: Demográfico":       ["FEMALE", "RACE_BLACK", "RACE_HISPANIC", "RACE_MIXED",
                               "BIOMOMAGE", "NUMSIBS"],
    "B2: Familia y SES":     ["FAM_ONE_PARENT", "FAM_BIO_STEP", "FAM_OTHER",
                               "PAR_ED_HS", "PAR_ED_SOME_COL", "PAR_ED_COL",
                               "POVRATIO97"],
    "B3: Cognitivo":         ["ASVAB_SCORE"],
    "B4: Conducta pasada":   ["EARLY_DESTROY", "EARLY_STEAL_UNDER", "EARLY_STEAL_OVER",
                               "EARLY_OTHER_PROP", "EARLY_GANG", "EARLY_GUN",
                               "EARLY_RUNAWAY"],
    "B5: Justicia/Sust.":    ["EARLY_SMOKING", "EARLY_ALCOHOL", "EARLY_MARIJUANA",
                               "EARLY_HARD_DRUGS", "EVER_ARR_BY17"],
}

print("Bloques definidos:")
for name, cols in BLOCKS.items():
    print(f"   {name:25s}  {len(cols):2d} variables")


# %% [13] ── ENTRENAR EL RANDOM FOREST ────────────────────────────────────
# Entrenamos un RF sobre los 26 predictores usando SOLO los datos de train.
#
# Hiperparámetros elegidos:
#   n_estimators=500   : 500 árboles — suficiente para estabilizar el OOB
#                        y las importancias sin ser excesivamente lento
#   max_features='sqrt': en cada split se evalúan sqrt(p) ≈ 5 variables al azar;
#                        es el default recomendado para clasificación (reduce
#                        correlación entre árboles)
#   min_samples_leaf=10: un nodo terminal debe tener al menos 10 observaciones;
#                        regulariza el árbol y reduce sobreajuste dada la
#                        muestra relativamente pequeña (n≈3,737)
#   class_weight='balanced': ajusta el peso de cada clase por su frecuencia
#                        inversa; necesario porque ASSAULT_ADULT≈14% (desbalance)
#   oob_score=True     : calcula AUC sobre las muestras out-of-bag de cada árbol
#                        (aproximación de validación cruzada "gratis")

rf = RandomForestClassifier(
    n_estimators    = 500,
    max_features    = "sqrt",
    min_samples_leaf= 10,
    class_weight    = "balanced",
    oob_score       = True,
    random_state    = SEED,
    n_jobs          = -1,          # usa todos los núcleos disponibles
)

rf.fit(X_tr, y_tr)

print(f"✓ RF entrenado")
print(f"   OOB AUC ≈ {roc_auc_score(y_tr, rf.oob_decision_function_[:, 1]):.4f}")


# %% [14] ── EVALUACIÓN EN TEST ────────────────────────────────────────────

prob_te  = rf.predict_proba(X_te)[:, 1]   # P(ASSAULT_ADULT=1)
auc_test = roc_auc_score(y_te, prob_te)

prob_tr  = rf.predict_proba(X_tr)[:, 1]
auc_train = roc_auc_score(y_tr, prob_tr)

print(f"AUC train : {auc_train:.4f}")
print(f"AUC test  : {auc_test:.4f}   ← métrica principal")
print(f"Diferencia: {auc_train - auc_test:.4f}  (sobreajuste residual)")


# %% [15] ── MARCO TEÓRICO: PERMUTATION & BLOCK PERMUTATION IMPORTANCE ─────
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  PERMUTATION IMPORTANCE (Fisher, Rudin & Dominici 2019;             │
# │  Breiman 2001 — variable importance en RF originales)               │
# └──────────────────────────────────────────────────────────────────────┘
#
# La idea central es: si una variable X_j es importante para el modelo,
# entonces ROMPER su relación con y (permutando aleatoriamente sus valores)
# debería degradar el rendimiento del modelo. Formalmente:
#
#   PI(X_j) = S(f, D) - S(f, D_j^perm)
#
#   donde S(f, D)        = AUC del modelo f sobre el conjunto de datos D
#         D_j^perm       = D pero con la columna j permutada
#         PI(X_j)        = drop en AUC al romper la información de X_j
#
# Ventajas sobre la importancia Gini (MDI) de un RF estándar:
#   1. Es model-agnostic: funciona con cualquier modelo, no solo árboles.
#   2. No está sesgada hacia variables de alta cardinalidad o continuas
#      (problema conocido de la impureza Gini).
#   3. Medida en unidades de la métrica de evaluación (AUC drops), por lo
#      que es directamente interpretable: "perder este predictor cuesta
#      X puntos de AUC".
#
# Nota: la permutation importance DEBE calcularse sobre el conjunto de TEST
# (o equivalentemente sobre datos OOB), nunca sobre train. En train, el
# modelo ya memorizó los datos, y la importancia estaría sobreestimada.
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  BLOCK PERMUTATION IMPORTANCE                                       │
# └──────────────────────────────────────────────────────────────────────┘
#
# Es una extensión natural: en lugar de permutar una variable a la vez,
# se permutan SIMULTÁNEAMENTE todas las variables de un bloque B_k.
#
#   BPI(B_k) = S(f, D) - S(f, D_{B_k}^perm)
#
# La permutación simultánea es crítica: si permutáramos cada variable del
# bloque por separado y sumáramos, obtendríamos una subestimación cuando
# las variables del bloque están correlacionadas entre sí (p.ej. EARLY_GANG
# y EARLY_GUN están positivamente correlacionadas; permutar solo una deja
# parte de la señal del bloque intacta en la otra).
#
# Al permutar el bloque completo de golpe, la covarianza interna se
# destruye y la señal del bloque desaparece por completo, dando la
# importancia CONJUNTA real del bloque.
#
# Analogía con el LPM jerárquico:
#   En analisis_lineal.py calculamos ΔR² al añadir cada bloque (F-test
#   incremental). La block permutation importance es el análogo no-lineal:
#   mide cuánto cae el AUC cuando se elimina la información de un bloque,
#   manteniendo el resto intacto.
#
#   | LPM (lineal)                  | RF (no-lineal)                |
#   |-------------------------------|-------------------------------|
#   | ΔR²(B_k) = R²_M5 − R²_M5\Bk | BPI(B_k) = AUC − AUC_{perm}  |
#   | F incremental                 | permutation drop              |
#
# Para obtener intervalos de confianza usamos N_REPEATS permutaciones
# independientes. La media y el percentil 2.5-97.5 dan un IC-95% empírico.
#
# Referencia metodológica:
#   Strobl et al. (2008). Conditional variable importance for Random Forests.
#   BMC Bioinformatics, 9, 307.
#   Molnar (2022). Interpretable Machine Learning. Cap. 8.5.

print("✓ Marco teórico documentado en el código")


# %% [16] ── BLOCK PERMUTATION IMPORTANCE ─────────────────────────────────
# Algoritmo:
#   Para cada bloque B_k:
#     Para cada repetición r = 1..N_REPEATS:
#       1. Copiar X_test
#       2. Permutar simultáneamente todas las columnas de B_k
#       3. Calcular AUC con la copia permutada
#       4. Registrar drop = AUC_base - AUC_perm
#   Reportar: media y SD del drop entre las N_REPEATS repeticiones.
#
# N_REPEATS=100 da estimaciones estables con intervalos de confianza
# razonables para un test set de ~561 personas.

N_REPEATS = 100
auc_base  = auc_test          # AUC del modelo con X_test original

bpi_results = {}

print(f"AUC baseline (test): {auc_base:.4f}\n")
print(f"{'Bloque':28s}  {'Drop medio':>11}  {'SD':>7}  {'IC 95%':>20}")
print("-" * 72)

for block_name, block_cols in BLOCKS.items():
    drops = []
    for _ in range(N_REPEATS):
        X_perm = X_te.copy()
        # Permutamos TODAS las columnas del bloque con el mismo índice
        # aleatorio → destruye la covarianza interna del bloque
        perm_idx = np.random.permutation(len(X_perm))
        X_perm[block_cols] = X_perm[block_cols].values[perm_idx]

        prob_perm = rf.predict_proba(X_perm)[:, 1]
        auc_perm  = roc_auc_score(y_te, prob_perm)
        drops.append(auc_base - auc_perm)

    drops = np.array(drops)
    bpi_results[block_name] = {
        "mean":  float(drops.mean()),
        "std":   float(drops.std()),
        "ci_lo": float(np.percentile(drops, 2.5)),
        "ci_hi": float(np.percentile(drops, 97.5)),
        "raw":   drops.tolist(),
    }
    m, s, lo, hi = drops.mean(), drops.std(), np.percentile(drops, 2.5), np.percentile(drops, 97.5)
    print(f"{block_name:28s}  {m:+11.4f}  {s:7.4f}  [{lo:+.4f}, {hi:+.4f}]")

# Guardar
with open(RES / "block_permutation_importance.json", "w", encoding="utf-8") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "raw"}
               for k, v in bpi_results.items()}, f, indent=2, ensure_ascii=False)

print(f"\n✓ Resultados guardados en resultados_rf/block_permutation_importance.json")


# %% [17] ── FIGURA 1: Block Permutation Importance ───────────────────────
# Gráfica de barras horizontales con IC 95% empírico.
# Un drop positivo y con IC que no cruza cero indica que el bloque
# aporta información real al modelo.

names_sorted = sorted(bpi_results.keys(),
                      key=lambda k: bpi_results[k]["mean"],
                      reverse=False)

means  = [bpi_results[k]["mean"]  for k in names_sorted]
ci_lo  = [bpi_results[k]["ci_lo"] for k in names_sorted]
ci_hi  = [bpi_results[k]["ci_hi"] for k in names_sorted]
xerr_lo = [m - lo for m, lo in zip(means, ci_lo)]
xerr_hi = [hi - m for m, hi in zip(means, ci_hi)]

colors = ["#E45756" if m > 0 else "#AAAAAA" for m in means]

fig, ax = plt.subplots(figsize=(8, 4.5))
y_pos = np.arange(len(names_sorted))

ax.barh(y_pos, means, xerr=[xerr_lo, xerr_hi],
        color=colors, alpha=0.85, capsize=5, height=0.55,
        error_kw={"ecolor": "black", "lw": 1.2})
ax.axvline(0, color="black", lw=0.8, ls="--")

ax.set_yticks(y_pos)
ax.set_yticklabels(names_sorted, fontsize=10)
ax.set_xlabel("Caída en AUC al permutar el bloque\n(IC 95% empírico — 100 repeticiones)")
ax.set_title(
    f"Block Permutation Importance — Random Forest\n"
    f"(AUC base = {auc_base:.4f} en test set, n = {len(y_te):,})"
)

# Anotar valores
for i, (m, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
    ax.text(max(means) * 1.02, i, f"{m:+.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIGS / "01_block_permutation_importance.png")
plt.close()
print("✓ Figura 1 guardada")


# %% [18] ── FIGURA 2: Curva ROC en test ──────────────────────────────────
# Curva ROC completa del RF en el conjunto de test.
# Compara visualmente la capacidad discriminativa con la diagonal (azar).

fpr, tpr, thresholds = roc_curve(y_te, prob_te)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, color="#4C78A8", lw=2,
        label=f"RF  AUC = {auc_test:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Azar (AUC = 0.5)")
ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)")
ax.set_title("Curva ROC — Random Forest\n(test set, 15% de la muestra)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIGS / "02_roc_curve.png")
plt.close()
print("✓ Figura 2 guardada")


# %% [19] ── FIGURA 3: Distribución del drop por bloque (boxplot) ─────────
# Cada bloque tiene 100 drops simulados. El boxplot muestra la estabilidad
# de la estimación: una caja angosta = drop estable entre repeticiones.

fig, ax = plt.subplots(figsize=(8, 4.5))
data_bp = [bpi_results[k]["raw"] for k in names_sorted]

bp = ax.boxplot(data_bp, vert=False, patch_artist=True,
                medianprops={"color": "black", "lw": 1.5})
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_yticks(range(1, len(names_sorted) + 1))
ax.set_yticklabels(names_sorted, fontsize=10)
ax.set_xlabel("Caída en AUC (100 repeticiones de permutación)")
ax.set_title("Variabilidad de Block Permutation Importance\n"
             "(cada punto = una permutación aleatoria del bloque)")
plt.tight_layout()
plt.savefig(FIGS / "03_block_perm_boxplot.png")
plt.close()
print("✓ Figura 3 guardada")


# %% [20] ── FIGURA 4: Comparación LPM vs RF (ΔR² vs AUC drop) ───────────
# Pone en paralelo el aporte de cada bloque según el LPM (ΔR² incremental
# de analisis_lineal.py) y según el RF (AUC drop de block permutation).
# Permite ver si ambos modelos cuentan la misma historia o divergen.
#
# Nota: las escalas son distintas (ΔR² vs ΔAUC), así que se normalizan
# ambas series a [0,1] para la comparación visual.

# ΔR² del LPM (tomados de los resultados guardados por analisis_lineal.py)
# Si el archivo no existe todavía, el bloque se salta con un aviso.
anova_path = OUT / "resultados" / "anova_sequence.json"

if anova_path.exists():
    with open(anova_path) as f:
        anova_seq = json.load(f)

    # Mapear ΔR² de M1→M2, M2→M3, M3→M4, M4→M5 a los bloques B2-B5
    # (M1 ya incluye B1, así que B1 no tiene ΔR² incremental directo;
    #  se usa el R²adj del M1 como aproximación)
    lpm_r2_m1 = None
    fit_path = OUT / "resultados" / "fit_table.json"
    if fit_path.exists():
        with open(fit_path) as f:
            fit_table = json.load(f)
        lpm_r2_m1 = fit_table.get("M1: Demográfico", {}).get("r2_adj", None)

    # delta_r2 por bloque según la secuencia ANOVA
    # anova_seq[0]: M1→M2 = B2 (Familia/SES)
    # anova_seq[1]: M2→M3 = B3 (Cognitivo)
    # anova_seq[2]: M3→M4 = B4 (Conducta pasada)
    # anova_seq[3]: M4→M5 = B5 (Justicia/Sust.)
    block_order = list(BLOCKS.keys())   # B1 … B5
    lpm_delta  = {}
    lpm_delta[block_order[0]] = lpm_r2_m1 if lpm_r2_m1 else 0.0   # R²adj M1 para B1
    for i, entry in enumerate(anova_seq[:4]):
        lpm_delta[block_order[i + 1]] = entry["delta_r2"]

    rf_drop = {k: bpi_results[k]["mean"] for k in block_order}

    # Normalizar ambas series a [0,1]
    lpm_vals = np.array([lpm_delta[k] for k in block_order])
    rf_vals  = np.array([rf_drop[k]   for k in block_order])
    lpm_norm = lpm_vals / lpm_vals.sum()
    rf_norm  = rf_vals  / rf_vals.sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(block_order))
    w = 0.35
    ax.bar(x - w/2, lpm_norm, w, label="LPM  (ΔR² normalizado)", color="#4C78A8", alpha=0.85)
    ax.bar(x + w/2, rf_norm,  w, label="RF   (AUC drop norm.)",  color="#E45756", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split(":")[0] for k in block_order], rotation=10)
    ax.set_ylabel("Importancia relativa (normalizada)")
    ax.set_title("Importancia por bloque: LPM vs Random Forest\n"
                 "(ambas series normalizadas a suma=1 para comparar proporciones)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "04_lpm_vs_rf_block_importance.png")
    plt.close()
    print("✓ Figura 4 guardada (comparación LPM vs RF)")
else:
    print("⚠ No se encontró anova_sequence.json — Figura 4 omitida.")
    print("  Corre primero analisis_lineal.py para generarla.")


# %% [21] ── RESUMEN FINAL ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("RESUMEN — RANDOM FOREST + BLOCK PERMUTATION IMPORTANCE")
print("=" * 65)
print(f"\nMuestra analítica : {len(analytic):,}  (train={len(y_tr):,} | test={len(y_te):,})")
print(f"Prevalencia target: {analytic['ASSAULT_ADULT'].mean():.3f}")
print(f"\nAUC OOB  : {roc_auc_score(y_tr, rf.oob_decision_function_[:, 1]):.4f}")
print(f"AUC train: {auc_train:.4f}")
print(f"AUC test : {auc_test:.4f}")

print(f"\nBlock Permutation Importance (N_REPEATS={N_REPEATS}):")
print(f"  {'Bloque':28s}  {'Drop medio':>11}  {'IC 95%':>22}")
for k in sorted(bpi_results, key=lambda k: bpi_results[k]["mean"], reverse=True):
    m  = bpi_results[k]["mean"]
    lo = bpi_results[k]["ci_lo"]
    hi = bpi_results[k]["ci_hi"]
    print(f"  {k:28s}  {m:+11.4f}  [{lo:+.4f}, {hi:+.4f}]")

print(f"\nFiguras  → {FIGS}")
print(f"Resultados → {RES}")
print("=" * 65)
