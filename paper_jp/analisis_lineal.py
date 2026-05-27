# ============================================================
# Análisis definitivo: Predictores adolescentes de agresión
# en la adultez — Modelo de Probabilidad Lineal jerárquico
#
# Datos: NLSY97 (ICPSR 34562), n ≈ 8,984 respondentes
# Target: ASSAULT_ADULT — reportó agresión en ALGÚN año 18-23
#         (target limpio: sin traslape con el período de los predictores)
# Predictores: conductas e información recopiladas a edades 14-17
#
# Estructura:
#   M1  Bloque demográfico
#   M2  + Familia y SES
#   M3  + Cognitivo (ASVAB)
#   M4  + Conducta delictiva temprana (14-17)
#   M5  + Justicia y sustancias (modelo preferido)
#   M6  + Interacción sexo × raza
#
# Para pegar en Jupyter: cada celda empieza con "# %%" y termina
# antes del siguiente "# %%". Selecciona el bloque → Insert Cell.
# ============================================================

# %% [1] ── IMPORTS Y CONFIGURACIÓN GLOBAL ──────────────────────────────────
# Todas las librerías necesarias. matplotlib.use("Agg") es obligatorio en
# entornos sin display (servidor, WSL sin GUI); en Jupyter normal puedes
# comentarlo o cambiarlo a "inline".

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # cambia a %matplotlib inline en Jupyter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/cord2108/ITAM/Semestre_9/proyecto_final_polpub")
DATA = ROOT / "data/ICPSR_34562/DS0001/34562-0001-Data.dta"
OUT  = ROOT / "paper_jp"
FIGS = OUT / "figuras"
RES  = OUT / "resultados"

FIGS.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

# ── Semilla global ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Estética de figuras ────────────────────────────────────────────────────
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
# El NLSY97 almacena muchas columnas como categorías tipo "Yes"/"No" (Stata
# value labels). Necesitamos mapearlas a 1/0 y tratar los no-válidos como NaN.

def to_yes_no_numeric(series):
    """
    Convierte una columna binaria a 1.0 / 0.0 / NaN.

    Si la columna es categórica u object (p.ej. "Yes", "No", "SKIP"):
      - "yes" → 1.0, "no" → 0.0, cualquier otra cosa → NaN
    Si ya es numérica:
      - valores < 0 (códigos de missing del NLSY: -1, -2, -3, -4, -5) → NaN
      - el resto se conserva como está
    """
    if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        out = pd.Series(np.nan, index=series.index)
        out[s.eq("yes")] = 1.0
        out[s.eq("no")]  = 0.0
        return out
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s >= 0)          # negativos = missing en NLSY


def clean_num(series):
    """
    Limpia una columna numérica continua: coerce a float y descarta negativos.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s >= 0)


# Función auxiliar: suma binaria de un prefijo a lo largo de un rango de años
# Ejemplo: sum_binary("GANGS", [14,15,16,17]) da el número de años (0-4)
# en que el respondente reportó pertenecer a una pandilla.
def sum_binary(df, prefix, ages):
    """
    Para cada respondente cuenta en cuántos años del rango informó "Yes".

    Parámetros
    ----------
    df     : DataFrame con los datos crudos
    prefix : prefijo de la variable (p.ej. "GANGS")
    ages   : lista de edades a incluir (p.ej. [14, 15, 16, 17])

    Retorna
    -------
    pd.Series con valores 0.0–len(ages); NaN si todas las columnas son NaN.
    """
    cols = [f"{prefix}_CY{a}" for a in ages if f"{prefix}_CY{a}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    mat = pd.concat([to_yes_no_numeric(df[c]) for c in cols], axis=1)
    return mat.sum(axis=1, min_count=1)   # min_count=1: si todo NaN → NaN


print("✓ Funciones de limpieza definidas")


# %% [3] ── CARGAR DATOS CRUDOS ────────────────────────────────────────────
# El archivo .dta tiene una fila por respondente (8,984 personas) y columnas
# para cada variable × año de vida. convert_categoricals=True transforma los
# value labels de Stata (ej. 1="Yes") a strings legibles.

print("Cargando datos...")
df = pd.read_stata(DATA, convert_categoricals=True)
print(f"   Shape crudo: {df.shape}")   # esperado: ~(8984, muchas columnas)

# Vista rápida de las columnas de agresión disponibles en el dataset
assault_all = sorted([c for c in df.columns if c.startswith("ASSAULT_CY")])
print(f"   Columnas ASSAULT_CY disponibles ({len(assault_all)}): {assault_all}")


# %% [4] ── CONSTRUIR EL TARGET: ASSAULT_ADULT ─────────────────────────────
# DECISIÓN DE DISEÑO CLAVE:
#   Los predictores se construyen con datos de edades 14-17.
#   Si usáramos EVER_ASSAULT (14-23), parte de la ventana del target
#   solaparía con la ventana de los predictores → data leakage temporal.
#   Solución: usar solo las columnas ASSAULT_CY18 a ASSAULT_CY23,
#   que representan agresión reportada en la adultez (18+).
#
# ASSAULT_ADULT = 1 si el respondente reportó agresión en AL MENOS UN año
#                 entre los 18 y los 23 años.
#              = 0 si tiene al menos una observación válida en ese rango
#                 y ninguna es "Yes".
#              = NaN si no tiene ninguna observación válida en 18-23.

adult_cols = [c for c in assault_all if int(c.replace("ASSAULT_CY", "")) >= 18]
print(f"Columnas adulto (18-23): {adult_cols}")

assault_adult_mat = pd.concat(
    [to_yes_no_numeric(df[c]) for c in adult_cols], axis=1
)

# Solo conservamos filas que tienen al menos UNA observación válida en 18-23
valid_adult = assault_adult_mat.notna().any(axis=1)
df = df[valid_adult].copy()
assault_adult_mat = assault_adult_mat[valid_adult]

df["ASSAULT_ADULT"] = (
    assault_adult_mat.sum(axis=1, min_count=1) > 0
).astype(int)

print(f"   Filas con al menos una obs. adulta válida: {len(df):,}")
print(f"   Prevalencia ASSAULT_ADULT: {df['ASSAULT_ADULT'].mean():.3f} "
      f"({df['ASSAULT_ADULT'].sum():,} personas)")


# %% [5] ── CONSTRUIR PREDICTORES: BLOQUE DEMOGRÁFICO ─────────────────────
# Variables fijas (una sola observación por respondente, no dependen del año).
#
# FEMALE      : 1 = mujer, 0 = hombre
# RACE_*      : dummies de raza/etnicidad (categoría de referencia = Non-Black
#               / Non-Hispanic, es decir, blanco no hispano)
# BIOMOMAGE   : edad de la madre biológica al nacer el respondente
# NUMSIBS     : número de hermanos (en 1997)

df["FEMALE"]        = (df["SEX"].astype(str).str.lower() == "female").astype(int)

race = df["RACE_ETHNICITY"].astype(str).str.strip().str.lower()
df["RACE_BLACK"]    = (race == "black").astype(int)
df["RACE_HISPANIC"] = (race == "hispanic").astype(int)
df["RACE_MIXED"]    = (race == "mixed race (non-hispanic)").astype(int)
# ref: non-black / non-hispanic (blancos)

df["BIOMOMAGE"] = clean_num(df["BIOMOMAGE"])
df["NUMSIBS"]   = clean_num(df["NUMSIBS"])

print("✓ Bloque demográfico construido")
print(f"   FEMALE          mean={df['FEMALE'].mean():.3f}")
print(f"   RACE_BLACK      mean={df['RACE_BLACK'].mean():.3f}")
print(f"   RACE_HISPANIC   mean={df['RACE_HISPANIC'].mean():.3f}")


# %% [6] ── CONSTRUIR PREDICTORES: BLOQUE FAMILIA Y SES ───────────────────
# ESTRUCTURA FAMILIAR (FMSTRC97): tipo de hogar en 1997 (cuando el
# respondente tenía 12-17 años).
#   Referencia: dos padres biológicos (FAM_TWO_BIO, omitida)
#   FAM_ONE_PARENT : un solo padre biológico/adoptivo
#   FAM_BIO_STEP   : un biológico + un padrastro/madrastra
#   FAM_OTHER      : otros tipos (adoptivo×2, otro) — poco frecuentes

fmstrc = df["FMSTRC97"].astype(str).str.strip().str.lower()
df["FAM_ONE_PARENT"] = (fmstrc == "one single biologic/adoptive parent").astype(int)
df["FAM_BIO_STEP"]   = (fmstrc == "one biological- and one step-parent").astype(int)
df["FAM_OTHER"]      = fmstrc.isin(
    ["other family type", "two adoptive parent(s)"]
).astype(int)
# FAM_TWO_BIO es la referencia (dos bio) — no entra como variable

# EDUCACIÓN DEL RESPONDENTE DE REFERENCIA (RESPARED):
#   Referencia: < preparatoria (menos de 12 años de escolaridad)
#   PAR_ED_HS       : terminó preparatoria
#   PAR_ED_SOME_COL : algo de universidad
#   PAR_ED_COL      : licenciatura o más
LT_HS = "zero through 11th grade of high school"
HS    = "graduated from hs"
SC    = "some college"
COL   = "college degree or higher"

resp_raw   = df["RESPARED"].astype(str).str.strip().str.lower()
valid_resp = resp_raw.isin([LT_HS, HS, SC, COL])

df["PAR_ED_HS"]       = np.where(valid_resp, (resp_raw == HS ).astype(int), np.nan)
df["PAR_ED_SOME_COL"] = np.where(valid_resp, (resp_raw == SC ).astype(int), np.nan)
df["PAR_ED_COL"]      = np.where(valid_resp, (resp_raw == COL).astype(int), np.nan)
# Cuando resp_raw no cae en ninguna categoría válida → NaN (excluye de muestra)

# POBREZA: razón ingreso/línea de pobreza en 1997 (continua, >0)
df["POVRATIO97"] = clean_num(df["POVRATIO97"])

print("✓ Bloque familia y SES construido")


# %% [7] ── CONSTRUIR PREDICTORES: BLOQUE COGNITIVO ───────────────────────
# ASVAB_SCORE: puntaje compuesto del Armed Services Vocational Aptitude Battery,
# administrado en 1999 (cuando los respondentes tenían ~15-18 años).
# Es una medida de habilidad cognitiva general; valores típicos 1-99 (percentil).
# Se usa como variable continua; coeficiente interpretado per punto de percentil.

df["ASVAB_SCORE"] = clean_num(df["ASVAB_SCORE"])

print(f"✓ ASVAB_SCORE: mean={df['ASVAB_SCORE'].mean():.1f}, "
      f"sd={df['ASVAB_SCORE'].std():.1f}")


# %% [8] ── CONSTRUIR PREDICTORES: BLOQUE CONDUCTA PASADA (14-17) ─────────
# Para cada conducta, contamos en CUÁNTOS de los 4 años (14, 15, 16, 17)
# el respondente reportó "Yes". El resultado es un entero 0-4.
#
# Conductas incluidas:
#   EARLY_DESTROY     : vandalismo / destrucción de propiedad
#   EARLY_STEAL_UNDER : robo < $50
#   EARLY_STEAL_OVER  : robo > $50
#   EARLY_OTHER_PROP  : otro delito contra la propiedad
#   EARLY_GANG        : pertenencia a pandilla
#   EARLY_GUN         : portar arma de fuego
#   EARLY_RUNAWAY     : huir de casa

early_ages = list(range(14, 18))   # [14, 15, 16, 17]

df["EARLY_DESTROY"]     = sum_binary(df, "DESTROY",        early_ages)
df["EARLY_STEAL_UNDER"] = sum_binary(df, "STEAL_UNDER_50", early_ages)
df["EARLY_STEAL_OVER"]  = sum_binary(df, "STEAL_OVER_50",  early_ages)
df["EARLY_OTHER_PROP"]  = sum_binary(df, "OTHER_PROPERTY", early_ages)
df["EARLY_GANG"]        = sum_binary(df, "GANGS",          early_ages)
df["EARLY_GUN"]         = sum_binary(df, "CARR_GUN",       early_ages)
df["EARLY_RUNAWAY"]     = sum_binary(df, "RUNAWAY",        early_ages)

print("✓ Bloque conducta pasada (14-17) construido")
print(f"   EARLY_GANG  mean={df['EARLY_GANG'].mean():.3f}  "
      f"(~{df['EARLY_GANG'].mean()/4*100:.1f}% de los años)")
print(f"   EARLY_GUN   mean={df['EARLY_GUN'].mean():.3f}")


# %% [9] ── CONSTRUIR PREDICTORES: BLOQUE JUSTICIA Y SUSTANCIAS (14-17) ───
# Mismo principio: conteo de años (0-4) en que el respondente reportó la conducta.
#
# Sustancias:
#   EARLY_SMOKING     : fumó cigarrillos
#   EARLY_ALCOHOL     : consumió alcohol
#   EARLY_MARIJUANA   : consumió marihuana
#   EARLY_HARD_DRUGS  : consumió drogas duras (cocaína, heroína, etc.)
#
# Contacto con el sistema de justicia:
#   EVER_ARR_BY17 : fue arrestado AL MENOS UNA VEZ entre los 14 y 17 años
#                   (binaria 0/1, no conteo — distinto a las de arriba)

df["EARLY_SMOKING"]    = sum_binary(df, "SMOKING",    early_ages)
df["EARLY_ALCOHOL"]    = sum_binary(df, "ALCOHOL",    early_ages)
df["EARLY_MARIJUANA"]  = sum_binary(df, "MARIJUANA",  early_ages)
df["EARLY_HARD_DRUGS"] = sum_binary(df, "HARD_DRUGS", early_ages)

arr_cols = [f"EVER_ARR_CY{a}" for a in early_ages
            if f"EVER_ARR_CY{a}" in df.columns]
arr_mat = pd.concat([to_yes_no_numeric(df[c]) for c in arr_cols], axis=1)
df["EVER_ARR_BY17"] = (arr_mat.max(axis=1) > 0).astype(float)
# max(axis=1) > 0 equivale a "¿alguna vez = Yes en 14-17?"

print("✓ Bloque justicia y sustancias (14-17) construido")
print(f"   EVER_ARR_BY17 prevalencia = {df['EVER_ARR_BY17'].mean():.3f}")


# %% [10] ── MUESTRA ANALÍTICA FINAL ──────────────────────────────────────
# Requisito para comparabilidad entre modelos:
# todos los modelos deben correr sobre exactamente el mismo subconjunto
# de personas (listwise deletion). Eliminamos cualquier fila que tenga
# al menos un NaN en cualquiera de las variables usadas en los 5 modelos.
#
# Esto garantiza que los cambios en R² entre M1-M5 reflejan el aporte
# informativo del bloque añadido, NO diferencias de muestra.

TODAS_LAS_VARS = [
    "ASSAULT_ADULT",
    # Bloque 1: demográfico
    "FEMALE", "RACE_BLACK", "RACE_HISPANIC", "RACE_MIXED",
    "BIOMOMAGE", "NUMSIBS",
    # Bloque 2: familia y SES
    "FAM_ONE_PARENT", "FAM_BIO_STEP", "FAM_OTHER",
    "PAR_ED_HS", "PAR_ED_SOME_COL", "PAR_ED_COL",
    "POVRATIO97",
    # Bloque 3: cognitivo
    "ASVAB_SCORE",
    # Bloque 4: conducta pasada
    "EARLY_DESTROY", "EARLY_STEAL_UNDER", "EARLY_STEAL_OVER",
    "EARLY_OTHER_PROP", "EARLY_GANG", "EARLY_GUN", "EARLY_RUNAWAY",
    # Bloque 5: justicia y sustancias
    "EARLY_SMOKING", "EARLY_ALCOHOL", "EARLY_MARIJUANA",
    "EARLY_HARD_DRUGS", "EVER_ARR_BY17",
]

analytic = df[TODAS_LAS_VARS].dropna().copy()
print(f"Muestra analítica: {len(analytic):,} personas × {len(TODAS_LAS_VARS)-1} variables")
print(f"   (excluidas por NaN en alguna variable: "
      f"{len(df) - len(analytic):,} filas)")
print(f"\nPrevalencia ASSAULT_ADULT en muestra analítica: "
      f"{analytic['ASSAULT_ADULT'].mean():.3f} "
      f"({analytic['ASSAULT_ADULT'].sum():,} personas)")

# Guardar muestra para reproducibilidad
analytic.to_csv(RES / "analytic_sample.csv", index=False)
print("✓ Muestra guardada en resultados/analytic_sample.csv")


# %% [11] ── VARIABLES AUXILIARES DE VISUALIZACIÓN ────────────────────────
# Para los gráficos de prevalencia necesitamos categorías legibles.
# Las creamos en el DataFrame analytic.

analytic["_race"] = np.where(
    analytic["RACE_BLACK"]    == 1, "Black",
    np.where(analytic["RACE_HISPANIC"] == 1, "Hispanic",
    np.where(analytic["RACE_MIXED"]    == 1, "Mixed",
                                             "Non-Black/Non-Hisp.")))

analytic["_par_ed"] = np.where(
    analytic["PAR_ED_HS"]       == 1, "Preparatoria",
    np.where(analytic["PAR_ED_SOME_COL"] == 1, "Algo de universidad",
    np.where(analytic["PAR_ED_COL"]      == 1, "Licenciatura+", "<Preparatoria")))

analytic["_sex"]  = np.where(analytic["FEMALE"] == 1, "Mujer", "Hombre")

print("✓ Variables auxiliares para visualización creadas")


# %% [12] ── DEFINICIÓN DE MODELOS JERÁRQUICOS ────────────────────────────
# Modelos anidados: cada modelo incluye todos los bloques anteriores + uno nuevo.
# Esto permite calcular el INCREMENTO en R² y hacer ANOVA incremental (F-test)
# para saber si cada bloque aporta significativamente.
#
# M1: solo demografía                  (6 predictores)
# M2: + familia y SES                  (+7, total 13)
# M3: + cognitivo                      (+1, total 14)
# M4: + conducta delictiva 14-17       (+7, total 21)
# M5: + justicia y sustancias 14-17    (+5, total 26)  ← modelo preferido
# M6: M5 + interacción FEMALE×RACE     (+2, total 28)

BLOCKS = {
    "demographic":       ["FEMALE", "RACE_BLACK", "RACE_HISPANIC", "RACE_MIXED",
                          "BIOMOMAGE", "NUMSIBS"],
    "family_ses":        ["FAM_ONE_PARENT", "FAM_BIO_STEP", "FAM_OTHER",
                          "PAR_ED_HS", "PAR_ED_SOME_COL", "PAR_ED_COL",
                          "POVRATIO97"],
    "cognitive":         ["ASVAB_SCORE"],
    "past_behavior":     ["EARLY_DESTROY", "EARLY_STEAL_UNDER", "EARLY_STEAL_OVER",
                          "EARLY_OTHER_PROP", "EARLY_GANG", "EARLY_GUN",
                          "EARLY_RUNAWAY"],
    "justice_substance": ["EARLY_SMOKING", "EARLY_ALCOHOL", "EARLY_MARIJUANA",
                          "EARLY_HARD_DRUGS", "EVER_ARR_BY17"],
}

MODEL_SPECS = {
    "M1: Demográfico": (
        BLOCKS["demographic"]
    ),
    "M2: +Familia y SES": (
        BLOCKS["demographic"] + BLOCKS["family_ses"]
    ),
    "M3: +Cognitivo": (
        BLOCKS["demographic"] + BLOCKS["family_ses"] + BLOCKS["cognitive"]
    ),
    "M4: +Conducta pasada": (
        BLOCKS["demographic"] + BLOCKS["family_ses"] +
        BLOCKS["cognitive"]   + BLOCKS["past_behavior"]
    ),
    "M5: +Justicia/Sustancias": (
        BLOCKS["demographic"] + BLOCKS["family_ses"] +
        BLOCKS["cognitive"]   + BLOCKS["past_behavior"] +
        BLOCKS["justice_substance"]
    ),
}

# Columnas del M5 para reutilizar en validación y diagnóstico
M5_COLS = MODEL_SPECS["M5: +Justicia/Sustancias"]

print("Modelos definidos:")
for name, cols in MODEL_SPECS.items():
    print(f"   {name:35s} {len(cols):2d} predictores")


# %% [13] ── FUNCIÓN OLS (LPM) ────────────────────────────────────────────
# Modelo de Probabilidad Lineal (LPM) = OLS con variable dependiente binaria.
# Ventajas: coeficientes directamente interpretables como cambios en probabilidad
#           en puntos porcentuales; fácil comparación entre modelos anidados.
# Corrección HC1: errores estándar heteroskedasticity-consistent (Eicker-Huber-
# White), obligatoria en LPM porque la varianza de y es p(1-p) ≠ cte.

def fit_ols(y, X):
    """
    Ajusta OLS con constante y errores HC1.

    Parámetros
    ----------
    y : pd.Series  variable dependiente binaria
    X : pd.DataFrame  predictores (sin constante)

    Retorna
    -------
    statsmodels RegressionResultsWrapper
    """
    X_ = sm.add_constant(X, has_constant="add")   # añade columna de 1s
    return sm.OLS(y, X_).fit(cov_type="HC1")


# %% [14] ── AJUSTAR LOS 5 MODELOS JERÁRQUICOS + M6 ───────────────────────
# Ajustamos cada modelo sobre la muestra analítica completa (n=analytic).
# Los 5 modelos son "en muestra" (in-sample); la validación fuera de muestra
# se hace aparte en la Celda 17.

y = analytic["ASSAULT_ADULT"].astype(float)

fitted = {}
for name, cols in MODEL_SPECS.items():
    fitted[name] = fit_ols(y, analytic[cols])
    r2 = fitted[name].rsquared_adj
    print(f"   {name:35s} R²adj = {r2:.4f}")

# Modelo 6: M5 + interacción FEMALE × RACE_BLACK y FEMALE × RACE_HISPANIC
# Permite que el gap de género sea diferente para hombres/mujeres por grupo racial
X6 = analytic[M5_COLS].copy()
X6["FEMALE_x_BLACK"]    = X6["FEMALE"] * X6["RACE_BLACK"]
X6["FEMALE_x_HISPANIC"] = X6["FEMALE"] * X6["RACE_HISPANIC"]
fitted["M6: +Interacción sex×raza"] = fit_ols(y, X6)

print(f"   {'M6: +Interacción sex×raza':35s} "
      f"R²adj = {fitted['M6: +Interacción sex×raza'].rsquared_adj:.4f}")

print(f"\n✓ {len(fitted)} modelos ajustados")


# %% [15] ── TABLA DE AJUSTE (R², AIC, BIC, F global) ─────────────────────
# Exporta las métricas de ajuste de cada modelo a JSON y CSV.
# Explicación de cada métrica:
#   n         : tamaño de muestra (igual para todos = comparabilidad)
#   k         : número de predictores (sin contar la constante)
#   r2        : R² (proporción de varianza explicada, no ajustada)
#   r2_adj    : R² ajustada por grados de libertad (penaliza variables extra)
#   aic / bic : criterios de información (penalizan complejidad; menor = mejor)
#   ll        : log-verosimilitud (mayor = mejor)
#   fstat     : estadístico F global (H0: todos los coefs = 0)
#   f_pvalue  : p-value del F global

def model_summary(res):
    return {
        "n":        int(res.nobs),
        "k":        int(res.df_model),
        "r2":       float(res.rsquared),
        "r2_adj":   float(res.rsquared_adj),
        "aic":      float(res.aic),
        "bic":      float(res.bic),
        "ll":       float(res.llf),
        "fstat":    float(res.fvalue)   if res.fvalue   is not None else None,
        "f_pvalue": float(res.f_pvalue) if res.f_pvalue is not None else None,
    }

fit_table = {name: model_summary(res) for name, res in fitted.items()}

# Guardar
with open(RES / "fit_table.json", "w", encoding="utf-8") as f:
    json.dump(fit_table, f, indent=2, ensure_ascii=False)

# Imprimir resumen
print(f"\n{'Modelo':40s}  {'R²adj':>7}  {'AIC':>9}  {'BIC':>9}")
print("-" * 72)
for name, m in fit_table.items():
    print(f"{name:40s}  {m['r2_adj']:7.4f}  {m['aic']:9.1f}  {m['bic']:9.1f}")


# %% [16] ── TABLA DE COEFICIENTES ────────────────────────────────────────
# Exporta para cada modelo: coeficiente, SE robusto, t, p, IC 95%.
# También genera una tabla "wide" con todos los modelos lado a lado
# (útil para replicar la tabla de regresión del paper en LaTeX).

def coef_frame(res):
    """Extrae coefs, SE, t, p e IC 95% de un resultado OLS."""
    return pd.DataFrame({
        "coef":    res.params,
        "se":      res.bse,
        "t":       res.tvalues,
        "p":       res.pvalues,
        "ci_low":  res.conf_int()[0],
        "ci_high": res.conf_int()[1],
    })

coef_tables = {}
for name, res in fitted.items():
    cf = coef_frame(res)
    safe_name = name.split(":")[0].replace(" ", "_")
    cf.to_csv(RES / f"coefs__{safe_name}.csv")
    coef_tables[name] = cf

# Tabla wide: todos los modelos en columnas
all_terms = []
for cf in coef_tables.values():
    for t in cf.index:
        if t not in all_terms:
            all_terms.append(t)

wide = pd.DataFrame(index=all_terms)
for name, cf in coef_tables.items():
    short = name.split(":")[0]
    wide[f"{short}_coef"] = cf["coef"].reindex(all_terms)
    wide[f"{short}_se"]   = cf["se"].reindex(all_terms)
    wide[f"{short}_p"]    = cf["p"].reindex(all_terms)
wide.to_csv(RES / "coefs_wide.csv")

print("✓ Tablas de coeficientes guardadas en resultados/")

# Vista de M5 (modelo preferido)
print("\nCoeficientes M5 (selección):")
m5_cf = coef_tables["M5: +Justicia/Sustancias"]
for var in ["FEMALE", "RACE_BLACK", "RACE_HISPANIC", "ASVAB_SCORE",
            "EARLY_GANG", "EARLY_GUN", "EVER_ARR_BY17"]:
    if var in m5_cf.index:
        c, s, p = m5_cf.loc[var, ["coef", "se", "p"]]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"   {var:22s}  β={c:+.4f}  SE={s:.4f}  p={p:.4f} {stars}")


# %% [17] ── ANOVA INCREMENTAL (F-TEST ENTRE MODELOS ANIDADOS) ────────────
# Para cada par consecutivo (M1→M2, M2→M3, ..., M5→M6) calculamos:
#   F incremental = [(RSS_pequeño - RSS_grande) / Δdf] / [RSS_grande / df_resid]
# donde RSS = suma de cuadrados de residuos y Δdf = número de predictores añadidos.
#
# La H0 es que todos los coeficientes NUEVOS del bloque son cero.
# Un p < 0.05 indica que el bloque añade poder explicativo estadísticamente
# significativo más allá de los bloques anteriores.

def incremental_f(res_small, res_big):
    rss_s   = float(np.sum(res_small.resid ** 2))
    rss_b   = float(np.sum(res_big.resid ** 2))
    df_diff = int(res_big.df_model - res_small.df_model)
    df_b    = int(res_big.df_resid)
    F  = ((rss_s - rss_b) / df_diff) / (rss_b / df_b)
    p  = 1 - stats.f.cdf(F, df_diff, df_b)
    return {
        "from":     res_small.model.endog_names if hasattr(res_small.model, "endog_names") else "?",
        "df_diff":  df_diff,
        "df_resid": df_b,
        "F":        float(F),
        "p":        float(p),
        "delta_r2": float(res_big.rsquared - res_small.rsquared),
        "rss_small": rss_s,
        "rss_big":   rss_b,
    }

names      = list(fitted.keys())
anova_seq  = []
print(f"\n{'Transición':45s}  {'ΔR²':>7}  {'F':>8}  {'df':>10}  p")
print("-" * 80)
for i in range(1, len(names)):
    info = incremental_f(fitted[names[i-1]], fitted[names[i]])
    info["from"] = names[i-1]
    info["to"]   = names[i]
    anova_seq.append(info)
    stars = "***" if info["p"] < 0.001 else "**" if info["p"] < 0.01 else "*" if info["p"] < 0.05 else ""
    print(f"{names[i-1][:20]:20s} → {names[i][:20]:20s}  "
          f"{info['delta_r2']:7.4f}  {info['F']:8.2f}  "
          f"({info['df_diff']},{info['df_resid']})  {info['p']:.3g} {stars}")

with open(RES / "anova_sequence.json", "w", encoding="utf-8") as f:
    json.dump(anova_seq, f, indent=2, ensure_ascii=False)

print("\n✓ ANOVA incremental guardado en resultados/anova_sequence.json")


# %% [18] ── ANOVA UNIVARIADO (DESCRIPTIVO) ──────────────────────────────
# ANOVA de un factor para ver si la prevalencia de ASSAULT_ADULT difiere
# significativamente entre grupos de sexo, raza y educación de los padres.
# Esto es solo descriptivo (no controla por nada).

def anova_oneway(group_var, label):
    groups = []
    labels = []
    for lvl, sub in analytic.groupby(group_var):
        groups.append(sub["ASSAULT_ADULT"].values)
        labels.append(str(lvl))
    F, p = stats.f_oneway(*groups)
    means = {lab: float(g.mean()) for lab, g in zip(labels, groups)}
    return {"label": label, "F": float(F), "p": float(p),
            "means": means, "n_groups": len(groups)}

anova_univ = [
    anova_oneway("_sex",    "Sexo"),
    anova_oneway("_race",   "Raza/etnicidad"),
    anova_oneway("_par_ed", "Educación de los padres"),
]

for a in anova_univ:
    print(f"{a['label']:30s}  F={a['F']:.2f}  p={a['p']:.3g}")
    for lvl, mean in sorted(a["means"].items()):
        print(f"   {lvl:25s}  {mean:.3f}")

with open(RES / "anova_univariate.json", "w", encoding="utf-8") as f:
    json.dump(anova_univ, f, indent=2, ensure_ascii=False)


# %% [19] ── VALIDACIÓN FUERA DE MUESTRA — 85/15 ──────────────────────────
# Separamos individuos (no años) al azar:
#   85% → entrenamiento  (ajustamos el modelo)
#   15% → prueba         (evaluamos sin haber visto esos datos)
# stratify=y asegura que la proporción de 1s sea la misma en train y test.
#
# IMPORTANTE: esto mide qué tan bien generaliza el modelo a NUEVAS PERSONAS,
# no si supera la leakage temporal (que ya resolvimos con el target adulto).

X_full = analytic[M5_COLS].copy()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y, test_size=0.15, random_state=SEED, stratify=y
)

print(f"Train: {len(y_tr):,} personas  |  Test: {len(y_te):,} personas")
print(f"Prevalencia train: {y_tr.mean():.3f}  |  test: {y_te.mean():.3f}")

# Ajustar solo con datos de entrenamiento
m5_train = fit_ols(y_tr, X_tr)
pred_tr  = m5_train.predict(sm.add_constant(X_tr, has_constant="add"))
pred_te  = m5_train.predict(sm.add_constant(X_te, has_constant="add"))


def eval_metrics(y_true, y_pred, label=""):
    """Calcula RMSE, MAE, R², accuracy (umbral 0.5) y AUC Mann-Whitney."""
    err    = y_true - y_pred
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    mae    = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1 - ss_res / ss_tot

    y_bin  = (y_pred >= 0.5).astype(int)
    acc    = float((y_bin == y_true.astype(int)).mean())

    pos = y_pred[y_true.astype(int) == 1].values
    neg = y_pred[y_true.astype(int) == 0].values
    if len(pos) > 0 and len(neg) > 0:
        u, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
        auc  = float(u / (len(pos) * len(neg)))
    else:
        auc  = None

    if label:
        print(f"  {label:10s}  R²={r2:.4f}  RMSE={rmse:.4f}  "
              f"AUC={auc:.4f}  Acc={acc:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae, "accuracy_0.5": acc, "auc": auc}

print("\nMétricas de validación M5 (target ASSAULT_ADULT):")
val = {
    "n_train": int(len(y_tr)),
    "n_test":  int(len(y_te)),
    "train":   eval_metrics(y_tr, pred_tr, "Train"),
    "test":    eval_metrics(y_te, pred_te, "Test"),
}
val["train"]["r2_adj"] = float(m5_train.rsquared_adj)

with open(RES / "validation.json", "w", encoding="utf-8") as f:
    json.dump(val, f, indent=2, ensure_ascii=False)


# %% [20] ── DIAGNÓSTICOS DEL MODELO FINAL (M5) ──────────────────────────
# Test de Breusch-Pagan: confirma heterocedasticidad → justifica HC1.
# Leverage y distancia de Cook: detecta observaciones influyentes.

res_m5    = fitted["M5: +Justicia/Sustancias"]
X_m5      = analytic[M5_COLS]
resid_m5  = res_m5.resid
fitted_m5 = res_m5.fittedvalues

# Breusch-Pagan (H0: homocedasticidad)
bp = sm.stats.diagnostic.het_breuschpagan(resid_m5, sm.add_constant(X_m5))
print(f"Breusch-Pagan  LM={bp[0]:.2f}  p={bp[1]:.4g}  "
      f"(p < 0.05 → rechaza H0 → HC1 necesario)")

# Leverage y Cook
infl     = res_m5.get_influence()
leverage = infl.hat_matrix_diag
cooks    = infl.cooks_distance[0]
n_m5     = int(res_m5.nobs)
p_m5     = int(res_m5.df_model + 1)
lev_thr  = 2 * p_m5 / n_m5     # umbral convencional: 2p/n
cook_thr = 4 / n_m5             # umbral convencional: 4/n

diag = {
    "n": n_m5, "k": p_m5,
    "lev_threshold":  float(lev_thr),
    "cook_threshold": float(cook_thr),
    "n_high_lev":     int(np.sum(leverage > lev_thr)),
    "pct_high_lev":   float(np.mean(leverage > lev_thr) * 100),
    "n_high_cook":    int(np.sum(cooks > cook_thr)),
    "pct_high_cook":  float(np.mean(cooks > cook_thr) * 100),
    "bp": {"lm": float(bp[0]), "lm_p": float(bp[1]),
           "f":  float(bp[2]), "f_p":  float(bp[3])},
}

print(f"Leverage > {lev_thr:.4f}: {diag['n_high_lev']:,} obs ({diag['pct_high_lev']:.1f}%)")
print(f"Cook > {cook_thr:.4f}: {diag['n_high_cook']:,} obs ({diag['pct_high_cook']:.1f}%)")

with open(RES / "diagnostics.json", "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2, ensure_ascii=False)


# %% [21] ── FIGURA 1: Distribución del target ───────────────────────────
# Barras simples: cuántas personas tienen ASSAULT_ADULT = 0 vs 1.

fig, ax = plt.subplots(figsize=(5.5, 3.5))
counts = analytic["ASSAULT_ADULT"].value_counts().sort_index()
bars = ax.bar(
    ["Sin agresión adulta\n(0)", "Reportó agresión\nen adultez (1)"],
    counts.values, color=["#4C78A8", "#E45756"], width=0.55
)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + max(counts.values) * 0.01,
            f"{v:,}\n({v / counts.sum() * 100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Número de respondentes")
ax.set_title("Distribución de ASSAULT_ADULT\n(agresión reportada a los 18-23 años)")
ax.set_ylim(0, max(counts.values) * 1.20)
plt.tight_layout()
plt.savefig(FIGS / "01_target_distribution.png")
plt.close()
print("✓ Figura 1 guardada")


# %% [22] ── FIGURA 2: Prevalencia por sexo y raza ───────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))

# Panel izquierdo: por sexo
prev_sex = analytic.groupby("_sex")["ASSAULT_ADULT"].mean()
axes[0].bar(prev_sex.index, prev_sex.values,
            color=["#54A24B", "#E45756"], width=0.5)
axes[0].set_ylabel("P(ASSAULT_ADULT = 1)")
axes[0].set_title("Prevalencia por sexo")
for i, v in enumerate(prev_sex.values):
    axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
axes[0].set_ylim(0, prev_sex.max() * 1.30)

# Panel derecho: por raza
prev_race = analytic.groupby("_race")["ASSAULT_ADULT"].mean().sort_values()
axes[1].barh(prev_race.index, prev_race.values, color="#4C78A8")
axes[1].set_xlabel("P(ASSAULT_ADULT = 1)")
axes[1].set_title("Prevalencia por raza/etnicidad")
for i, v in enumerate(prev_race.values):
    axes[1].text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIGS / "02_prevalence_sex_race.png")
plt.close()
print("✓ Figura 2 guardada")


# %% [23] ── FIGURA 3: Prevalencia por deciles de ASVAB ──────────────────
# Muestra el gradiente cognitivo: personas con mayor habilidad (ASVAB alto)
# reportan menos agresión en la adultez.

fig, ax = plt.subplots(figsize=(6.5, 3.7))
analytic["_asvab_dec"] = pd.qcut(analytic["ASVAB_SCORE"], 10, labels=False)
g = analytic.groupby("_asvab_dec")["ASSAULT_ADULT"].mean()
ax.plot(g.index + 1, g.values, marker="o", color="#4C78A8", linewidth=2)
ax.set_xlabel("Decil de ASVAB (1 = menor habilidad, 10 = mayor)")
ax.set_ylabel("P(ASSAULT_ADULT = 1)")
ax.set_title("Prevalencia de agresión adulta por habilidad cognitiva (ASVAB)")
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig(FIGS / "03_prevalence_by_asvab.png")
plt.close()
print("✓ Figura 3 guardada")


# %% [24] ── FIGURA 4: Prevalencia por conductas de riesgo (14-17) ────────
# Para pandilla, arma y drogas duras: compara la prevalencia de ASSAULT_ADULT
# entre quienes reportaron la conducta (al menos 1 año) y quienes no.

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, var, title in zip(
        axes,
        ["EARLY_GANG", "EARLY_GUN", "EARLY_HARD_DRUGS"],
        ["Pandilla (14-17)", "Portó arma (14-17)", "Drogas duras (14-17)"]):
    tmp = analytic.copy()
    tmp["_bin"] = (tmp[var] > 0).astype(int)
    g = tmp.groupby("_bin")["ASSAULT_ADULT"].mean()
    colors = ["#4C78A8", "#E45756"]
    ax.bar(["No", "Sí"], g.values, color=colors, width=0.55)
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylim(0, max(g.values) * 1.25)
axes[0].set_ylabel("P(ASSAULT_ADULT = 1)")
plt.suptitle("Prevalencia de agresión adulta según conductas de riesgo adolescentes",
             y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "04_prevalence_risk_behaviors.png")
plt.close()
print("✓ Figura 4 guardada")


# %% [25] ── FIGURA 5: R² ajustada por modelo (capacidad explicativa) ─────
# Muestra el incremento en R²adj al añadir cada bloque.
# El salto más grande esperado es en M4 (conducta pasada) y M5 (arresto).

fig, ax = plt.subplots(figsize=(7.5, 3.8))
model_names = list(fit_table.keys())
short_labels = [n.split(":")[0] for n in model_names]
r2adj_vals   = [fit_table[n]["r2_adj"] for n in model_names]

bars = ax.bar(short_labels, r2adj_vals, color="#4C78A8")
for i, (b, v) in enumerate(zip(bars, r2adj_vals)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002,
            f"{v:.3f}", ha="center", fontsize=9)

ax.set_ylabel("R² ajustada")
ax.set_title("Capacidad explicativa por modelo\n(target = ASSAULT_ADULT, 18-23 años)")
ax.set_ylim(0, max(r2adj_vals) * 1.18)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGS / "05_r2_by_model.png")
plt.close()
print("✓ Figura 5 guardada")


# %% [26] ── FIGURA 6: Diagnósticos del modelo final (M5) ─────────────────
# Cuatro paneles estándar para evaluar supuestos del OLS:
#   (a) Residuos vs ajustados: patrón cero = sin sesgo sistemático
#   (b) QQ plot: evalúa normalidad de residuos (no crítica en LPM n grande)
#   (c) Scale-Location: evalúa homocedasticidad
#   (d) Residuos vs leverage: identifica observaciones influyentes

fig, axes = plt.subplots(2, 2, figsize=(9, 7))

# (a) Residuos vs ajustados
axes[0, 0].scatter(fitted_m5, resid_m5, s=4, alpha=0.25, color="#4C78A8")
axes[0, 0].axhline(0, color="red", lw=0.8)
axes[0, 0].set_xlabel("Valor ajustado")
axes[0, 0].set_ylabel("Residuo")
axes[0, 0].set_title("Residuos vs ajustados")

# (b) QQ
sm.qqplot(resid_m5, line="45", fit=True, ax=axes[0, 1],
          markerfacecolor="#4C78A8", markeredgecolor="#4C78A8",
          markersize=3, alpha=0.5)
axes[0, 1].set_title("Q-Q plot de residuos")

# (c) Scale-Location (√|e| vs ajustados)
axes[1, 0].scatter(fitted_m5, np.sqrt(np.abs(resid_m5)),
                   s=4, alpha=0.25, color="#4C78A8")
axes[1, 0].set_xlabel("Valor ajustado")
axes[1, 0].set_ylabel("√|residuo|")
axes[1, 0].set_title("Scale-Location")

# (d) Residuos vs leverage
axes[1, 1].scatter(leverage, resid_m5, s=4, alpha=0.25, color="#4C78A8")
axes[1, 1].axvline(lev_thr, color="red", lw=0.6, ls="--",
                   label=f"2p/n = {lev_thr:.4f}")
axes[1, 1].set_xlabel("Leverage")
axes[1, 1].set_ylabel("Residuo")
axes[1, 1].set_title("Residuos vs leverage")
axes[1, 1].legend(fontsize=8)

plt.suptitle("Diagnóstico del modelo final (M5)", y=1.01, fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "06_diagnostics_m5.png")
plt.close()
print("✓ Figura 6 guardada")


# %% [27] ── FIGURA 7: Predichos vs observados en test ───────────────────
# Visualiza el desempeño fuera de muestra del M5.
# Como la y observada es binaria (0/1), se añade jitter vertical para ver
# la distribución de las probabilidades predichas por clase.

fig, ax = plt.subplots(figsize=(6.5, 4.2))
jitter = np.random.uniform(-0.035, 0.035, size=len(y_te))
ax.scatter(pred_te, y_te.values + jitter,
           s=6, alpha=0.25, color="#4C78A8")
ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Umbral 0.5")
ax.axvline(0.5, color="gray", ls=":",  lw=0.6)
ax.set_xlabel("Probabilidad predicha (M5)")
ax.set_ylabel("ASSAULT_ADULT observado (con jitter)")
ax.set_title(
    f"Test set (n={val['n_test']:,}) — R²={val['test']['r2']:.3f}, "
    f"AUC={val['test']['auc']:.3f}"
)
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "07_pred_vs_obs_test.png")
plt.close()
print("✓ Figura 7 guardada")


# %% [28] ── FIGURA 8: Coeficientes del M5 con IC 95% ─────────────────────
# Forest plot de los coeficientes clave del M5. Muestra el tamaño del efecto
# y el intervalo de confianza robusto (HC1) al 95%.
# ASVAB se escala a 80 puntos (rango intercuartílico aprox.) para compararlo
# con los demás en una escala interpretable.

KEY_VARS = [
    "FEMALE", "RACE_BLACK", "RACE_HISPANIC", "RACE_MIXED",
    "ASVAB_SCORE",
    "EARLY_GANG", "EARLY_GUN", "EARLY_RUNAWAY",
    "EARLY_DESTROY", "EARLY_HARD_DRUGS",
    "EVER_ARR_BY17",
]
LABELS = {
    "FEMALE":           "Mujer (ref: hombre)",
    "RACE_BLACK":       "Black (ref: white)",
    "RACE_HISPANIC":    "Hispanic (ref: white)",
    "RACE_MIXED":       "Mixed (ref: white)",
    "ASVAB_SCORE":      "ASVAB (×80 pp)",
    "EARLY_GANG":       "Pandilla 14-17 (años)",
    "EARLY_GUN":        "Portó arma 14-17 (años)",
    "EARLY_RUNAWAY":    "Huyó de casa 14-17 (años)",
    "EARLY_DESTROY":    "Vandalismo 14-17 (años)",
    "EARLY_HARD_DRUGS": "Drogas duras 14-17 (años)",
    "EVER_ARR_BY17":    "Arrestado antes de los 18",
}
SCALE = {v: 80 if v == "ASVAB_SCORE" else 1 for v in KEY_VARS}

cf_m5 = coef_tables["M5: +Justicia/Sustancias"]
coefs = [(LABELS[v],
          cf_m5.loc[v, "coef"] * SCALE[v],
          cf_m5.loc[v, "ci_low"] * SCALE[v],
          cf_m5.loc[v, "ci_high"] * SCALE[v],
          cf_m5.loc[v, "p"])
         for v in KEY_VARS if v in cf_m5.index]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
y_pos = np.arange(len(coefs))

for i, (label, est, lo, hi, pval) in enumerate(reversed(coefs)):
    color = "#E45756" if pval < 0.05 else "#AAAAAA"
    ax.plot([lo, hi], [i, i], color=color, lw=2)
    ax.plot(est, i, "o", color=color, markersize=7)

ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_yticks(y_pos)
ax.set_yticklabels([c[0] for c in reversed(coefs)], fontsize=9)
ax.set_xlabel("Cambio en P(agresión adulta) — puntos porcentuales")
ax.set_title("Coeficientes M5 con IC 95% (HC1)\nrojo = p < 0.05")
plt.tight_layout()
plt.savefig(FIGS / "08_coefs_m5_forest.png")
plt.close()
print("✓ Figura 8 guardada")


# %% [29] ── RESUMEN FINAL ─────────────────────────────────────────────────
# Imprime un resumen ejecutivo de los resultados más importantes.

print("\n" + "=" * 65)
print("RESUMEN EJECUTIVO — ASSAULT_ADULT (18-23 años)")
print("=" * 65)
print(f"\nMuestra analítica: n = {len(analytic):,}")
print(f"Prevalencia target: {analytic['ASSAULT_ADULT'].mean():.3f} "
      f"({analytic['ASSAULT_ADULT'].sum():,}/{len(analytic):,})")

print("\nAjuste por modelo:")
print(f"  {'Modelo':35s}  {'R²adj':>7}  {'AIC':>9}")
for name, m in fit_table.items():
    print(f"  {name:35s}  {m['r2_adj']:7.4f}  {m['aic']:9.1f}")

print(f"\nValidación fuera de muestra (M5, test 15%):")
print(f"  AUC  = {val['test']['auc']:.4f}")
print(f"  R²   = {val['test']['r2']:.4f}")
print(f"  RMSE = {val['test']['rmse']:.4f}")

print(f"\nFiguras guardadas en: {FIGS}")
print(f"Resultados guardados en: {RES}")
print("=" * 65)
