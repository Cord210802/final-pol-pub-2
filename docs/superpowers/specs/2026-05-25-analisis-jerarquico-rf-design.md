# Spec — Análisis jerárquico con regresión logística anidada y Random Forest

**Proyecto:** Rediseño del análisis empírico del paper "Análisis de importancia de variables con modelos de aprendizaje de máquina para la agresión física en la adultez joven"

**Fecha:** 2026-05-25

**Autores:** Jerónimo Deli Larios, Juan Pablo Cordero Mayorga

---

## 1. Pregunta de investigación

**Principal:** ¿Qué bloques de información de la adolescencia (14-17) están más asociados con la agresión física en la adultez joven (18-23), y qué bloque aporta más al ajuste predictivo del modelo, controlando por los anteriores?

**Secundaria:** ¿La importancia relativa de los bloques es robusta entre dos métodos distintos: regresión logística clásica y Random Forest con permutation importance por bloque?

El diseño es **asociativo y predictivo**, no causal. No se interpretan los coeficientes como efectos causales.

---

## 2. Datos

- **Fuente:** ICPSR 34562, NLSY97 (Bureau of Justice Statistics)
- **Muestra analítica:** Personas con al menos una observación válida del outcome entre los 18 y 23 años (`n ≈ 8,609`)
- **Códigos de no-respuesta** (`-3`, `-4`, `-6`, `-7`, `-8`) se tratan como `NaN`
- **Imputación principal:** mediana para continuas, modo para categóricas
- **Imputación de robustez:** Multiple Imputation by Chained Equations (MICE) con `m=20` (anexo)

### Variable dependiente

```
Y = ASSAULT_18_23  ∈ {0, 1}
  = 1 si la persona reportó haber agredido físicamente a alguien con intención
       de herir en al menos una ocasión entre los 18 y 23 años
  = 0 si reportó ninguna agresión en ese rango
  = NaN (excluido) si nunca fue observado en ese rango
```

Prevalencia esperada: ~17%.

---

## 3. Estructura en 7 bloques temáticos

Los bloques se ordenan **de más exógeno a más proximal** (los más estables al nacer entran primero; las conductas modificables del propio joven entran al final). El orden no es arbitrario: refleja una jerarquía teórica de causas distales a proximales.

### Bloque 1 — Demográficas
Variables fijas o casi fijas al nacer.

| Variable | Tipo | Codificación |
|---|---|---|
| `SEX` | binaria | 1=Hombre, 0=Mujer (referencia) |
| `RACE_ETHNICITY` | categórica 4 niveles | dummies; referencia = Blanco no-hispano |
| `BIOMOMAGE` | continua | edad de la madre al nacer |
| `NUMSIBS` | continua | número de hermanos en 1997 |

### Bloque 2 — Familia y capital social
Contexto de crianza, no resultado del joven.

| Variable | Tipo | Codificación |
|---|---|---|
| `FMSTRC97` | categórica 5 niveles | dummies; referencia = Dos padres biológicos |
| `RESPARED` | categórica 4 niveles | dummies; referencia = College o más |

### Bloque 3 — Recursos económicos
Restricción material del hogar.

| Variable | Tipo | Codificación |
|---|---|---|
| `POVRATIO97` | continua → cuartiles | dummies Q1, Q2, Q3; referencia = Q4 (más alto) |
| `HHINC_EARLY` | continua → cuartiles | dummies Q1, Q2, Q3; referencia = Q4 |

Razón de discretizar: la relación entre ingreso y agresión no es lineal y cuartiles son más interpretables para política pública ("el cuartil más pobre tiene OR X vs el más rico").

### Bloque 4 — Capital cognitivo
Característica individual medida temprano (baseline 1997).

| Variable | Tipo | Codificación |
|---|---|---|
| `ASVAB_SCORE` | continua → cuartiles | dummies Q1, Q2, Q3; referencia = Q4 |

### Bloque 5 — Conducta antisocial temprana 14-17
Comportamiento observado del joven en la adolescencia.

| Variable | Tipo | Codificación |
|---|---|---|
| `EARLY_ASSAULT` | binaria | 1 si reportó agresión a alguna edad 14-17 |
| `EARLY_DESTROY` | binaria | 1 si destruyó propiedad ajena 14-17 |
| `EARLY_STEAL_LO` | binaria | 1 si robó <$50 a alguna edad 14-17 |
| `EARLY_STEAL_HI` | binaria | 1 si robó ≥$50 a alguna edad 14-17 |
| `EARLY_SELL_DRUGS` | binaria | 1 si vendió drogas 14-17 |

### Bloque 6 — Sustancias y riesgo 14-17
Otras conductas adolescentes paralelas.

| Variable | Tipo | Codificación |
|---|---|---|
| `EARLY_ALCOHOL` | binaria | 1 si consumió alcohol 14-17 |
| `EARLY_MARIJUANA` | binaria | 1 si consumió marihuana 14-17 |
| `EARLY_HARD_DRUGS` | binaria | 1 si consumió drogas duras 14-17 |
| `EARLY_SMOKING` | binaria | 1 si fumó tabaco 14-17 |
| `EARLY_CARR_GUN` | binaria | 1 si portó arma 14-17 |
| `EARLY_GANG` | binaria | 1 si perteneció a pandilla 14-17 |

### Bloque 7 — Contacto con sistema de justicia (14-17)
Outcome intermedio del sistema; se separa de Bloque 6 porque es resultado institucional, no conducta directa.

| Variable | Tipo | Codificación |
|---|---|---|
| `ARR_BY17` | binaria | 1 si fue arrestado antes de los 17 |
| `INCARC_BY17` | binaria | 1 si fue encarcelado antes de los 17 |

### Resumen de dimensionalidad

- Variables conceptuales: ~26
- Features tras dummies y cuartiles: ~35
- Ratio n/k ≈ 8,609 / 35 ≈ 246 (muy cómodo, lejos de overfitting)

---

## 4. Diseño empírico: 7 regresiones logísticas anidadas

### 4.1 Especificación general

Para el modelo `m ∈ {1, ..., 7}`:

```
logit[P(Y_i = 1)] = α_m + Σ_{b=1}^{m} X_i^{(b)} β^{(b)}
```

donde `X_i^{(b)}` es el vector de variables del bloque `b` para el individuo `i`, y `β^{(b)}` el vector de coeficientes correspondiente. Cada modelo `m` incluye **todos los bloques de 1 a m** (anidados).

### 4.2 Estimación

- **Método:** Máxima verosimilitud (MLE) vía `statsmodels.GLM` o `statsmodels.Logit`
- **Errores estándar:** robustos a heterocedasticidad (Huber-White, `cov_type='HC1'`)
- **Sin pesos muestrales** en el modelo principal; con pesos en robustez
- **Multicolinealidad:** reportar VIF en anexo; flaggear si VIF > 5

### 4.3 Comparación entre modelos anidados (Deviance / LR test)

Para cada par consecutivo `(m, m+1)`:

```
LR = -2 × [logL(M_m) - logL(M_{m+1})]
   = Deviance(M_m) - Deviance(M_{m+1})
```

donde `Deviance(M) = -2 × logL(M)`.

Bajo la hipótesis nula de que las variables del bloque `m+1` no aportan información:

```
LR ~ χ²_k   con k = número de parámetros nuevos del bloque m+1
```

**Implementación en Python:**

```python
import statsmodels.api as sm
from scipy import stats

# Modelos anidados ya estimados como modelo_1, ..., modelo_7
# Cada uno con .llf (log-likelihood) y .df_model

for m in range(1, 7):
    lr = -2 * (modelo[m].llf - modelo[m+1].llf)
    df = modelo[m+1].df_model - modelo[m].df_model
    p = 1 - stats.chi2.cdf(lr, df)
    print(f"M{m} → M{m+1}: LR χ²({df}) = {lr:.2f}, p = {p:.4g}")
```

**Equivalente en R** (para validación cruzada):

```r
anova(modelo_1, modelo_2, modelo_3, modelo_4, modelo_5, modelo_6, modelo_7,
      test = "LRT")
```

### 4.4 ANOVA de comparación de modelos

Reportar tabla tipo `anova()` con:

- **Df** (grados de libertad del modelo)
- **Deviance** (residual deviance)
- **Df cambio** (variables nuevas)
- **Deviance cambio** (= LR χ²)
- **p-value** (de la distribución χ²)

Esta tabla es **el output central del análisis jerárquico**.

### 4.5 Métricas de ajuste reportadas por modelo

Para cada `M_m`:

- **Log-likelihood** (logL)
- **Deviance** (-2 × logL)
- **AIC** = -2logL + 2k
- **BIC** = -2logL + k·ln(n)
- **Pseudo-R² de McFadden** = 1 - logL(M_m) / logL(M_0)
- **Pseudo-R² de Nagelkerke** (más interpretable, escala 0-1)
- **AUC en muestra completa** (para conexión con el RF)

---

## 5. Random Forest con permutation importance individual y por bloque

### 5.1 Especificación

- **Algoritmo:** `RandomForestClassifier` (sklearn)
- **Hiperparámetros:**
  - `n_estimators=500`
  - `max_depth=10`
  - `min_samples_leaf=20`
  - `max_features='sqrt'`
  - `class_weight='balanced'` (prevalencia ~17%)
  - `random_state=42`
- **Features:** los ~35 features tras dummies y cuartiles (mismas que la logística completa M7)
- **Split:** 80/20 estratificado por outcome
- **Validación adicional:** 5-fold CV estratificada, reportar AUC media y SD

### 5.2 Permutation importance — versión INDIVIDUAL

Mide la importancia de **cada variable por separado** permutando solo esa columna.

**Procedimiento:**

```python
from sklearn.inspection import permutation_importance

result_ind = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring='roc_auc',
    n_jobs=-1
)

# Para cada variable j:
#   importance_j = AUC_original - mean(AUC tras permutar X_j, 30 veces)
```

**Output reportado:**

| Variable | ΔAUC media | ΔAUC SD | IC 95% (percentiles) |
|---|---|---|---|
| EARLY_ASSAULT | 0.041 | 0.005 | [0.032, 0.050] |
| EARLY_GANG | 0.018 | 0.003 | [0.013, 0.024] |
| ... | ... | ... | ... |

**Notas técnicas:**
- Calcular sobre **test set**, no train (mide capacidad de generalización, no uso del modelo)
- `n_repeats=30` para tener IC robustos
- Reportar con error bars en gráfica de barras horizontales

**Limitación conocida:** variables correlacionadas se "cubren" entre sí. Si `EARLY_ALCOHOL` y `EARLY_MARIJUANA` están altamente correlacionadas, permutar solo una no destruye toda la señal y ambas aparecen menos importantes de lo que son conjuntamente. Esto motiva el análisis por bloque.

### 5.3 Permutation importance — versión POR BLOQUE

Mide la importancia conjunta de **todo un bloque temático** permutando todas sus variables a la vez.

**Procedimiento:**

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def permutation_importance_block(model, X, y, block_columns, n_repeats=30, random_state=42):
    """
    Permuta SIMULTÁNEAMENTE todas las columnas del bloque y mide ΔAUC.
    """
    rng = np.random.RandomState(random_state)
    auc_orig = roc_auc_score(y, model.predict_proba(X)[:, 1])

    diffs = []
    for _ in range(n_repeats):
        X_perm = X.copy()
        # Permutar cada columna del bloque independientemente
        for col in block_columns:
            X_perm[col] = rng.permutation(X_perm[col].values)
        auc_perm = roc_auc_score(y, model.predict_proba(X_perm)[:, 1])
        diffs.append(auc_orig - auc_perm)

    return {
        'mean': np.mean(diffs),
        'std': np.std(diffs),
        'ci_low': np.percentile(diffs, 2.5),
        'ci_high': np.percentile(diffs, 97.5),
    }

# Definición de bloques (las mismas que la regresión jerárquica)
BLOCKS = {
    'B1_Demograficas':       ['SEX', 'RACE_ETH_*', 'BIOMOMAGE', 'NUMSIBS'],
    'B2_Familia':             ['FMSTRC97_*', 'RESPARED_*'],
    'B3_Recursos_economicos': ['POVRATIO97_Q*', 'HHINC_EARLY_Q*'],
    'B4_Cognitivo':           ['ASVAB_SCORE_Q*'],
    'B5_Conducta_antisocial': ['EARLY_ASSAULT', 'EARLY_DESTROY',
                               'EARLY_STEAL_LO', 'EARLY_STEAL_HI',
                               'EARLY_SELL_DRUGS'],
    'B6_Sustancias_riesgo':   ['EARLY_ALCOHOL', 'EARLY_MARIJUANA',
                               'EARLY_HARD_DRUGS', 'EARLY_SMOKING',
                               'EARLY_CARR_GUN', 'EARLY_GANG'],
    'B7_Justicia':            ['ARR_BY17', 'INCARC_BY17'],
}

# Calcular importancia por bloque
results = {}
for block_name, cols in BLOCKS.items():
    cols_real = [c for c in X_test.columns if any(c.startswith(p.rstrip('*'))
                                                   for p in cols)]
    results[block_name] = permutation_importance_block(
        rf, X_test, y_test, cols_real, n_repeats=30
    )
```

**Output reportado:**

| Bloque | ΔAUC media | ΔAUC SD | IC 95% | Ranking |
|---|---|---|---|---|
| B5 Conducta antisocial | 0.082 | 0.008 | [0.067, 0.097] | 1 |
| B6 Sustancias/riesgo | 0.041 | 0.006 | [0.030, 0.052] | 2 |
| B4 Cognitivo | 0.024 | 0.004 | [0.017, 0.031] | 3 |
| B7 Justicia | 0.019 | 0.003 | [0.013, 0.025] | 4 |
| B2 Familia | 0.015 | 0.003 | [0.010, 0.020] | 5 |
| B3 Recursos | 0.013 | 0.003 | [0.008, 0.018] | 6 |
| B1 Demográficas | 0.011 | 0.002 | [0.007, 0.015] | 7 |

### 5.4 Triangulación con la regresión jerárquica

Construir tabla de comparación de rankings entre los dos métodos:

| Bloque | Ranking logística (ΔR²) | Ranking RF (ΔAUC permutación) | ¿Coinciden? |
|---|---|---|---|
| B5 Conducta antisocial | 1 | 1 | ✓ |
| B6 Sustancias/riesgo | 2-3 | 2 | ≈ |
| B4 Cognitivo | 3-4 | 3 | ≈ |
| ... | ... | ... | ... |

**Interpretación:**
- Si los rankings convergen → resultado robusto a la elección de método
- Si difieren → hay no-linealidad o interacciones que la logística no captura. Investigar interacciones en sección de extensión.

---

## 6. Análisis complementarios

### 6.1 Estadística descriptiva (Tabla 1)

Comparación bivariada por outcome (`Y=0` vs `Y=1`):
- **Variables continuas:** test t (Welch si varianzas desiguales)
- **Variables categóricas:** Chi-cuadrado o test exacto de Fisher (n pequeño)
- Reportar media (SD) o % por grupo, diferencia y p-value

### 6.2 Tabla principal de coeficientes (Tabla 2)

Tabla apaisada con 7 columnas (M1 a M7), filas = variables, celdas = coeficientes con SE entre paréntesis y estrellas de significancia. Al pie:
- N
- Pseudo-R² Nagelkerke
- AIC
- LR χ² vs modelo anterior con p-value

### 6.3 Aporte por bloque (Tabla 3)

| Bloque agregado | df | LR χ² | p-value | ΔR² Nagelkerke | AIC |
|---|---|---|---|---|---|
| M1: Demográficas | 4 | — | — | 0.038 | 7,420 |
| M2: + Familia | 7 | 189 | <0.001 | 0.023 | 7,231 |
| M3: + Recursos | 6 | 154 | <0.001 | 0.014 | 7,062 |
| M4: + Cognitivo | 3 | 98 | <0.001 | 0.012 | 6,964 |
| M5: + Conducta antisocial | 5 | 612 | <0.001 | 0.076 | 6,352 |
| M6: + Sustancias/riesgo | 6 | 98 | <0.001 | 0.018 | 6,254 |
| M7: + Justicia | 2 | 64 | <0.001 | 0.009 | 6,190 |

### 6.4 Coefficient stability plot (Figura 1)

Para las 6-8 variables top, mostrar cómo el coeficiente cambia al agregar bloques (líneas con puntos en cada modelo M1 a M7). Visualiza qué tan robusto es cada efecto a controles adicionales.

### 6.5 Permutation importance del RF (Figura 2 y 3)

- **Figura 2:** importancia individual (barras horizontales con error bars), top 15
- **Figura 3:** importancia por bloque (barras horizontales con error bars), 7 bloques ordenados

### 6.6 Comparación de rankings (Figura 4)

Plot tipo "slope chart": cada bloque tiene un ranking en logística y otro en RF, unidos por una línea. Visualiza convergencia o divergencia entre métodos.

---

## 7. Análisis de robustez (Anexo)

### 7.1 Imputación múltiple (MICE)

- `m = 20` imputaciones con `IterativeImputer` o `mice` de R
- Estimar M7 en cada imputación
- Combinar con Rubin's rules
- Tabla comparativa de coeficientes M7: mediana vs MICE

### 7.2 Pesos muestrales

- Re-estimar M7 con `SAMPWEIGHT_R1`
- Comparar coeficientes con versión sin pesos (Solon et al. 2015)

### 7.3 Target alternativo

- `Y_alt = PERSISTENT_18_23` = 1 si la persona reportó agresión en **2 o más años** entre 18-23
- Re-estimar M7 con `Y_alt`
- Comparar top variables vs target original

### 7.4 Subgrupos

- Re-estimar M7 estratificado por sexo (hombres / mujeres)
- Re-estimar M7 estratificado por raza/etnia
- Mostrar si los predictores top son los mismos

### 7.5 Test de subgrupos via interacciones

En M7, agregar interacciones `EARLY_ASSAULT × SEX` y `EARLY_ASSAULT × RACE_ETHNICITY`. Wald test de los términos de interacción.

---

## 8. Estructura del paper rediseñado

```
1. Introducción y mérito intelectual (1-2 pp)
2. Datos y variables (2 pp)
   2.1 Fuente y muestra
   2.2 Variable dependiente
   2.3 Bloques de variables explicativas (los 7 bloques)
3. Estrategia empírica (2-3 pp)
   3.1 Regresión logística jerárquica con 7 modelos anidados
   3.2 Deviance / Likelihood Ratio test entre modelos
   3.3 Random Forest con permutation importance individual y por bloque
   3.4 Triangulación entre métodos
4. Resultados (2-3 pp)
   4.1 Estadística descriptiva (Tabla 1)
   4.2 Tabla de coeficientes anidados (Tabla 2)
   4.3 Aporte incremental por bloque (Tabla 3, ANOVA)
   4.4 Permutation importance del RF (Figuras 2-3)
   4.5 Comparación de rankings entre métodos (Figura 4)
5. Discusión y política pública (1-2 pp)
6. Conclusión y limitaciones (1 pp)
7. Anexo
   A. Estadística descriptiva extendida
   B. VIF y diagnósticos
   C. Robustez (MICE, pesos, target alternativo, subgrupos)
   D. Coefficient stability plot
   E. Reliability diagrams (calibración)
```

---

## 9. Output esperado

### Tablas
- **Tabla 1:** Descriptiva por outcome
- **Tabla 2:** Coeficientes anidados M1-M7
- **Tabla 3:** Aporte por bloque (ANOVA LR test)
- **Tabla 4:** Comparación ranking logit vs RF

### Figuras
- **Figura 1:** Coefficient stability plot
- **Figura 2:** Permutation importance individual (top 15)
- **Figura 3:** Permutation importance por bloque (7 bloques)
- **Figura 4:** Ranking comparison slope chart

### Anexo
- Tablas A-E (robustez)
- Figuras A-C (diagnósticos)

---

## 10. Cosas que NO se hacen

Para evitar scope creep, se descartan explícitamente:

- ❌ XGBoost / Gradient Boosting (overkill para la pregunta)
- ❌ Causal Forest / Double Machine Learning (no es el foco; el diseño es asociativo)
- ❌ SHAP values (permutation importance basta)
- ❌ Decision Curve Analysis (no es el foco)
- ❌ Conformal Prediction (no es el foco)
- ❌ Análisis multinivel / longitudinal (unidad = persona, no persona-año)
- ❌ Reportar Gini importance del RF (sesgada; usar solo permutation)
- ❌ Features de trayectoria temporal (age_first, duration) — colapsar a `ever`

---

## 11. Stack técnico

- **Python 3.11 + conda env `rappi`**
- `pandas`, `numpy` — manipulación de datos
- `statsmodels` — regresión logística, LR tests, ANOVA, VIF
- `scikit-learn` — Random Forest, permutation_importance, train/test split, CV
- `scipy.stats` — chi-cuadrado, test t, distribución χ²
- `matplotlib` + `seaborn` — figuras
- `pyreadstat` / `pandas.read_stata` — lectura de `.dta`

Validación cruzada de resultados clave en R (`stats::glm`, `anova`) opcional.

---

## 12. Plan de implementación (en orden)

1. **Preparación de datos** (1 día)
   - Construir features de los 7 bloques sobre el parquet existente
   - Discretizar continuas en cuartiles
   - Generar dummies de categóricas
   - Validar coberturas, missingness

2. **Estimación de los 7 modelos logísticos** (1 día)
   - Estimar M1 a M7 secuencialmente
   - Generar Tabla 2 (coeficientes anidados)
   - Calcular LR tests, AIC, pseudo-R²
   - Generar Tabla 3 (ANOVA)

3. **Random Forest + permutation importance** (1 día)
   - Estimar RF con CV
   - Permutation individual (top 15)
   - Permutation por bloque (7 bloques)
   - Generar Figuras 2-3

4. **Triangulación y figuras finales** (0.5 día)
   - Tabla 4 (comparación rankings)
   - Figura 1 (coefficient stability)
   - Figura 4 (slope chart)

5. **Robustez** (1 día)
   - MICE
   - Pesos muestrales
   - Target alternativo
   - Subgrupos

6. **Redacción del paper** (2-3 días)

**Total estimado: ~7-8 días de trabajo.**
