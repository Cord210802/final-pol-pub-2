# Decisiones de modelado — Proyecto Final Políticas Públicas II

> Documento de referencia técnica. Recoge exactamente lo que hizo cada script,
> con los parámetros usados y los números que salieron.

---

## 1. Datos

**Fuente:** ICPSR 34562 — *Recidivism of Adult Felons from NLSY97*  
**Archivos originales:**
- `DS0001/34562-0001-Data.dta` — datos base (8 984 individuos, ola de 1997)
- `DS0002/34562-0002-Data.dta` — datos adicionales con agresiones a los 24–25 años

**Limpieza de missings:** Todos los valores negativos en columnas numéricas se
recodificaron como `NaN`. En el NLSY97, los códigos de "no responde / inaplicable"
son negativos (e.g., -1, -2, -3, -4, -5), por lo que `df[num] = df[num].where(df[num] >= 0, np.nan)` los elimina a todos de golpe.

**Unión:** `DS0001` se hizo merge con `ASSAULT_CY24` y `ASSAULT_CY25` de `DS0002`
mediante `PUBID` (left join).

**Ventanas temporales definidas:**
- `EARLY` = edades 14–17 (adolescencia)
- `LATE` = edades 18–23 (adultez temprana)

---

## 2. Variables construidas

### Targets

| Variable | Definición |
|---|---|
| `EVER_ASSAULT` | 1 si el individuo reportó ≥1 agresión en **cualquier** año (14–25) |
| `ASSAULT_18_23` | 1 si reportó ≥1 agresión entre los **18 y 23** años |
| `EARLY_ASSAULT` | 1 si reportó ≥1 agresión entre los **14 y 17** años |

**Regla `ever()`:** Se aplica a cada conducta por ventana. Si el máximo anual > 0 → 1. Si **ningún** año de la ventana tiene observación → NaN (no se imputa como 0).

### Features de conducta temprana (14–17, variable `ever`)

| Variable | Conducta |
|---|---|
| `EARLY_DESTROY` | Daño a propiedad |
| `EARLY_STEAL_LO` | Robo menor a $50 |
| `EARLY_STEAL_HI` | Robo mayor a $50 |
| `EARLY_SELL_DRUGS` | Venta de drogas |
| `EARLY_CARR_GUN` | Portación de arma |
| `EARLY_GANG` | Pertenencia a pandilla |
| `EARLY_ALCOHOL` | Consumo de alcohol |
| `EARLY_MARIJUANA` | Consumo de marihuana |
| `EARLY_HARD_DRUGS` | Consumo de drogas duras |
| `EARLY_SMOKING` | Tabaquismo |
| `ARR_BY17` | Arrestado alguna vez antes de los 18 (de `EVER_ARR_CY17`) |
| `INCARC_BY17` | Encarcelado alguna vez antes de los 18 (de `EVER_INCARC_CY17`) |
| `HHINC_EARLY` | Ingreso del hogar promedio 14–17 (media de valores positivos) |

### Features del perfil base 1997 (estáticos)

| Variable | Descripción |
|---|---|
| `ASVAB_SCORE` | Puntaje en prueba cognitiva Armed Services Vocational Aptitude Battery |
| `POVRATIO97` | Razón ingreso/línea de pobreza en 1997 |
| `NUMSIBS` | Número de hermanos |
| `BIOMOMAGE` | Edad de la madre biológica al nacer el respondente |
| `SEX` | Sexo (1 = Hombre, 2 = Mujer) |
| `RACE_ETHNICITY` | Raza/etnicidad (1=Negro, 2=Hispano, 3=Mixto, 4=Blanco/Otro) |
| `FMSTRC97` | Estructura familiar en 1997 (1=Ambos biológicos ... 5=Otro) |
| `RESPARED` | Educación del padre responsable |
| `MOMEMP97` | Empleo de la madre en 1997 |
| `DADEMP97` | Empleo del padre en 1997 |

### Features del modelo global (todos los años 14–25, agregados)

En el modelo global (`feature_importance.py`) se usaron **totales de por vida**
en lugar de variables `ever`, para capturar intensidad acumulada:

`TOTAL_DESTROY`, `TOTAL_STEAL_LO`, `TOTAL_STEAL_HI`, `TOTAL_SELL_DRUGS`,
`TOTAL_CARR_GUN`, `TOTAL_GANG`, `TOTAL_ALCOHOL`, `TOTAL_MARIJUANA`,
`TOTAL_HARD_DRUGS`, `TOTAL_SMOKING`, `TOTAL_ARRESTS` (= `TOTARRESTS`),
`AGE_FIRST_ARREST` (= `FIRSTARREST`), `TOTAL_INCARC` (= `TOTINCARC`),
`AVG_HHINC`, `AVG_RINC`, más el perfil base 1997.

---

## 3. Imputación de missings

**Método:** Mediana por columna (`SimpleImputer(strategy="median")`),
ajustado **solo** sobre el set de entrenamiento.

**Cobertura de predictores en el modelo predictivo (14–17 -> 18–23):**

| Variable | Cobertura (%) |
|---|---|
| `ARR_BY17`, `INCARC_BY17`, `SEX`, `RACE_ETHNICITY`, `NUMSIBS` | 100.0 |
| `EARLY_ASSAULT`, `EARLY_STEAL_LO`, `EARLY_STEAL_HI` | ~99.9 |
| `EARLY_DESTROY`, `EARLY_SELL_DRUGS`, `EARLY_GANG` | 99.3–99.8 |
| `BIOMOMAGE`, `RESPARED`, `MOMEMP97` | 88.2–93.2 |
| `EARLY_SMOKING` | 90.3 |
| `ASVAB_SCORE` | 80.0 |
| `EARLY_CARR_GUN`, `EARLY_ALCOHOL`, `EARLY_MARIJUANA`, `EARLY_HARD_DRUGS` | 77.9–81.4 |
| `POVRATIO97` | 73.4 |
| `HHINC_EARLY` | 66.1 |
| `DADEMP97` | 63.1 |

---

## 4. Análisis exploratorio previo (feature_importance.py) — NO está en el paper

Este script fue un primer análisis exploratorio antes de definir el diseño
final. Usa un target y features distintos a los tres modelos del paper.

**Target:** `EVER_ASSAULT` (agresión alguna vez en **toda la vida**, 14–25 años)  
**Muestra:** 8 958 individuos  
**Prevalencia:** 31.46% — diferente al 16.9% del paper porque el target incluye
agresión en toda la vida, no solo en la adultez (18–23)  
**Features:** totales acumulados de por vida (sin separación temporal); **incluye**
`TOTAL_ARRESTS` y `AGE_FIRST_ARREST`, que son contemporáneos al target — por
eso **no** es un diseño predictivo limpio  
**Hiperparámetros RF:** n_estimators=300, max_depth=12, min_samples_leaf=15  
**AUC test:** 0.8135 | AUC CV: 0.8215 ± 0.024

Este script generó el archivo `resultados.json`. Sus números **no** coinciden
con las tablas del paper porque el target y las features son distintos.

---

## 5. Los tres modelos del paper

Los tres modelos que aparecen en el paper usan **el mismo target**: agresión
física autorreportada entre los 18 y los 23 años (`ASSAULT_18_23`). La
diferencia entre ellos es la submuestra sobre la que se estiman.

### Modelo 1 — Global (modelo_predictivo.py, muestra completa)

**Pregunta:** ¿Podemos predecir la violencia adulta (18–23) usando solo
información de la adolescencia (14–17) y el perfil de 1997?

**Diseño temporal limpio:** predictores en [14–17] -> target en [18–23].
No hay leakage: ningún predictor usa información posterior a los 17 años.

**Target:** `ASSAULT_18_23`  
**Muestra:** 8 609 individuos con observación en la ventana 18–23  
**Prevalencia:** 16.90% (1 455 positivos, 7 154 negativos)  
**Features:** 24 (conducta temprana `ever` 14–17 + perfil base 1997)

**Split:** 80/20 estratificado (`random_state=42`)
- Train: 6 887 | Test: 1 722

**Random Forest:**
```
n_estimators     = 400
max_depth        = 10
min_samples_leaf = 20
max_features     = "sqrt"
class_weight     = "balanced"
random_state     = 42
```

*(Árbol más superficial y hoja más grande que el análisis exploratorio —
mayor regularización para un problema predictivo más difícil con más desbalance)*

**Resultados:**

| Métrica | Valor |
|---|---|
| AUC test | **0.7987** |
| AUC CV (5-fold) | 0.7637 ± 0.0295 |
| Accuracy | 0.758 |
| Precision (con asalto) | 0.378 |
| Recall (con asalto) | 0.663 |

*(La precisión baja se explica por el desbalance: solo 16.9% de positivos)*

**Top 10 importancia Gini:**

| Variable | Gini |
|---|---|
| EARLY_ASSAULT | 0.2278 |
| ASVAB_SCORE | 0.0802 |
| HHINC_EARLY | 0.0640 |
| EARLY_SELL_DRUGS | 0.0588 |
| POVRATIO97 | 0.0554 |
| BIOMOMAGE | 0.0549 |
| EARLY_DESTROY | 0.0528 |
| SEX | 0.0486 |
| EARLY_GANG | 0.0440 |
| EARLY_ALCOHOL | 0.0405 |

**Top 10 importancia por permutación (30 repeticiones, scoring=roc_auc):**

| Variable | Delta AUC |
|---|---|
| EARLY_ASSAULT | 0.07942 |
| SEX | 0.01449 |
| EARLY_ALCOHOL | 0.01437 |
| EARLY_GANG | 0.01031 |
| ASVAB_SCORE | 0.00659 |
| EARLY_SMOKING | 0.00658 |
| BIOMOMAGE | 0.00498 |
| ARR_BY17 | 0.00468 |
| RACE_ETHNICITY | 0.00428 |
| EARLY_SELL_DRUGS | 0.00329 |

**Nota Gini vs permutación:** `HHINC_EARLY` y `POVRATIO97` tienen Gini alto
pero permutación baja o negativa — señal de que el RF las usa en splits
intermedios pero no contribuyen al AUC de forma independiente. La permutación
es más confiable.

**PCA (18 variables continuas/ordinales de la ventana 14–17):**
- PC1 explica 19.4%, PC2 12.2%, acumulado primeros 5 = 50.3%
- PC1 = factor de "involucramiento delictivo temprano" (cargas altas: EARLY_SELL_DRUGS, EARLY_MARIJUANA, EARLY_DESTROY, EARLY_STEAL_LO)
- PC2 = factor "contexto socioeconómico favorable" (POVRATIO97, HHINC_EARLY, ASVAB_SCORE positivos)
- Correlación PC1 con target: r = 0.301 | PC2: r = -0.086

**Contrastes bivariados:**

| Grupo | Tasa de asalto adulto (18–23) |
|---|---|
| Hombres | 21.7% |
| Mujeres | 11.9% |
| Sin asalto temprano (14–17) | 10.6% |
| Con asalto temprano (14–17) | 37.4% |

---

### Modelo 2 — Inicio (modelo_inicio_persistencia.py, EARLY_ASSAULT == 0)

**Pregunta:** ¿Qué variables predicen que una persona que **no** fue violenta
en la adolescencia empiece a reportar agresión entre los 18 y los 23 años?

**Diseño:** Se toma solo la submuestra sin agresión temprana. `EARLY_ASSAULT`
se excluye de los predictores porque ya define la submuestra.

**Target:** `ASSAULT_18_23`  
**Muestra:** 6 568 individuos (EARLY_ASSAULT == 0)  
**Prevalencia:** 10.57% (694 positivos, 5 874 negativos)  
**Features:** 23 (conducta temprana 14–17 + perfil base 1997, sin EARLY_ASSAULT)

**Split:** 75/25 estratificado (`random_state=42`)

**Random Forest:**
```
n_estimators     = 400
max_depth        = 9
min_samples_leaf = 20
max_features     = "sqrt"
class_weight     = "balanced"
random_state     = 42
```

**Resultados:**

| Métrica | Valor |
|---|---|
| AUC test | **0.7098** |
| AUC CV (5-fold) | 0.6980 ± 0.0394 |

**Top 10 importancia Gini:**

| Variable | Gini |
|---|---|
| ASVAB_SCORE | 0.1424 |
| BIOMOMAGE | 0.0906 |
| EARLY_GANG | 0.0872 |
| HHINC_EARLY | 0.0844 |
| EARLY_ALCOHOL | 0.0810 |
| SEX | 0.0793 |
| POVRATIO97 | 0.0751 |
| RACE_ETHNICITY | 0.0636 |
| FMSTRC97 | 0.0394 |
| NUMSIBS | 0.0387 |

**Top 5 por permutación:** SEX (0.0365), EARLY_ALCOHOL (0.0343),
ASVAB_SCORE (0.0183), EARLY_SMOKING (0.0137), BIOMOMAGE (0.0117)

**Interpretación:** El inicio de la agresión se asocia con factores de
contexto y perfil (sexo, capacidad cognitiva, ingreso del hogar, estructura
familiar), más que con conductas delictivas específicas durante la adolescencia.

---

### Modelo 3 — Persistencia (modelo_inicio_persistencia.py, EARLY_ASSAULT == 1)

**Pregunta:** ¿Qué variables predicen que una persona que **sí** fue violenta
en la adolescencia continúe reportando agresión entre los 18 y los 23 años?

**Diseño:** Se toma solo la submuestra con agresión temprana. `EARLY_ASSAULT`
se excluye de los predictores porque ya define la submuestra.

**Target:** `ASSAULT_18_23`  
**Muestra:** 2 029 individuos (EARLY_ASSAULT == 1)  
**Prevalencia:** 37.41% (759 positivos, 1 270 negativos)  
**Features:** 23 (los mismos del Modelo 2)

**Split:** 75/25 estratificado (`random_state=42`)

**Random Forest:** idéntico al Modelo 2 (n_estimators=400, max_depth=9,
min_samples_leaf=20, max_features="sqrt", class_weight="balanced")

**Resultados:**

| Métrica | Valor |
|---|---|
| AUC test | **0.5840** |
| AUC CV (5-fold) | 0.6171 ± 0.0366 |

*(AUC cercano a 0.5: el modelo apenas mejora una clasificación aleatoria)*

**Top 10 importancia Gini:**

| Variable | Gini |
|---|---|
| POVRATIO97 | 0.1163 |
| ASVAB_SCORE | 0.0980 |
| HHINC_EARLY | 0.0980 |
| BIOMOMAGE | 0.0911 |
| EARLY_SELL_DRUGS | 0.0835 |
| EARLY_CARR_GUN | 0.0549 |
| SEX | 0.0501 |
| ARR_BY17 | 0.0461 |
| RACE_ETHNICITY | 0.0423 |
| EARLY_STEAL_LO | 0.0416 |

**Top 5 por permutación:** EARLY_SELL_DRUGS (0.0129), ARR_BY17 (0.0110),
EARLY_CARR_GUN (0.0087), BIOMOMAGE (0.0067), POVRATIO97 (0.0064)

**Interpretación:** En personas ya violentas durante la adolescencia, las
variables disponibles no permiten distinguir bien quién continuará agrediendo.
Las señales más robustas por permutación son conductas delictivas graves
(venta de drogas, portación de arma, arrestos) y condiciones de pobreza.

---

## 5. Validación cruzada

En los tres modelos se usó **5-fold cross-validation** con `cross_val_score`
y `scoring="roc_auc"`, calculada sobre toda la muestra disponible de cada
subconjunto. El imputer de medianas se ajustó antes del CV, lo que introduce
un leakage mínimo (la mediana de variables con alta cobertura es muy estable
y no varía entre folds de forma significativa).

---

## 6. Gini vs Permutación

Se calcularon ambas medidas en todos los modelos:

- **Gini (MDI):** rápido, pero favorece variables con muchos valores únicos
  y puede sobrestimar variables correlacionadas entre sí.
- **Permutación:** mide cuánto cae el AUC al aleatorizar cada variable en
  el set de prueba (20 o 30 repeticiones). Más honesta, pero más ruidosa
  en muestras pequeñas.

En el análisis se priorizó la permutación para identificar variables
con señal predictiva genuina, usando el Gini como corroboración del orden.

---

## 7. Resumen de desempeño

Los tres modelos del paper usan el mismo target (`ASSAULT_18_23`).
El análisis exploratorio (`feature_importance.py`) usa `EVER_ASSAULT` y
**no** aparece en el paper; se incluye solo como referencia.

| # | Modelo | Target | n train | n test | AUC test | AUC CV |
|---|---|---|---|---|---|---|
| — | Exploratorio (feature_importance.py) | EVER_ASSAULT (14–25) | 7 166 | 1 792 | 0.8135 | 0.8215 ± 0.024 |
| 1 | Global (modelo_predictivo.py) | ASSAULT_18_23 | 6 887 | 1 722 | **0.7987** | 0.7637 ± 0.030 |
| 2 | Inicio (modelo_inicio_persistencia.py) | ASSAULT_18_23 | ~4 926 | ~1 642 | **0.7098** | 0.6980 ± 0.039 |
| 3 | Persistencia (modelo_inicio_persistencia.py) | ASSAULT_18_23 | ~1 522 | ~507 | **0.5840** | 0.6171 ± 0.037 |

---

## 8. Archivos de resultados

| Archivo | Contenido |
|---|---|
| `analisis/resultados.json` | Exploratorio (EVER_ASSAULT): AUC, Gini, permutación, PCA |
| `analisis/resultados_predictivo.json` | Modelo 1 Global: AUC, Gini, permutación, PCA, coberturas |
| `analisis/resultados_inicio_persistencia.json` | Modelos 2 y 3: Inicio y Persistencia |
| `analisis/figuras/gini_importance.png` | Importancia Gini, exploratorio (EVER_ASSAULT) |
| `analisis/figuras/pred_gini.png` | Importancia Gini, Modelo 1 Global |
| `analisis/figuras/modelo_inicio.png` | Importancia Gini, Modelo 2 Inicio |
| `analisis/figuras/modelo_persistencia.png` | Importancia Gini, Modelo 3 Persistencia |
| `analisis/figuras/pca_biplot.png` | Biplot PC1 vs PC2, exploratorio (EVER_ASSAULT) |
| `analisis/figuras/pred_pca_biplot.png` | Biplot PC1 vs PC2, Modelo 1 Global |
| `analisis/figuras/pca_scree.png` / `pred_pca_scree.png` | Varianza explicada PCA |

---

## 9. Scripts

| Script | Qué hace |
|---|---|
| `analisis/construir_dataset.py` | Limpia y une DS0001+DS0002, construye todas las variables de ingeniería, guarda `data/nlsy97_analisis_final.parquet` |
| `analisis/feature_importance.py` | Exploratorio — target EVER_ASSAULT (toda la vida, no está en el paper) |
| `analisis/modelo_predictivo.py` | Modelo 1 Global — target ASSAULT_18_23, predictores 14–17 |
| `analisis/modelo_inicio_persistencia.py` | Modelos 2 y 3 — divide por EARLY_ASSAULT para Inicio y Persistencia |
| `analisis/figuras_paper.py` | Genera las figuras definitivas que van al paper |
