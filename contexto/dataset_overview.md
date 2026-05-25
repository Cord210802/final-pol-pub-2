# ICPSR 34562 — Recidivism in the NLSY97: Descripción del Dataset

## 1. ¿Qué es este dataset?

**Nombre oficial:** Recidivism in the National Longitudinal Survey of Youth 1997 – Standalone Data (Rounds 1 to 13)  
**Fuente:** Bureau of Justice Statistics (BJS) / ICPSR  
**DOI:** http://doi.org/10.3886/ICPSR34562.v1  
**Publicado:** 2014-02-06  

### Origen de los datos

El dataset es un **subconjunto derivado del NLSY97** (National Longitudinal Survey of Youth, cohorte 1997), una encuesta panel financiada por el Bureau of Labor Statistics. El NLSY97 reclutó a 8,984 residentes estadounidenses nacidos entre **1980 y 1984** y los ha seguido anualmente desde 1997.

El BJS extrajo de ese panel las variables relevantes para el estudio de **reincidencia criminal**, enriqueciendo los datos de autoinforme con información de arresto, encarcelamiento y condenas. La cohorte tenía **14 años en 1997** y llegó a **25 años** en la última ronda cubierta.

### ¿Por qué es valioso?

A diferencia de los registros administrativos, este dataset combina:
- **Autoreporte de conductas delictivas** (crímenes que nunca llegaron al sistema formal)
- **Trayectoria socioeconómica completa** (empleo, ingresos, educación, familia)
- **Variables de contexto familiar** medidas en la adolescencia temprana
- **Seguimiento longitudinal** desde los 14 hasta los 25 años

---

## 2. Estructura general

| Archivo | Individuos | Variables | Edades cubiertas | Descripción |
|---------|-----------|-----------|-----------------|-------------|
| `DS0001/34562-0001-Data.dta` | 8,984 | 392 | 14–23 años (Rounds 1–7) | Muestra completa |
| `DS0002/34562-0002-Data.dta` | 2,977 | 405 | 24–25 años (Rounds 8–13) | Submuestra: quienes fueron entrevistados en prisión |

Las dos tablas comparten `PUBID` como llave. DS0002 añade las edades 24–25 para una submuestra de individuos encarcelados, más dos variables de diseño muestral (`CUSTOMWEIGHT`, `CONTROL_GROUP`).

---

## 3. Convención de nombres

La mayoría de las variables siguen el patrón:

```
NOMBRE_VARIABLE_CY{edad}
```

`CY` = "Calendar Year age". El número que sigue es la **edad del respondente** cuando fue observado esa ronda, no el año calendario.

| Columna | Significado |
|---------|-------------|
| `ASSAULT_CY14` | ¿Cometió una agresión cuando tenía 14 años? |
| `ASSAULT_CY17` | ¿Cometió una agresión cuando tenía 17 años? |
| `HHINC_CY20`   | Ingreso del hogar cuando el respondente tenía 20 años |
| `EVER_ARR_CY16`| ¿Fue arrestado alguna vez antes o durante los 16 años? |

Las variables **sin sufijo** `_CY{n}` son estáticas (medidas una sola vez, generalmente en 1997):

```
SEX, RACE_ETHNICITY, BIOMOMAGE, NUMSIBS, FMSTRC97, RESPARED, ASVAB_SCORE, POVRATIO97 ...
```

---

## 4. Códigos de valor faltante

El NLSY97 usa códigos especiales que aparecen como **categorías** en Stata (`.dta`):

| Etiqueta en el archivo | Significado |
|------------------------|-------------|
| `Non-interview` | El respondente no fue entrevistado ese año |
| `Not interviewed at that age` | Demasiado joven / fuera de rango |
| `Refused` | Se negó a responder |
| `Don't know` | No sabe / no recuerda |
| `Valid missing` | Pregunta no aplicable (skip lógico) |

**Todos estos valores deben tratarse como `NaN`** antes de cualquier análisis numérico. En el notebook se aplica:

```python
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].where(df[num_cols] >= 0, np.nan)
```

### Ejemplo real — `ASSAULT_CY17`

```
No               7,492  (83.4%)
Yes                907  (10.1%)
Non-interview      554   (6.2%)
Refused             23   (0.3%)
Don't know           6   (0.1%)
Valid missing        2   (<0.1%)
```

---

## 5. Variables estáticas (demográficas / baseline 1997)

Estas variables fueron medidas **una sola vez** al inicio del estudio.

| Variable | Descripción | Rango / Categorías |
|----------|-------------|-------------------|
| `PUBID` | Identificador único del respondente | 1 – 8,984 |
| `SEX` | Sexo biológico | Male (4,599) / Female (4,385) |
| `RACE_ETHNICITY` | Raza/etnicidad autodefinida | Non-black/non-Hispanic (4,665), Black (2,335), Hispanic (1,901), Mixed (83) |
| `BIOMOMAGE` | Edad de la madre biológica al nacer el respondente | 12–54, media ≈ 25.5 |
| `NUMSIBS` | Número de hermanos en 1997 | 0–13, media ≈ 1.6 |
| `MOMEMP97` | Status laboral de la madre en 1997 | Categorical |
| `DADEMP97` | Status laboral del padre en 1997 | Categorical |
| `FMSTRC97` | Estructura familiar en 1997 | Ver tabla abajo |
| `RESPARED` | Máximo nivel educativo de los padres | Ver tabla abajo |
| `POVRATIO97` | Razón ingreso/línea de pobreza en 1997 | 0–16.3, media ≈ 2.83, 27% NaN |
| `ASVAB_SCORE` | Percentil en el Armed Services Vocational Aptitude Battery (proxy de habilidad cognitiva) | 0–100, media ≈ 45, 21% NaN |
| `HEALTHCONDITIONS16` | Condiciones de salud crónicas a los 16 | Categorical |
| `HEALTHLIMIT2002` | Limitaciones de salud en 2002 | Categorical |
| `BDATE_Y` | Año de nacimiento | 1980–1984 |
| `BDATE_M` | Mes de nacimiento | 1–12 |
| `SAMPWEIGHT_R1` | Peso muestral ronda 1 (para inferencia poblacional) | Continuo |

### Estructura familiar (`FMSTRC97`)

| Categoría | N |
|-----------|---|
| Two biological parents | 4,395 |
| One single biologic/adoptive parent | 2,826 |
| One biological + one step-parent | 1,205 |
| Other family type | 424 |
| Two adoptive parents | 103 |

### Educación de los padres (`RESPARED`)

| Categoría | N |
|-----------|---|
| College degree or higher | 2,069 |
| Some college | 2,127 |
| Graduated from HS | 2,617 |
| Zero through 11th grade | 1,538 |
| Missing | 633 |

---

## 6. Variables de conducta delictiva (autoreporte por edad)

Todas son **binarias (Sí/No)** por edad. Cubren edades 14–23 en DS0001 y 24–25 en DS0002.

| Variable | Descripción |
|----------|-------------|
| `DESTROY_CY{n}` | ¿Destruyó intencionalmente propiedad ajena? |
| `STEAL_UNDER_50_CY{n}` | ¿Robó algo de valor < $50? |
| `STEAL_OVER_50_CY{n}` | ¿Robó algo de valor ≥ $50? |
| `OTHER_PROPERTY_CY{n}` | ¿Cometió otro delito contra la propiedad? |
| `SELL_DRUGS_CY{n}` | ¿Vendió drogas ilegales? |
| `ASSAULT_CY{n}` | ¿Atacó físicamente a alguien con intención de herir? **(variable objetivo)** |

### Ejemplo: trayectoria de ASSAULT para el respondente #101

| Edad | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|------|----|----|----|----|----|----|----|----|----|----|
| ASSAULT | NI | NI | No | No | No | No | No | No | No | No |

*(NI = No interviewed at that age)*

---

## 7. Variables de estilo de vida y factores de riesgo (por edad)

Binarias salvo indicación. Cubren diferentes rangos de edad.

| Variable | Edades | Descripción |
|----------|--------|-------------|
| `IN_A_GANG_CY{n}` | 14–25 | ¿Pertenecía a una pandilla? |
| `GANGS_CY{n}` | 14–25 | ¿Tuvo contacto con pandilleros? (versión alternativa) |
| `CARR_GUN_CY{n}` | 14–25 | ¿Portó un arma de fuego ilegalmente? |
| `SMOKING_CY{n}` | 14–25 | ¿Fumaba cigarrillos? |
| `ALCOHOL_CY{n}` | 14–25 | ¿Consumió alcohol? |
| `MARIJUANA_CY{n}` | 14–25 | ¿Consumió mariguana? |
| `HARD_DRUGS_CY{n}` | 14–25 | ¿Usó drogas duras (cocaína, heroína, etc.)? |
| `RUNAWAY_CY{n}` | 14–18 | ¿Se escapó de casa? |
| `WEIGHT_CY{n}` | 14, 19, 23 | Peso corporal (lbs) |
| `HEIGHT_CY{n}` | 14, 19, 23 | Estatura (pulgadas) |

---

## 8. Variables del sistema de justicia criminal (por edad)

| Variable | Descripción |
|----------|-------------|
| `EVER_ARR_CY{n}` | ¿Fue arrestado alguna vez hasta esa edad? (acumulativo) |
| `MOS_INCARC_CY{n}` | Meses encarcelado durante ese año de vida |
| `EVER_INCARC_CY{n}` | ¿Fue encarcelado alguna vez hasta esa edad? (acumulativo) |
| `EVER_CONV_CY{n}` | ¿Fue convicto alguna vez hasta esa edad? (acumulativo) |
| `NI_INCARC_CY{n}` | ¿No fue entrevistado por estar encarcelado? |
| `DEAD_CY{n}` | ¿Estaba muerto a esa edad? |

### Variables de resumen (sin sufijo de edad)

| Variable | Descripción | Estadísticas |
|----------|-------------|-------------|
| `TOTARRESTS` | Total de arrestos en toda la vida (14–25) | 0–63, media ≈ 1.14 |
| `FIRSTARREST` | Fecha del primer arresto (YYYYMM) | 198703–200912, 67% NaN (nunca arrestados) |
| `TOTINCARC` | Total de encarcelamientos | 0–7, media ≈ 0.13 |
| `AGE_FIRSTINCARC` | Edad al primer encarcelamiento | 11–28, 92.7% NaN (nunca encarcelados) |
| `FIRSTINCARC` | Fecha del primer encarcelamiento | Similar a FIRSTARREST |

> **Nota:** El alto porcentaje NaN en `FIRSTARREST` y `AGE_FIRSTINCARC` no es un problema de calidad de datos. Refleja que la mayoría (≈67%) **nunca fue arrestada**.

---

## 9. Variables socioeconómicas dinámicas (por edad)

| Variable | Edades | Descripción |
|----------|--------|-------------|
| `RINC_CY{n}` | 14–25 | Ingreso personal del respondente (USD) |
| `HHINC_CY{n}` | 14–25 | Ingreso total del hogar (USD) |
| `WKS_WK_CY{n}` | 14–25 | Semanas trabajadas ese año |
| `AVG_HRS_WK_CY{n}` | 14–25 | Promedio de horas trabajadas por semana |
| `AVG_HRS_WK_WK_CY{n}` | 14–25 | Horas promedio (versión alternativa de cálculo) |

---

## 10. Variables educativas (por edad)

| Variable | Edades | Descripción |
|----------|--------|-------------|
| `ENROLL_ED_CY{n}` | 14–25 | ¿Estaba inscrito en algún sistema educativo? |
| `HGC_CY{n}` | 14–25 | Highest Grade Completed (grado más alto completado) |
| `HDR_CY{n}` | 14–25 | ¿Tiene diploma de preparatoria? |

---

## 11. Variables familiares (por edad)

| Variable | Edades | Descripción |
|----------|--------|-------------|
| `MARSTAT_CY{n}` | 14–25 | Estado civil |
| `BKIDS_HH_CY{n}` | 14–25 | Número de hijos biológicos viviendo en el hogar |
| `BKIDS_NR_CY{n}` | 14–25 | Número de hijos biológicos fuera del hogar |
| `BKIDS_TTL_CY{n}` | 14–25 | Total de hijos biológicos |

---

## 12. Variables exclusivas de DS0002 (edades 24–25)

DS0002 es una **submuestra de 2,977 individuos** entrevistados en rondas 8–13, la mayoría estando encarcelados. Añade:

| Variable | Descripción |
|----------|-------------|
| `ASSAULT_CY24`, `ASSAULT_CY25` | Agresiones a los 24–25 (extienden DS0001) |
| `DESTROY_CY24/25`, `STEAL_*_CY24/25`, etc. | Mismo patrón para todos los delitos |
| `CONTROL_GROUP` | Indica si el individuo es parte del grupo de control del diseño de sobremuestra |
| `CUSTOMWEIGHT` | Peso muestral ajustado para esta submuestra |

---

## 13. Variable objetivo construida para el análisis

En el notebook creamos una variable sintética a partir del dataset:

```python
assault_cols = [f'ASSAULT_CY{age}' for age in range(14, 26)]

df['TOTAL_ASSAULTS'] = df[assault_cols].sum(axis=1, min_count=1)
df['EVER_ASSAULT']   = (df['TOTAL_ASSAULTS'] > 0).astype(int)
```

| `EVER_ASSAULT` | Significado | N aproximado |
|---------------|-------------|-------------|
| `1` | Reportó ≥1 agresión en cualquier edad (14–25) | ~2,100 |
| `0` | Nunca reportó haber agredido a alguien | ~5,000 |
| `NaN` | Sin datos válidos de agresión en ninguna edad | Resto |

**Desbalance de clases:** ~70% negativo vs ~30% positivo. Se usa `class_weight='balanced'` en el modelo.

---

## 14. Variables de features usadas en el análisis

Para evitar **data leakage**, las variables de `ASSAULT_CY{n}` se excluyen completamente de los features. Se usan agregados por toda la vida:

| Feature creado | Fuente |
|---------------|--------|
| `TOTAL_DESTROY` | Suma de `DESTROY_CY14`–`DESTROY_CY23` |
| `TOTAL_SELL_DRUGS` | Suma de `SELL_DRUGS_CY14`–`SELL_DRUGS_CY23` |
| `TOTAL_CARR_GUN` | Suma de `CARR_GUN_CY14`–`CARR_GUN_CY23` |
| `TOTAL_GANG` | Suma de `GANGS_CY14`–`GANGS_CY23` |
| `TOTAL_ARRESTS` | = `TOTARRESTS` |
| `AGE_FIRST_ARREST` | = `FIRSTARREST` (fecha YYYYMM) |
| `TOTAL_ALCOHOL` | Suma de `ALCOHOL_CY14`–`ALCOHOL_CY23` |
| `TOTAL_MARIJUANA` | Suma de `MARIJUANA_CY14`–`MARIJUANA_CY23` |
| `AVG_HHINC` | Promedio de `HHINC_CY14`–`HHINC_CY23` |
| `ASVAB_SCORE` | Percentil cognitivo (baseline) |
| `SEX`, `RACE_ETHNICITY`, `FMSTRC97`, `RESPARED` | Variables demográficas estáticas |
| `BIOMOMAGE`, `NUMSIBS`, `POVRATIO97` | Contexto familiar baseline |

---

## 15. Resumen estadístico de variables clave

| Variable | Min | Max | Media | NaN% |
|----------|-----|-----|-------|------|
| `ASVAB_SCORE` | 0 | 100 | 45.3 | 21.0% |
| `POVRATIO97` | 0 | 16.3 | 2.83 | 27.0% |
| `NUMSIBS` | 0 | 13 | 1.55 | 0.0% |
| `BIOMOMAGE` | 12 | 54 | 25.5 | 6.8% |
| `TOTARRESTS` | 0 | 63 | 1.14 | 0.0% |
| `TOTINCARC` | 0 | 7 | 0.13 | 0.0% |
| `AGE_FIRSTINCARC` | 11 | 28 | 20.8 | 92.7%* |
| `FIRSTARREST` | 1987-03 | 2009-12 | 2000-01 | 67.4%* |

*Alto NaN esperado: la mayoría nunca fue arrestada/encarcelada.

---

## 16. Limitaciones del dataset para este análisis

1. **Autoinforme:** Las conductas delictivas son autoreportadas. Existe subreporte, especialmente en crímenes graves.
2. **Attrition:** Algunos individuos dejan de ser entrevistados con el tiempo (`Non-interview`). DS0002 es una submuestra con sesgo de selección (quienes estaban disponibles o encarcelados).
3. **Assault ≠ Aggravated Assault:** La variable `ASSAULT_CY{n}` captura cualquier agresión física intencional. No distingue entre simple assault y aggravated assault (con arma, con lesiones graves).
4. **Cohorte única:** Solo cubre nacidos 1980–1984 en EE.UU. No es generalizable a otras cohortes o países.
5. **Censura temporal:** El seguimiento termina a los 25 años; la trayectoria criminal post-25 no está capturada.
