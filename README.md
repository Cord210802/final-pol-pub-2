# final-pol-pub-2

Proyecto final de Tópicos de Políticas Públicas II (ITAM). Análisis de importancia
de variables sobre el inicio y la persistencia de la violencia juvenil con la
Encuesta Nacional Longitudinal de la Juventud de 1997 (estudio ICPSR 34562).

## Estructura

- `paper/` — documento final en Quarto (`paper.qmd`) y su PDF, con bibliografía.
- `analisis/` — scripts de análisis (modelos de bosque aleatorio, PCA, construcción del dataset) y figuras.
- `contexto/` — descripción del dataset e indicaciones del trabajo.
- `eda_recidivism_nlsy97.ipynb` — notebook de exploración (EDA) y feature importance.

## Datos

Los microdatos de ICPSR 34562 no se incluyen en el repositorio por sus términos de
uso, que prohíben la redistribución sin autorización de ICPSR. Para reproducir el
análisis hay que descargarlos desde ICPSR y colocarlos en `data/`.
