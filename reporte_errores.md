# Reporte de Errores y Observaciones

**Proyecto:** Analítica de Datos — Labs 4 y 5
**Fecha de auditoría:** 2026-06-17
**Estado:** Solo lectura — ningún archivo de código fue modificado

---

## LAB 4 — `ml-proyecto_analitica_datos/lab4_regresion.py`

### 🔴 ERROR CRÍTICO 1 — Rutas relativas inconsistentes

**Ubicación:** Líneas 83-87, 788-794, 593, 619, 681, 775

**Problema:**
El script mezcla dos convenciones de ruta relativa que solo funcionan si se ejecuta desde directorios distintos:

- `RUTA_DATASET = os.path.join("ml-proyecto_analitica_datos", "data", ...)` (L84)
  Funciona si el CWD es la **raíz del repo** (`/workspaces/ml-project_analitica_datosv/`).

- `os.makedirs("../models", ...)` y `joblib.dump(..., "../models/model_regression.joblib")` (L788-794)
  Asume que el CWD es `ml-proyecto_analitica_datos/`, guardando el modelo un nivel arriba.
  Si el script se corre desde la raíz, `../models` apunta fuera del workspace.

- Rutas de figuras: `plt.savefig("fig1_comparacion_modelos.png", ...)` (L593, L619, L681, L775)
  Sin directorio destino, se guardan en el CWD. Evidencia: los archivos `fig1-fig4.png` aparecen en la raíz del repo en lugar de `reports/`.

**Consecuencia observada:** Las cuatro figuras generadas (`fig1_comparacion_modelos.png`, `fig2_metricas_cv.png`, `fig3_analisis_residuos.png`, `fig4_importancia_variables.png`) están en la raíz del repositorio en vez de en `reports/` o dentro de `ml-proyecto_analitica_datos/`.

**Corrección sugerida:** Usar `os.path.dirname(os.path.abspath(__file__))` como base de todas las rutas.

---

### 🔴 ERROR CRÍTICO 2 — Etiquetado incorrecto del modelo no-lineal

**Ubicación:** Líneas 446, 504-507, 763

**Problema:**
El modelo no lineal que se ajusta y evalúa es **LightGBM** (`gs_lgbm`), pero está referenciado y etiquetado como "Random Forest" en tres lugares:

| Línea | Código problemático |
|---|---|
| 446 | Comentario: `"Ajustando Random Forest (modelo de árboles)..."` |
| 506 | `"Random Forest (ajustado)": (gs_lgbm.best_estimator_, mejor_rmse_lgbm)` |
| 763 | Título de gráfica: `"Importancia de Variables – Random Forest (Top 15)"` |

**Consecuencia:** La selección del modelo final puede mostrar "Random Forest (ajustado)" cuando en realidad es LightGBM. Los resultados son correctos, pero la identidad del modelo es falsa en la salida del programa y en la figura.

---

### 🟡 ERROR MODERADO 3 — Posible mismatch en feature importances (LightGBM vs OHE)

**Ubicación:** Líneas 746-761

**Problema:**
La variable `nombres_features` se construye como `VARIABLES_NUMERICAS + nombres_cat`, donde `nombres_cat` proviene del `OneHotEncoder` del pipeline de **Ridge** (L708-712). LightGBM dentro de su propio pipeline recibe las features tal como las entrega su preprocesador (que también tiene un OHE). Si el número de categorías únicas en `room_type` coincide, no habrá error; pero si difiere entre el pipeline de Ridge y el de LightGBM (posible al usar `handle_unknown="ignore"` con splits de CV distintos), `importancias_rf[:n_imp]` recortará silenciosamente importancias o nombres.

El código usa `n_imp = min(len(importancias_rf), len(nombres_features))` como guarda, pero esto solo silencia el error sin corregirlo.

---

### 🟡 ERROR MODERADO 4 — `cross_val_score` redundante post GridSearchCV

**Ubicación:** Líneas 483-490

**Problema:**
Después de `gs_lgbm.fit(X_train, y_train)`, se llama `cross_val_score(gs_lgbm.best_estimator_, X_train, y_train, ...)`. Esto re-entrena el pipeline completo (preprocesador + modelo) 5 veces sobre `X_train`, cuando `GridSearchCV` ya calculó el R² internamente durante la búsqueda. Aumenta el tiempo de cómputo sin aportar información nueva.

**Corrección sugerida:** Extraer R² directamente con `gs_lgbm.cv_results_` o usar `scoring` múltiple desde el inicio.

---

### 🟡 OBSERVACIÓN 5 — Variables predictoras importantes omitidas

**Ubicación:** Líneas 146-157

**Problema de diseño:**
El dataset de Airbnb típicamente contiene `neighbourhood_group`, `neighbourhood` y `minimum_nights`, que son predictores de alto impacto en el precio. El script solo usa 5 variables numéricas + `room_type`. La omisión puede explicar un R² bajo y un RMSE elevado.

No es un bug de código, pero debería justificarse en el informe del laboratorio.

---

### 🟢 NOTA 6 — IsotonicRegression documentalmente excluida

**Ubicación:** Líneas 55, 239-241, 313

`IsotonicRegression` se importa y se documenta su exclusión (requiere entrada 1D univariada). Correcto. El informe debe incluir esta justificación explícitamente para cumplir el punto 3d del enunciado.

---

## LAB 5 — `ml-proyecto_analitica_datos/notebooks/lab5_diabetes_desbalanceo.ipynb`

### 🔴 ERROR CRÍTICO 1 — Kernel crash después de Easy Ensemble

**Ubicación:** Celda 34 (output visible en el notebook)

**Problema:**
El notebook muestra el mensaje de crash del kernel tras ejecutar `EasyEnsembleClassifier`. Causa probable: agotamiento de memoria (OOM) con 253K muestras y múltiples estimadores AdaBoost internos. El `EasyEnsembleClassifier` con parámetros por defecto (`n_estimators=10`) crea 10 clasificadores AdaBoost sobre subconjuntos balanceados del dataset completo.

**Consecuencia:** Toda ejecución secuencial del notebook falla a partir de la celda 34. Las celdas 35-50 no se ejecutaron.

---

### 🔴 ERROR CRÍTICO 2 — Validación cruzada incompleta / colgada

**Ubicación:** Celda 36

**Problema:**
`cross_validate(model, X, y, cv=cv, ...)` usa el dataset completo (`X` con 253K filas) con Stratified K-Fold k=5. Para `BalancedRandomForestClassifier(n_estimators=300)` y `EasyEnsembleClassifier`, esto implica entrenar ~10+ modelos pesados por fold, sobre datasets de 200K muestras. La salida del notebook muestra solo:

```
Evaluando: Baseline -- Sin balanceo ... OK
Evaluando: Random Oversampling ...
```

La ejecución se detuvo en el segundo modelo. `results_df` está incompleto o vacío, rompiendo las celdas 39-50 que dependen de `results_df.iloc[0]`.

**Consecuencia directa:** `best_model_name`, `best_model_pipe`, `y_prob_best`, importancias de variables, modelo final serializado — todo sin ejecutar.

---

### 🔴 ERROR CRÍTICO 3 — `imbalanced-learn` ausente en `requirements.txt`

**Ubicación:** Celda 2 (importaciones), archivo `requirements.txt`

**Problema:**
El notebook importa:
```python
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, ...
from imblearn.pipeline import Pipeline as ImbPipeline
```

`imbalanced-learn` **no aparece en `requirements.txt`**. En un ambiente limpio (sin el `venv/` existente), la celda 2 fallará con `ModuleNotFoundError`.

**Corrección:** Agregar `imbalanced-learn>=0.12.0` a `requirements.txt`.

---

### 🔴 ERROR CRÍTICO 4 — Secciones implementadas incompletas vs. descripción del notebook

**Ubicación:** Celda 0 (markdown introductorio), Celdas 29-30

**Problema:**
La celda de introducción lista las siguientes técnicas como implementadas o por implementar:

| Técnica | Estado real |
|---|---|
| Random Oversampling | Implementada (celda 25) |
| SMOTE | Importada, **sin celda de entrenamiento** |
| ADASYN | Importada, **sin celda de entrenamiento** |
| Random Undersampling | Implementada (celda 28) |
| TomekLinks | Importada, **sin celda de entrenamiento** |
| ENN | Importada, **sin celda de entrenamiento** |
| SMOTETomek | Importada, **sin celda de entrenamiento** |
| SMOTEENN | Importada, **sin celda de entrenamiento** |
| Balanced Random Forest | Implementada (celda 32) |
| Easy Ensemble | Implementada parcialmente (celda 34, crash) |
| RUSBoostClassifier | Importada, **sin celda de entrenamiento** |
| BalancedBaggingClassifier | Importada, **sin celda de entrenamiento** |

Las secciones 10 (Combinaciones) y parte de la 11 (Ensambles) están vacías salvo por los títulos markdown.

---

### 🔴 ERROR CRÍTICO 5 — Modelos requeridos por el enunciado ausentes

**Ubicación:** `Lab_5.pdf` — sección "Modelos de Clasificación", notebook celdas 22-34

**Problema:**
El enunciado del Lab 5 exige entrenar:
1. Regresión Logística ✅ (celda 22, baseline)
2. Árbol de Clasificación ❌ **Ausente**
3. Random Forest Classifier ✅ (dentro de pipelines de balanceo)
4. XGBoost Classifier ❌ **Ausente como modelo standalone**

Además, el ajuste de hiperparámetros con `RandomizedSearchCV` **no está implementado** (requerido por el enunciado).

---

### 🔴 ERROR CRÍTICO 6 — Curva ROC no implementada

**Ubicación:** `Lab_5.pdf` — sección "Curva ROC"

El enunciado exige construir la curva ROC del mejor modelo e interpretar sensibilidad, especificidad y capacidad discriminativa. No existe ninguna celda con esta implementación en el notebook.

---

### 🟡 ERROR MODERADO 7 — Train-test split 80/20 vs. enunciado 70/30

**Ubicación:** Celda 18

**Problema:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
```

El enunciado del Lab 5 especifica división **70 % train / 30 % test**. El notebook usa 80/20. Desalineado con el Lab 4 (que sí usa 70/30) y con la especificación del lab.

---

### 🟡 ERROR MODERADO 8 — Modelo final entrenado con datos de test incluidos

**Ubicación:** Celda 49

**Problema:**
```python
final_model.fit(X, y)  # X incluye X_test
```

Las métricas de celdas 22-34 se calcularon sobre `X_test`. Luego, la celda 49 incluye ese mismo `X_test` en el entrenamiento final. Si bien es práctica aceptable para producción, metodológicamente implica que el modelo final no puede ser evaluado de forma imparcial sobre los datos ya expuestos.

---

### 🟢 NOTA 9 — Análisis de FP/FN sin celda dedicada

**Ubicación:** Celda 51 (tabla resumen)

La celda 51 lista en una tabla los riesgos de FP/FN, pero no hay análisis cuantitativo ni interpretación narrativa con los números reales del modelo. El enunciado exige "analizar: Falsos positivos, Falsos negativos, Riesgos asociados a errores de clasificación".

---

## Resumen ejecutivo de errores

| # | Lab | Severidad | Descripción |
|---|---|---|---|
| L4-1 | Lab 4 | 🔴 Crítico | Rutas relativas inconsistentes (dataset, modelos, figuras) |
| L4-2 | Lab 4 | 🔴 Crítico | LightGBM etiquetado como "Random Forest" en 3 lugares |
| L4-3 | Lab 4 | 🟡 Moderado | Posible mismatch feature importances LightGBM vs OHE |
| L4-4 | Lab 4 | 🟡 Moderado | `cross_val_score` redundante post GridSearchCV |
| L4-5 | Lab 4 | 🟡 Observación | Variables predictoras importantes omitidas |
| L4-6 | Lab 4 | 🟢 Nota | Isotónica excluida correctamente, justificar en informe |
| L5-1 | Lab 5 | 🔴 Crítico | Kernel crash — OOM en EasyEnsembleClassifier |
| L5-2 | Lab 5 | 🔴 Crítico | CV incompleta — `results_df` roto, celdas 39-50 sin ejecutar |
| L5-3 | Lab 5 | 🔴 Crítico | `imbalanced-learn` ausente en `requirements.txt` |
| L5-4 | Lab 5 | 🔴 Crítico | 8 de 12 técnicas de balanceo importadas pero sin implementar |
| L5-5 | Lab 5 | 🔴 Crítico | Árbol de Clasificación y XGBoost standalone ausentes |
| L5-6 | Lab 5 | 🔴 Crítico | Curva ROC no implementada |
| L5-7 | Lab 5 | 🟡 Moderado | Split 80/20 vs. spec 70/30 |
| L5-8 | Lab 5 | 🟡 Moderado | Modelo final entrenado con datos de test |
| L5-9 | Lab 5 | 🟢 Nota | Análisis FP/FN solo tabular, sin interpretación cuantitativa |

---

*Auditoría de solo lectura — ningún archivo de código fue modificado durante este proceso.*
