# Analítica de Datos — Modelación

**Universidad de Antioquia · Facultad de Ciencias Exactas y Naturales · Instituto de Matemáticas**
Curso: Analítica de Datos | Profesor: Duván Cataño

---

## Estructura del proyecto

```
ml-project_analitica_datosv/
│
├── ml-proyecto_analitica_datos/
│   ├── data/
│   │   └── raw/
│   │       ├── dataset_regresion_listings.csv          # Dataset Airbnb (Lab 4)
│   │       └── dataset_clasificacion/                  # Datasets BRFSS 2015 (Lab 5)
│   │           ├── diabetes_binary_health_indicators_BRFSS2015.csv
│   │           ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   │           └── diabetes_012_health_indicators_BRFSS2015.csv
│   │
│   ├── notebooks/
│   │   ├── eda_regresion.ipynb                         # EDA Lab 4
│   │   ├── eda_clasificacion.ipynb                     # EDA Lab 5
│   │   ├── lab5_diabetes_desbalanceo.ipynb             # Solución Lab 5
│   │   └── sqlite_regresion.ipynb                      # Exploración SQLite
│   │
│   ├── reports/
│   │   └── EDA_clasificacion/                          # Figuras del EDA de clasificación
│   │
│   ├── models/                                         # Modelos serializados (.joblib)
│   ├── src/                                            # Scripts auxiliares y apps
│   ├── lab4_regresion.py                               # Solución Lab 4
│   └── requirements.txt
│
├── .gitignore
├── README.md
└── reporte_errores.md
```

---

## Laboratorio 4 — Regresión

### Descripción del problema

Pipeline completo de **regresión supervisada** para predecir el **precio por noche (USD)** de alojamientos en Airbnb. El dataset contiene listados con variables geográficas, de popularidad y tipo de habitación.

- **Variable objetivo:** `price` (precio por noche en USD)
- **Tipo de regresión:** Regresión continua multivariada
- **Dataset:** `dataset_regresion_listings.csv`

### Pipeline técnico

| Fase | Descripción |
|---|---|
| Preprocesamiento | Limpieza de `price` (símbolo `$`, outliers p99), imputación por mediana, escalamiento estándar |
| División | 70 % train / 30 % test (`random_state=42`) |
| Modelos | Regresión Lineal, Ridge, LASSO, Árbol de Decisión, Random Forest, XGBoost, LightGBM |
| Validación | K-Fold CV (k=5), métricas: MAE ± std, RMSE ± std, R² |
| Tuning | GridSearchCV (Ridge), RandomizedSearchCV (LightGBM) |
| Interpretabilidad | Coeficientes Ridge, feature importances LightGBM |
| Guardado | `models/model_regression.joblib`, `models/features_regression.joblib` |

### Ejecución

```bash
cd ml-proyecto_analitica_datos
python lab4_regresion.py
```

---

## Laboratorio 5 — Clasificación con Datos Desbalanceados

### Descripción del problema

Pipeline de **clasificación binaria** para detección temprana de **diabetes** en el dataset BRFSS 2015 del CDC (253.680 adultos). El desbalanceo de clases (86.1 % sin diabetes / 13.9 % con diabetes) es el eje central del laboratorio.

- **Variable objetivo:** `Diabetes_binary` (0 = sin diabetes, 1 = prediabetes/diabetes)
- **Clases:** Binaria (0/1)
- **Dataset:** `diabetes_binary_health_indicators_BRFSS2015.csv`
- **Métrica prioritaria:** Recall > F1 > ROC-AUC (minimizar falsos negativos es crítico)

### Técnicas de balanceo implementadas

| Categoría | Técnica |
|---|---|
| Sobremuestreo | Random Oversampling |
| Submuestreo | Random Undersampling |
| Ensambles | Balanced Random Forest, Easy Ensemble |
| Referencia | Baseline (sin balanceo) — Regresión Logística |

### Técnicas descritas / por implementar

SMOTE, ADASYN, TomekLinks, ENN, SMOTETomek, SMOTEENN, RUSBoost, BalancedBagging

### Pipeline técnico

| Fase | Descripción |
|---|---|
| Preprocesamiento | Escalamiento en variables continuas (BMI, MentHlth, PhysHlth), passthrough para binarias/ordinales |
| División | 80/20 stratified (ver `reporte_errores.md` — spec exige 70/30) |
| Validación | Stratified K-Fold CV (k=5), métricas: Accuracy, Precision, Recall, F1, AUC |
| Umbral | Ajuste dinámico del umbral de decisión via curva Precision-Recall |
| Guardado | `models/model_classification.joblib`, `models/features_classification.joblib` |

### Ejecución

```bash
pip install imbalanced-learn   # No incluido en requirements.txt
jupyter lab notebooks/lab5_diabetes_desbalanceo.ipynb
```

---

## Instalación de dependencias

```bash
cd ml-proyecto_analitica_datos
pip install -r requirements.txt
pip install imbalanced-learn   # Requerido para Lab 5 (faltante en requirements.txt)
```

> El proyecto usa Python 3.12 con los paquetes listados en `requirements.txt`. Se recomienda usar el entorno virtual `venv/` incluido (no versionado en git).

---

## Métricas resumen (resultados parciales)

### Lab 4 — Validación Cruzada (k=5)

> Resultados al ejecutar `lab4_regresion.py`. Mejor modelo por RMSE determinado dinámicamente.

### Lab 5 — Validación Cruzada (k=5, Stratified)

| Modelo | Accuracy | F1 | Recall | ROC-AUC |
|---|---|---|---|---|
| Baseline (sin balanceo) | 0.86 | 0.23 | 0.15 | — |
| Random Oversampling + RF | 0.84 | 0.35 | 0.30 | — |
| Random Undersampling + RF | 0.71 | 0.42 | 0.77 | — |
| Balanced Random Forest | 0.76 | 0.44 | 0.68 | — |
| Easy Ensemble | 0.73 | 0.44 | 0.77 | — |

> CV completa pendiente de ejecutar (ver `reporte_errores.md` para detalles).

---

## Autor

**Andrés Gómez** — [@AndresGomez87](https://github.com/AndresGomez87)
