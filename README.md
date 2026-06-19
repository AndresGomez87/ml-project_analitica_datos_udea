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
│   │       ├── dataset_regresion_listings.csv          # Dataset Airbnb CDMX (Lab 4)
│   │       └── dataset_clasificacion/                  # Datasets BRFSS 2015 (Lab 5)
│   │           ├── diabetes_binary_health_indicators_BRFSS2015.csv
│   │           ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   │           └── diabetes_012_health_indicators_BRFSS2015.csv
│   │
│   ├── notebooks/
│   │   ├── lab4_regresion.ipynb                        # Regresión Airbnb (Lab 4 + v2 optimizado)
│   │   ├── eda_regresion.ipynb                         # EDA Lab 4
│   │   ├── eda_clasificacion.ipynb                     # EDA Lab 5
│   │   ├── lab5_diabetes_desbalanceo.ipynb             # Clasificación diabetes (Lab 5)
│   │   └── sqlite_regresion.ipynb                      # Exploración SQLite
│   │
│   ├── models/                                         # Modelos serializados (.joblib)
│   │   ├── model_regression_v2.joblib                  # Ensemble LightGBM+XGBoost v2
│   │   ├── features_regression_v2.joblib
│   │   ├── model_metadata_regression.joblib
│   │   ├── model_metadata_regression_v2.joblib
│   │   ├── model_classification.joblib
│   │   └── model_metadata_classification.joblib
│   │
│   ├── reports/                                        # Figuras generadas
│   ├── src/
│   │   ├── app_analytix.py                             # ← APP PRINCIPAL (Flask)
│   │   └── templates/
│   │       └── index.html                              # UI profesional (HTML + JS)
│   │
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## Aplicación web — Analytix Intelligence

Plataforma que expone los dos modelos entrenados (Lab 4 y Lab 5) como una aplicación web interactiva con diseño profesional.

### Modelos incluidos

| Producto | Modelo | Problema |
|---|---|---|
| **NightPrice AI** | Ensemble LightGBM + XGBoost | Predicción de precio por noche en Airbnb CDMX |
| **MediGuard** | XGBoost Classifier | Detección de riesgo de diabetes (BRFSS 2015) |

### Requisitos previos

Python 3.10+ y los siguientes paquetes (además del `requirements.txt` base):

```bash
pip install flask optuna
```

> `flask` y `optuna` no están en `requirements.txt`. Todo lo demás (scikit-learn, lightgbm, xgboost, joblib, pandas, numpy) ya está incluido.

### Pasos para correr la app

```bash
# 1. Instalar dependencias base
cd ml-proyecto_analitica_datos
pip install -r requirements.txt

# 2. Instalar los paquetes adicionales de la app
pip install flask optuna

# 3. Lanzar el servidor
cd src
python3 app_analytix.py
```

El servidor queda disponible en: **http://localhost:8501**

### En VS Code / GitHub Codespaces

Si trabajas en un Codespace o dev container, el puerto 8501 se redirige automáticamente. Ve a la pestaña **Ports** en el panel inferior y abre el enlace que aparece para el puerto 8501.

### Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Interfaz web principal |
| `POST` | `/api/predict/airbnb` | Predicción de precio Airbnb (JSON) |
| `POST` | `/api/predict/diabetes` | Evaluación de riesgo de diabetes (JSON) |

---

## Laboratorio 4 — Regresión (v2 optimizada)

Predicción del precio por noche de alojamientos Airbnb en Ciudad de México.

- **Dataset:** `dataset_regresion_listings.csv` — 27,051 listados, 18 columnas
- **Target:** `log(price)` en MXN/noche

### Técnicas aplicadas en la versión optimizada (v2)

| Fase | Descripción |
|---|---|
| Feature engineering | 24 features nuevas: 8 distancias geográficas (Zócalo, Polanco, Condesa, Reforma, AICM…), 11 indicadores de texto/amenidades, 4 interacciones |
| Encoding | `TargetEncoder` (sklearn 1.8) en `neighbourhood` — sin data leakage |
| Modelos | LightGBM + XGBoost, ambos con tuning bayesiano vía **Optuna** (100 trials totales) |
| Validación | KFold-5 |
| Ensemble | Media ponderada por 1/RMSE_CV |
| Guardado | `models/model_regression_v2.joblib` |

### Métricas — comparativa v1 vs v2

| Modelo | R²_log | R²_orig | MAE (MXN) | RMSE (MXN) | Gap R² |
|---|---|---|---|---|---|
| Baseline XGBoost v1 [TEST] | 0.6415 | 0.5176 | $353 | $561 | 0.339 |
| LightGBM v2 [TEST] | 0.6585 | 0.5363 | $344 | $550 | 0.236 |
| XGBoost v2 [TEST] | 0.6584 | 0.5359 | $345 | $551 | 0.224 |
| **Ensemble v2 [TEST]** | **0.6621** | **0.5394** | **$343** | **$549** | **0.229** |

Mejoras v2 vs v1: R²_log +3.2% · MAE −2.8% · overfitting gap −32.5%

### Ejecutar notebook

```bash
jupyter lab notebooks/lab4_regresion.ipynb
```

---

## Laboratorio 5 — Clasificación con Datos Desbalanceados

Detección temprana de diabetes usando el dataset BRFSS 2015 del CDC (253,680 adultos).

- **Target:** `Diabetes_binary` (0 = sin diabetes, 1 = prediabetes/diabetes)
- **Desbalanceo:** ~6.18 : 1 (sanos / diabéticos)
- **Métrica prioritaria:** Recall (minimizar falsos negativos)

### Métricas del modelo final (XGBoost con `scale_pos_weight`)

| Métrica | Valor |
|---|---|
| Recall | ~0.789 |
| F1-Score | ~0.456 |
| ROC-AUC | ~0.829 |
| Precisión | ~0.313 |
| Umbral óptimo (Youden) | ~0.26 |

### Ejecutar notebook

```bash
pip install imbalanced-learn   # requerido para Lab 5
jupyter lab notebooks/lab5_diabetes_desbalanceo.ipynb
```

---

## Instalación completa (todo el proyecto)

```bash
git clone <repo-url>
cd ml-project_analitica_datosv/ml-proyecto_analitica_datos
pip install -r requirements.txt
pip install flask optuna imbalanced-learn
```

> El proyecto usa Python 3.12. Se recomienda crear un entorno virtual antes de instalar.

---

## Autor

**Andrés Gómez** — [@AndresGomez87](https://github.com/AndresGomez87)
