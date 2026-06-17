# Contexto del Proyecto — Analítica de Datos (UdeA)

> **Propósito de este archivo:** Memoria técnica para restaurar el contexto completo del repositorio en nuevas sesiones de Claude. Actualizar cuando cambie el estado de un laboratorio.

---

## Repositorio

- **Ruta raíz:** `/workspaces/ml-project_analitica_datosv/`
- **Rama principal:** `main`
- **Git user:** `AndresGomez87`
- **Curso:** Analítica de Datos · Instituto de Matemáticas · Universidad de Antioquia
- **Profesor:** Duván Cataño

---

## Estructura de Carpetas

```
ml-project_analitica_datosv/          ← raíz del repo git
├── .gitignore                         ← cubre: venv/, __pycache__/, *.joblib, *.db, figs generadas
├── README.md                          ← descripción técnica de Labs 4 y 5
├── reporte_errores.md                 ← auditoría de errores (solo lectura, no modifica código)
├── contexto_proyecto.md               ← este archivo
│
└── ml-proyecto_analitica_datos/       ← directorio del proyecto Python
    ├── requirements.txt               ← dependencias (incluye imbalanced-learn>=0.12.0)
    ├── data/
    │   └── raw/
    │       ├── dataset_regresion_listings.csv          ← Airbnb NYC (Lab 4)
    │       └── dataset_clasificacion/
    │           ├── diabetes_binary_health_indicators_BRFSS2015.csv   ← Lab 5 (usado)
    │           ├── diabetes_binary_5050split_...csv                  ← balanceado 50/50
    │           └── diabetes_012_health_indicators_...csv             ← 3 clases
    ├── notebooks/
    │   ├── eda_regresion.ipynb         ← EDA Lab 4 (PROTEGIDO — no eliminar)
    │   ├── eda_clasificacion.ipynb     ← EDA Lab 5 (PROTEGIDO — no eliminar)
    │   ├── lab4_regresion.ipynb        ← Solución Lab 4 (migrado desde .py, corregido)
    │   ├── lab5_diabetes_desbalanceo.ipynb  ← Solución Lab 5 (corregido)
    │   └── sqlite_regresion.ipynb      ← Exploración SQLite auxiliar
    ├── models/                         ← Artefactos serializados (.joblib) — gitignoreados
    ├── reports/
    │   ├── lab4/                       ← Figuras generadas por Lab 4 (fig1-4_*.png)
    │   └── EDA_clasificacion/          ← Figuras EDA Lab 5 (PROTEGIDAS)
    ├── database/                       ← SQLite DB (gitignoreado)
    └── src/                            ← Apps Streamlit y scripts auxiliares
```

---

## Stack Tecnológico

| Componente | Versión / Detalle |
|---|---|
| Python | 3.12 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| imbalanced-learn | ≥0.12.0 (añadido manualmente a requirements.txt) |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| plotly | 6.6.0 |
| joblib | 1.5.3 |
| jupyter / jupyterlab | 1.1.1 / 4.5.6 |
| scipy | 1.17.1 |

**Entorno virtual:** `ml-proyecto_analitica_datos/venv/` (no versionado, Python 3.12)

---

## Estado de los Laboratorios

### Laboratorio 4 — Regresión (precio Airbnb)

**Archivo:** `notebooks/lab4_regresion.ipynb` (31 celdas)
**Estado:** ✅ Migrado desde `.py` y completamente corregido

**Dataset:** `data/raw/dataset_regresion_listings.csv`
- Variable objetivo: `price` (USD/noche)
- División: 70% train / 30% test
- Variables: latitud, longitud, reseñas, disponibilidad, room_type (+ extras si existen)

**Pipeline:**
1. Limpieza de `price` (símbolo `$`, outliers p99)
2. `ColumnTransformer` → `SimpleImputer` + `StandardScaler` (numéricas) / `OneHotEncoder` (categóricas)
3. 7 modelos: Lineal, Ridge, LASSO, Árbol, RandomForest, XGBoost, LightGBM
4. K-Fold CV k=5 → MAE±σ, RMSE±σ, R²
5. GridSearchCV (Ridge) + RandomizedSearchCV (LightGBM, con R² en `cv_results_`)
6. Figuras → `reports/lab4/fig1-4_*.png`
7. Modelos → `models/model_regression.joblib` + `features_regression.joblib`

**Correcciones aplicadas (ver `reporte_errores.md` sección Lab 4):**
- L4-1: Rutas absolutas con `pathlib` desde la ubicación del notebook
- L4-2: LightGBM correctamente etiquetado (antes decía "Random Forest" en 3 lugares)
- L4-3: Feature importances extraídas del preprocesador de LightGBM (no del de Ridge)
- L4-4: R² extraído de `cv_results_['mean_test_R2']`, sin `cross_val_score` redundante
- L4-5: Detección dinámica de columnas adicionales (`neighbourhood_group`, etc.)

---

### Laboratorio 5 — Clasificación con Datos Desbalanceados (Diabetes)

**Archivo:** `notebooks/lab5_diabetes_desbalanceo.ipynb` (78 celdas)
**Estado:** ✅ Corregido — técnicas completas, CV funcional, curva ROC implementada

**Dataset:** `data/raw/dataset_clasificacion/diabetes_binary_health_indicators_BRFSS2015.csv`
- 253.680 registros, 22 columnas (todas float64, sin NaN)
- Variable objetivo: `Diabetes_binary` (0=sin diabetes, 1=prediabetes/diabetes)
- Desbalanceo: 86.1% clase 0 / 13.9% clase 1 → ratio 6.2:1
- Variables clave: BMI, GenHlth, Age, HighBP, HighChol, PhysActivity
- Métrica prioritaria: **Recall > F1 > ROC-AUC > Accuracy**

**División:** 70% train (202.944) / 30% test (50.736), `stratify=y`

**Constante de semilla:** `SEED = 123`

**Técnicas de balanceo implementadas (12):**

| Categoría | Técnica | Variable |
|---|---|---|
| Sobremuestreo | Random Oversampling | `ros_model` |
| Sobremuestreo | SMOTE | `smote_model` |
| Sobremuestreo | ADASYN | `adasyn_model` |
| Submuestreo | Random Undersampling | `rus_model` |
| Submuestreo | TomekLinks | `tomek_model` |
| Submuestreo | EditedNearestNeighbours | `enn_model` |
| Combinación | SMOTETomek | `smote_tomek_model` |
| Combinación | SMOTEENN | `smote_enn_model` |
| Ensamble | Balanced Random Forest (300 est.) | `brf_model` |
| Ensamble | Easy Ensemble (**n_estimators=5**) | `easy_model` |
| Ensamble | RUSBoost (50 est.) | `rusboost_model` |
| Ensamble | Balanced Bagging (50 est.) | `bbagging_model` |

**Modelos standalone (requeridos por enunciado):**
- Regresión Logística (`baseline_model`) — `preprocessor_scaled`
- Árbol de Clasificación (`dt_model`, max_depth=10) — `preprocessor_no_scaled`
- XGBoost (`xgb_model`, scale_pos_weight=6) — `preprocessor_no_scaled`
- Random Forest (dentro de pipelines de balanceo) — `preprocessor_no_scaled`

**CV:** `StratifiedKFold(k=5)` sobre `X_train` (no X completo — corrección crítica)
- Modelos en CV: Baseline, ROS, SMOTE, ADASYN, RUS, SMOTETomek, BRF(50 est.), EasyEnsemble, RUSBoost, BalancedBagging, DT, XGBoost

**Ajuste de hiperparámetros:** `RandomizedSearchCV(n_iter=10, cv=StratifiedKFold(3), scoring='f1')`
- LR: espacio `C`, `max_iter`, `solver`
- XGBoost: espacio `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `scale_pos_weight`

**Curva ROC:** implementada con `roc_curve` + índice de Youden para umbral óptimo

**Guardado:** `models/model_classification.joblib` + `features_classification.joblib`
- Entrenado solo sobre `X_train` (corrección L5-8)

**Correcciones aplicadas (ver `reporte_errores.md` sección Lab 5):**
- L5-1: EasyEnsemble `n_estimators=5` (evita OOM con 200K muestras)
- L5-2: CV usa `X_train` (no el dataset completo X)
- L5-3: `imbalanced-learn>=0.12.0` añadido a `requirements.txt`
- L5-4: Las 8 técnicas faltantes implementadas (SMOTE, ADASYN, TomekLinks, ENN, SMOTETomek, SMOTEENN, RUSBoost, BalancedBagging)
- L5-5: `DecisionTreeClassifier` + `XGBClassifier` como modelos standalone
- L5-6: Curva ROC con interpretación de sensibilidad/especificidad/AUC
- L5-7: `test_size=0.30` (spec 70/30; antes era 0.20)
- L5-8: `final_model.fit(X_train, y_train)` (antes usaba el dataset completo X)

---

## Convenciones y Reglas de Estilo

### Reglas estrictas
1. **EDA protegido:** nunca eliminar ni modificar las celdas de EDA ni los notebooks `eda_*.ipynb`
2. **Sin modificar `.gitignore` interno** de `ml-proyecto_analitica_datos/` — ya cubre `venv/`, `__pycache__/`, `*.db`
3. **Rutas en notebooks:** siempre usar `pathlib.Path` con base en la ubicación del notebook (`Path(os.path.abspath('')).parent` o equivalente), nunca rutas relativas al CWD
4. **Modelos:** guardar en `models/` dentro del proyecto con nombres `model_<tipo>.joblib` y `features_<tipo>.joblib`
5. **Figuras:** guardar en `reports/lab<N>/` con nombres descriptivos
6. **Semilla:** `random_state=42` en Lab 4, `SEED=123` en Lab 5

### Notebooks
- Estructura: celdas markdown de sección → celdas de código con comentarios de bloque (`# ══ ... ══`)
- `display()` para DataFrames en notebooks (no `print`)
- Plotly para gráficas interactivas en Lab 5; matplotlib/seaborn en Lab 4
- `ImbPipeline` de `imblearn.pipeline` en Lab 5 (no `sklearn.pipeline.Pipeline`) para que los samplers operen solo en train durante CV
- El `preprocessor_scaled` se usa para modelos lineales (LR); `preprocessor_no_scaled` para árboles y ensambles

### Preprocesadores Lab 5
```python
# numeric_features = ["BMI", "MentHlth", "PhysHlth"]  — escalan
# ordinal_features = 18 variables binarias/ordinales    — passthrough
preprocessor_scaled    → numeric: Imputer + StandardScaler | ordinal: Imputer
preprocessor_no_scaled → all_features: Imputer (passthrough)
```

---

## Glosario — Datasets

### Dataset Airbnb NYC (Lab 4)
- Fuente: Inside Airbnb / Kaggle listings NYC
- Columnas clave: `price`, `room_type`, `latitude`, `longitude`, `number_of_reviews`, `reviews_per_month`, `availability_365`
- Limpieza: `price` puede venir como string `"$120.00"` → regex strip + float

### Dataset BRFSS 2015 Diabetes (Lab 5)
- Fuente: CDC BRFSS 2015 / Kaggle (Alex Teboul)
- 253.680 adultos estadounidenses, 22 variables de salud/estilo de vida
- Variables más correlacionadas con diabetes: `GenHlth`, `BMI`, `HighBP`, `HighChol`, `Age`, `DiffWalk`
- 3 versiones en el repo: binaria original (86/14%), 50/50 split, y 3 clases (0/1/2)
- El lab usa la binaria original (desbalanceada) para ilustrar el problema de desbalanceo
