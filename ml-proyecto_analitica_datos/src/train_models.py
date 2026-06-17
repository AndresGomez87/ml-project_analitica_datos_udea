"""
Entrenamiento y guardado de modelos ML — Proyecto Final
Analítica de Datos · Universidad de Antioquia

Lab 4: Regresión  — Predicción de precio Airbnb CDMX (XGBoost)
Lab 5: Clasificación — Detección de Diabetes BRFSS 2015 (XGBoost + scale_pos_weight)
"""

import warnings
warnings.filterwarnings('ignore')

import re
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    recall_score, f1_score, precision_score, roc_auc_score, roc_curve,
)
from xgboost import XGBRegressor, XGBClassifier

SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR  = PROJECT_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  LAB 4 — REGRESIÓN (Airbnb CDMX)
# ═══════════════════════════════════════════════════════════════════

FEATURES_NUM = [
    'latitude', 'longitude',
    'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm',
    'has_reviews', 'days_since_review', 'is_recently_active',
    'log_minimum_nights', 'log_number_of_reviews', 'log_reviews_per_month',
    'log_host_listings', 'log_reviews_ltm',
    'availability_ratio', 'reviews_per_listing',
    'high_min_nights', 'is_monthly_rental', 'is_multi_listing_host', 'is_entire_flag',
    'is_studio', 'is_luxury', 'is_loft', 'has_parking', 'has_pool', 'has_terrace',
    'n_bedrooms', 'neighbourhood_freq',
]
FEATURES_CAT = ['room_type', 'neighbourhood']


def engineer_features(df_input, neighbourhood_freq_map=None):
    """Replica exacta del feature engineering del Lab 4."""
    df = df_input.copy()

    valid_dates = pd.to_datetime(df['last_review'], errors='coerce').dropna()
    ref_date = valid_dates.max() if len(valid_dates) > 0 else pd.Timestamp('2025-09-30')

    last_review_dt          = pd.to_datetime(df['last_review'], errors='coerce')
    df['has_reviews']       = last_review_dt.notna().astype(int)
    df['days_since_review'] = (ref_date - last_review_dt).dt.days.fillna(9999).astype(int)
    df['is_recently_active']= (df['days_since_review'] < 180).astype(int)

    df['log_minimum_nights']    = np.log1p(df['minimum_nights'])
    df['log_number_of_reviews'] = np.log1p(df['number_of_reviews'])
    df['log_reviews_per_month'] = np.log1p(df['reviews_per_month'].fillna(0))
    df['log_host_listings']     = np.log1p(df['calculated_host_listings_count'])
    df['log_reviews_ltm']       = np.log1p(df['number_of_reviews_ltm'])

    df['availability_ratio']    = df['availability_365'] / 365.0
    df['reviews_per_listing']   = df['number_of_reviews'] / (df['calculated_host_listings_count'] + 1)
    df['high_min_nights']       = (df['minimum_nights'] > 7).astype(int)
    df['is_monthly_rental']     = (df['minimum_nights'] >= 28).astype(int)
    df['is_multi_listing_host'] = (df['calculated_host_listings_count'] > 1).astype(int)

    names = df['name'].fillna('').str.lower()
    df['is_studio']      = names.str.contains(r'studio|estudio').astype(int)
    df['is_luxury']      = names.str.contains(r'luxury|lujo|deluxe|premium|vip|suite|penthouse').astype(int)
    df['is_loft']        = names.str.contains(r'loft').astype(int)
    df['has_parking']    = names.str.contains(r'parking|estacionamiento|garage').astype(int)
    df['has_pool']       = names.str.contains(r'pool|alberca|piscina').astype(int)
    df['has_terrace']    = names.str.contains(r'terrace|terraza|rooftop|balcon').astype(int)
    df['is_entire_flag'] = (df['room_type'] == 'Entire home/apt').astype(int)

    def _extract_beds(name):
        m = re.search(r'(\d+)\s*(?:bedroom|habitaci[oó]n|rec[aá]mara|bed\s*room|br\b)', str(name).lower())
        return min(int(m.group(1)), 6) if m else 0
    df['n_bedrooms'] = df['name'].fillna('').apply(_extract_beds)

    if neighbourhood_freq_map is not None:
        mean_freq = np.mean(list(neighbourhood_freq_map.values()))
        df['neighbourhood_freq'] = df['neighbourhood'].map(neighbourhood_freq_map).fillna(mean_freq)
    else:
        neigh_freq = df['neighbourhood'].value_counts(normalize=True)
        df['neighbourhood_freq'] = df['neighbourhood'].map(neigh_freq)

    return df, ref_date


def train_regression():
    print('\n' + '='*60)
    print('  LAB 4 — REGRESIÓN — Airbnb CDMX')
    print('='*60)

    DATA_PATH = PROJECT_DIR / 'data' / 'raw' / 'dataset_regresion_listings.csv'
    if not DATA_PATH.exists():
        print(f'  ERROR: {DATA_PATH} no encontrado.')
        sys.exit(1)

    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f'  Datos: {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas')

    df = df_raw.dropna(subset=['price'])
    df = df[df['price'] > 0].copy()

    # Feature engineering con frecuencias del dataset completo
    df_eng, ref_date = engineer_features(df)
    neighbourhood_freq_map = df_eng['neighbourhood'].value_counts(normalize=True).to_dict()
    print(f'  Fecha referencia: {ref_date.date()} | Alcaldías: {len(neighbourhood_freq_map)}')

    # Outlier removal (igual al notebook)
    Q1, Q3 = df_eng['price'].quantile(0.25), df_eng['price'].quantile(0.75)
    IQR    = Q3 - Q1
    p99    = df_eng['price'].quantile(0.99)
    upper  = min(Q3 + 3.0 * IQR, p99)
    df_clean = df_eng[df_eng['price'] <= upper].copy()
    print(f'  Outliers eliminados: {len(df_eng)-len(df_clean):,} (límite ${upper:,.0f})')

    # Estadísticas de vecindarios para la app
    neighbourhood_latlon = (
        df_clean.groupby('neighbourhood')[['latitude', 'longitude']].mean().to_dict()
    )
    neighbourhood_price_stats = (
        df_clean.groupby('neighbourhood')['price']
        .agg(media='mean', mediana='median', listings='count')
        .round(0).to_dict()
    )

    X      = df_clean[FEATURES_NUM + FEATURES_CAT]
    y_log  = np.log(df_clean['price'])
    y_orig = df_clean['price']

    X_train, X_test, y_train_log, y_test_log, _, y_test_orig = train_test_split(
        X, y_log, y_orig,
        test_size=0.30, random_state=SEED,
        stratify=df_clean['room_type']
    )
    print(f'  Split: train={len(X_train):,} | test={len(X_test):,}')

    prep_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  RobustScaler()),
    ])
    prep_cat = Pipeline([
        ('imputer',  SimpleImputer(strategy='most_frequent')),
        ('encoder',  OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')),
    ])
    preprocessor = ColumnTransformer([
        ('num', prep_num, FEATURES_NUM),
        ('cat', prep_cat, FEATURES_CAT),
    ], remainder='drop')

    model_pipe = Pipeline([
        ('prep',  preprocessor),
        ('model', XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=0,
        )),
    ])

    print('  Entrenando XGBoost Regresión...')
    model_pipe.fit(X_train, y_train_log)

    y_pred_log  = model_pipe.predict(X_test)
    y_pred_orig = np.exp(y_pred_log)
    r2_log  = r2_score(y_test_log,  y_pred_log)
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    mae     = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse    = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    gap     = r2_score(y_train_log, model_pipe.predict(X_train)) - r2_log

    print(f'  R² log={r2_log:.4f} | R² orig={r2_orig:.4f} | MAE=${mae:,.0f} | Gap={gap:.4f}')

    joblib.dump(model_pipe, MODELS_DIR / 'model_regression.joblib')
    joblib.dump(
        {'numericas': FEATURES_NUM, 'categoricas': FEATURES_CAT},
        MODELS_DIR / 'features_regression.joblib'
    )

    meta = {
        'nombre_modelo'        : 'XGBoost Regresión (Airbnb CDMX)',
        'r2_test_log'          : round(r2_log, 4),
        'r2_test_original'     : round(r2_orig, 4),
        'mae_test_mxn'         : round(float(mae), 2),
        'rmse_test_mxn'        : round(float(rmse), 2),
        'gap_r2'               : round(float(gap), 4),
        'target_transform'     : 'log → exp(predicción)',
        'fecha_entrenamiento'  : str(datetime.now().date()),
        'ref_date'             : str(ref_date.date()),
        'neighbourhood_freq'   : neighbourhood_freq_map,
        'neighbourhood_latlon' : neighbourhood_latlon,
        'neighbourhood_stats'  : neighbourhood_price_stats,
        'room_types'           : sorted(df_clean['room_type'].unique().tolist()),
        'neighbourhoods'       : sorted(df_clean['neighbourhood'].unique().tolist()),
        'price_upper_bound'    : round(float(upper), 2),
        'price_median'         : round(float(df_clean['price'].median()), 2),
    }
    joblib.dump(meta, MODELS_DIR / 'model_metadata_regression.joblib')
    print(f'  Guardado: model_regression.joblib ✔')
    return meta


# ═══════════════════════════════════════════════════════════════════
#  LAB 5 — CLASIFICACIÓN (Diabetes BRFSS 2015)
# ═══════════════════════════════════════════════════════════════════

CLF_NUMERIC_FEATURES  = ['BMI', 'MentHlth', 'PhysHlth']
CLF_ORDINAL_FEATURES  = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk',
    'Sex', 'GenHlth', 'Age', 'Education', 'Income',
]
CLF_ALL_FEATURES = CLF_NUMERIC_FEATURES + CLF_ORDINAL_FEATURES


def train_classification():
    print('\n' + '='*60)
    print('  LAB 5 — CLASIFICACIÓN — Diabetes BRFSS 2015')
    print('='*60)

    DATA_PATH = (PROJECT_DIR / 'data' / 'raw' / 'dataset_clasificacion' /
                 'diabetes_binary_health_indicators_BRFSS2015.csv')
    if not DATA_PATH.exists():
        print(f'  ERROR: {DATA_PATH} no encontrado.')
        sys.exit(1)

    data = pd.read_csv(DATA_PATH)
    print(f'  Datos: {data.shape[0]:,} filas × {data.shape[1]} columnas')

    target = 'Diabetes_binary'
    X = data.drop(columns=[target])
    y = data[target]
    print(f'  Desbalanceo: {(y==0).sum():,} sin diabetes | {(y==1).sum():,} con diabetes')
    print(f'  Ratio: {(y==0).sum()/(y==1).sum():.1f}:1')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    print(f'  Split: train={len(X_train):,} | test={len(X_test):,}')

    preprocessor = ColumnTransformer(transformers=[
        ('pass', SimpleImputer(strategy='median'), CLF_ALL_FEATURES)
    ])

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f'  scale_pos_weight: {ratio:.2f}')

    model_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=ratio,
            min_child_weight=5, gamma=0.1,
            random_state=SEED, n_jobs=-1, verbosity=0,
            eval_metric='logloss',
        )),
    ])

    print('  Entrenando XGBoost + scale_pos_weight...')
    model_pipe.fit(X_train, y_train)

    y_pred = model_pipe.predict(X_test)
    y_prob = model_pipe.predict_proba(X_test)[:, 1]

    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_prob)
    accuracy  = float((y_pred == y_test).mean())

    # Umbral óptimo por índice de Youden
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_test, y_prob)
    youden_idx = int(np.argmax(tpr_arr - fpr_arr))
    optimal_threshold = float(thresh_arr[youden_idx])

    print(f'  Recall={recall:.4f} | F1={f1:.4f} | AUC={auc:.4f} | Precision={precision:.4f}')
    print(f'  Umbral óptimo (Youden): {optimal_threshold:.4f}')

    # Importancia de variables
    feat_importance = dict(zip(
        CLF_ALL_FEATURES,
        model_pipe.named_steps['model'].feature_importances_.tolist()
    ))

    joblib.dump(model_pipe, MODELS_DIR / 'model_classification.joblib')
    joblib.dump(X.columns.tolist(), MODELS_DIR / 'features_classification.joblib')

    meta = {
        'nombre_modelo'       : 'XGBoost + scale_pos_weight (Diabetes BRFSS 2015)',
        'recall_test'         : round(recall, 4),
        'f1_test'             : round(f1, 4),
        'precision_test'      : round(precision, 4),
        'roc_auc_test'        : round(auc, 4),
        'accuracy_test'       : round(accuracy, 4),
        'optimal_threshold'   : round(optimal_threshold, 4),
        'fecha_entrenamiento' : str(datetime.now().date()),
        'numeric_features'    : CLF_NUMERIC_FEATURES,
        'ordinal_features'    : CLF_ORDINAL_FEATURES,
        'all_features'        : CLF_ALL_FEATURES,
        'feature_importance'  : feat_importance,
        'ratio_desbalanceo'   : round(float(ratio), 2),
    }
    joblib.dump(meta, MODELS_DIR / 'model_metadata_classification.joblib')
    print(f'  Guardado: model_classification.joblib ✔')
    return meta


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('='*60)
    print('  ENTRENAMIENTO DE MODELOS — PROYECTO FINAL')
    print('  Analítica de Datos · Universidad de Antioquia')
    print('='*60)

    reg_meta = train_regression()
    clf_meta = train_classification()

    print('\n' + '='*60)
    print('  RESUMEN FINAL')
    print('='*60)
    print(f'  Regresión  — R² (orig): {reg_meta["r2_test_original"]:.4f} | MAE: ${reg_meta["mae_test_mxn"]:,.0f} MXN')
    print(f'  Clasificación — AUC:    {clf_meta["roc_auc_test"]:.4f}  | F1: {clf_meta["f1_test"]:.4f}')
    print(f'  Modelos en: {MODELS_DIR}')
    print('='*60)
