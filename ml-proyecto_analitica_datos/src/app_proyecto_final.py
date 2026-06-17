"""
Proyecto Final — Predictor de Modelos ML
Analítica de Datos · Universidad de Antioquia
"""

import warnings
warnings.filterwarnings("ignore")

import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import date

# ── Configuración ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Proyecto Final ML · UdeA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

.header-reg {
    background: linear-gradient(135deg, #FF385C 0%, #BD1E59 55%, #6A0F49 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.header-clf {
    background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 55%, #01257d 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.header-reg h1, .header-clf h1 { color: white !important; font-size: 2rem; margin: 0; }
.header-reg p,  .header-clf p  { color: rgba(255,255,255,.82); margin: .3rem 0 0; font-size: .9rem; }

.metric-card {
    background: white; border-radius: 12px; padding: 1rem 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,.08); text-align: center; margin-bottom: .5rem;
}
.metric-card .val { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700; }
.metric-card .lbl { font-size: .72rem; color: #888; text-transform: uppercase; letter-spacing: .08em; }
.metric-card .sub { font-size: .78rem; margin-top: .1rem; }
.mc-red  .val { color: #FF385C; } .mc-red  .sub { color: #FF385C; }
.mc-blue .val { color: #1a73e8; } .mc-blue .sub { color: #1a73e8; }
.mc-grn  .val { color: #10b981; } .mc-grn  .sub { color: #10b981; }
.mc-ora  .val { color: #f59e0b; } .mc-ora  .sub { color: #f59e0b; }

.result-box {
    border-radius: 16px; padding: 1.8rem 2rem; margin-top: 1rem; text-align: center;
}
.result-reg  { background: linear-gradient(135deg,#fff5f6,#ffe0e5); border: 2px solid #FF385C; }
.result-clf0 { background: linear-gradient(135deg,#f0f9ff,#dbeafe); border: 2px solid #1a73e8; }
.result-clf1 { background: linear-gradient(135deg,#fff7ed,#fde68a); border: 2px solid #f59e0b; }
.result-clf2 { background: linear-gradient(135deg,#fef2f2,#fecaca); border: 2px solid #ef4444; }
.result-price { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; color: #FF385C; }
.result-label { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; }
.result-sub   { color: #555; font-size: .9rem; margin-top: .5rem; }

.section-title {
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    padding-bottom: .35rem; margin: 1.2rem 0 .8rem;
}
.st-reg  { border-bottom: 2px solid #FF385C; color: #1a1a1a; }
.st-blue { border-bottom: 2px solid #1a73e8; color: #1a1a1a; }

[data-testid="stSidebar"] { background: #111827 !important; }
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #9ca3af !important; font-size:.8rem; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: #f3f4f6; border-radius: 8px 8px 0 0;
    padding: .45rem 1.1rem;
    font-family: 'Syne', sans-serif; font-weight: 600; font-size: .85rem;
}

.info-badge {
    display: inline-block; background: #f3f4f6;
    border-radius: 6px; padding: .2rem .6rem;
    font-size: .78rem; color: #374151; margin: .15rem;
}
</style>
""", unsafe_allow_html=True)

# ── Carga de modelos ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_meta(path):
    return joblib.load(path)

def models_ready(model_file):
    return (MODELS_DIR / model_file).exists()


# ── Feature engineering (réplica del Lab 4) ─────────────────────────────────
def build_regression_input(inputs: dict, meta: dict) -> pd.DataFrame:
    neigh          = inputs['neighbourhood']
    lat_map        = meta['neighbourhood_latlon']['latitude']
    lon_map        = meta['neighbourhood_latlon']['longitude']
    neigh_freq_map = meta['neighbourhood_freq']

    lat  = lat_map.get(neigh, float(np.mean(list(lat_map.values()))))
    lon  = lon_map.get(neigh, float(np.mean(list(lon_map.values()))))

    ref_date   = pd.Timestamp(meta.get('ref_date', '2025-09-30'))
    last_rev   = inputs.get('last_review')
    if last_rev:
        lr_dt             = pd.Timestamp(last_rev)
        has_reviews       = 1
        days_since_review = max(0, (ref_date - lr_dt).days)
        is_recently_active= int(days_since_review < 180)
    else:
        has_reviews        = 0
        days_since_review  = 9999
        is_recently_active = 0

    mn      = inputs['minimum_nights']
    n_rev   = inputs['number_of_reviews']
    rpm     = inputs['reviews_per_month']
    h_list  = inputs['calculated_host_listings_count']
    avail   = inputs['availability_365']
    rev_ltm = inputs['number_of_reviews_ltm']
    rt      = inputs['room_type']
    name    = str(inputs.get('name', '')).lower()

    mean_freq = float(np.mean(list(neigh_freq_map.values())))

    m_beds = re.search(r'(\d+)\s*(?:bedroom|habitaci[oó]n|rec[aá]mara|bed\s*room|br\b)', name)

    row = {
        'latitude'                        : lat,
        'longitude'                       : lon,
        'minimum_nights'                  : mn,
        'number_of_reviews'               : n_rev,
        'reviews_per_month'               : rpm,
        'calculated_host_listings_count'  : h_list,
        'availability_365'                : avail,
        'number_of_reviews_ltm'           : rev_ltm,
        'has_reviews'                     : has_reviews,
        'days_since_review'               : days_since_review,
        'is_recently_active'              : is_recently_active,
        'log_minimum_nights'              : np.log1p(mn),
        'log_number_of_reviews'           : np.log1p(n_rev),
        'log_reviews_per_month'           : np.log1p(rpm),
        'log_host_listings'               : np.log1p(h_list),
        'log_reviews_ltm'                 : np.log1p(rev_ltm),
        'availability_ratio'              : avail / 365.0,
        'reviews_per_listing'             : n_rev / (h_list + 1),
        'high_min_nights'                 : int(mn > 7),
        'is_monthly_rental'               : int(mn >= 28),
        'is_multi_listing_host'           : int(h_list > 1),
        'is_entire_flag'                  : int(rt == 'Entire home/apt'),
        'is_studio'                       : int(bool(re.search(r'studio|estudio', name))),
        'is_luxury'                       : int(bool(re.search(r'luxury|lujo|deluxe|premium|vip|suite|penthouse', name))),
        'is_loft'                         : int(bool(re.search(r'loft', name))),
        'has_parking'                     : int(bool(re.search(r'parking|estacionamiento|garage', name))),
        'has_pool'                        : int(bool(re.search(r'pool|alberca|piscina', name))),
        'has_terrace'                     : int(bool(re.search(r'terrace|terraza|rooftop|balcon', name))),
        'n_bedrooms'                      : min(int(m_beds.group(1)), 6) if m_beds else 0,
        'neighbourhood_freq'              : neigh_freq_map.get(neigh, mean_freq),
        'room_type'                       : rt,
        'neighbourhood'                   : neigh,
    }
    return pd.DataFrame([row])


# ── Gauge chart ──────────────────────────────────────────────────────────────
def gauge_chart(value_pct: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value_pct, 1),
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': '%', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar':  {'color': color},
            'steps': [
                {'range': [0,  30], 'color': '#dcfce7'},
                {'range': [30, 60], 'color': '#fef9c3'},
                {'range': [60, 100],'color': '#fee2e2'},
            ],
            'threshold': {
                'line': {'color': '#1a1a1a', 'width': 3},
                'thickness': 0.75, 'value': value_pct,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Proyecto Final ML")
    st.markdown("**Analítica de Datos · UdeA**")
    st.markdown("---")

    modelo_sel = st.radio(
        "Selecciona el modelo",
        ["🏠 Regresión — Precio Airbnb CDMX", "🩺 Clasificación — Riesgo de Diabetes"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if modelo_sel.startswith("🏠"):
        st.markdown("""
        **Modelo:** XGBoost Regresión
        **Dataset:** Inside Airbnb — CDMX
        **Target:** Precio por noche (MXN)
        **Features:** 30 variables + OHE
        """)
    else:
        st.markdown("""
        **Modelo:** XGBoost Clasificación
        **Dataset:** BRFSS 2015 — CDC
        **Target:** Diabetes_binary (0/1)
        **Features:** 21 indicadores de salud
        """)

    st.markdown("---")
    st.caption("Lab 4 & 5 · Analítica de Datos · UdeA")


# ═══════════════════════════════════════════════════════════════════════════
#  REGRESIÓN — Airbnb CDMX
# ═══════════════════════════════════════════════════════════════════════════
if modelo_sel.startswith("🏠"):

    st.markdown("""
    <div class="header-reg">
        <h1>🏠 Predictor de Precio Airbnb — Ciudad de México</h1>
        <p>Laboratorio 4 · Regresión con Ingeniería de Características · XGBoost tuned</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ready('model_regression.joblib'):
        st.error("""
        **Modelos no encontrados.** Ejecuta primero el script de entrenamiento:
        ```bash
        cd ml-proyecto_analitica_datos
        python src/train_models.py
        ```
        """)
        st.stop()

    reg_model = load_model(MODELS_DIR / 'model_regression.joblib')
    reg_meta  = load_meta(MODELS_DIR / 'model_metadata_regression.joblib')

    # ── Métricas del modelo ──────────────────────────────────────────────────
    st.markdown('<div class="section-title st-reg">Desempeño del Modelo (Test Hold-Out 30%)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metrics_reg = [
        (c1, "mc-red",  f"{reg_meta['r2_test_original']:.4f}", "R² Test",          "escala original MXN"),
        (c2, "mc-blue", f"{reg_meta['r2_test_log']:.4f}",      "R² Test (log)",    "escala entrenamiento"),
        (c3, "mc-grn",  f"${reg_meta['mae_test_mxn']:,.0f}",   "MAE",              "MXN / noche"),
        (c4, "mc-ora",  f"{reg_meta['gap_r2']:.4f}",           "Gap R²",           "train − test"),
    ]
    for col, cls, val, lbl, sub in metrics_reg:
        col.markdown(f"""
        <div class="metric-card {cls}">
            <div class="lbl">{lbl}</div>
            <div class="val">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title st-reg">Ingresa las características del listing</div>', unsafe_allow_html=True)

    neighbourhoods = reg_meta['neighbourhoods']
    room_types     = reg_meta['room_types']

    with st.form("form_reg"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Ubicación y tipo**")
            neighbourhood = st.selectbox("Alcaldía (neighbourhood)", neighbourhoods, index=0)
            room_type     = st.selectbox("Tipo de habitación (room_type)", room_types, index=0)
            name          = st.text_input(
                "Título del listing (name)",
                value="Cozy Studio with Parking",
                help="Se extraen keywords: studio, loft, luxury, parking, pool, terrace, n° bedrooms"
            )

            st.markdown("**Disponibilidad y estadías**")
            minimum_nights = st.number_input("Noches mínimas", min_value=1, max_value=365, value=2)
            availability   = st.slider("Disponibilidad (días/año)", 0, 365, 180)

        with col_b:
            st.markdown("**Reseñas y actividad**")
            number_of_reviews  = st.number_input("Total de reseñas", min_value=0, max_value=2000, value=25)
            reviews_per_month  = st.number_input("Reseñas por mes", min_value=0.0, max_value=50.0, value=1.5, step=0.1)
            reviews_ltm        = st.number_input("Reseñas últimos 12 meses", min_value=0, max_value=500, value=10)
            host_listings      = st.number_input("Listings del host", min_value=1, max_value=100, value=1)

            st.markdown("**Última reseña (opcional)**")
            tiene_resenas = st.toggle("¿Tiene reseñas registradas?", value=True)
            if tiene_resenas:
                last_review_date = st.date_input("Fecha de última reseña", value=date(2024, 6, 1))
            else:
                last_review_date = None

        submitted_reg = st.form_submit_button("🔮 Predecir precio por noche", use_container_width=True)

    if submitted_reg:
        inputs = {
            'neighbourhood'                  : neighbourhood,
            'room_type'                      : room_type,
            'name'                           : name,
            'minimum_nights'                 : minimum_nights,
            'availability_365'               : availability,
            'number_of_reviews'              : number_of_reviews,
            'reviews_per_month'              : reviews_per_month,
            'number_of_reviews_ltm'          : reviews_ltm,
            'calculated_host_listings_count' : host_listings,
            'last_review'                    : last_review_date,
        }

        with st.spinner("Calculando predicción..."):
            X_input    = build_regression_input(inputs, reg_meta)
            log_pred   = reg_model.predict(X_input)[0]
            price_pred = float(np.exp(log_pred))

        # Stats del vecindario
        stats_nb = reg_meta.get('neighbourhood_stats', {})
        med_nb   = stats_nb.get('mediana', {}).get(neighbourhood, reg_meta['price_median'])
        med_nb   = float(med_nb)

        pct_diff = (price_pred - med_nb) / med_nb * 100

        st.markdown(f"""
        <div class="result-box result-reg">
            <div style="font-size:.85rem;color:#888;margin-bottom:.3rem">Precio estimado por noche</div>
            <div class="result-price">${price_pred:,.0f} <span style="font-size:1.2rem">MXN</span></div>
            <div class="result-sub">
                Mediana en <strong>{neighbourhood}</strong>: ${med_nb:,.0f} MXN/noche &nbsp;·&nbsp;
                Tu listing está <strong>{abs(pct_diff):.0f}%
                {"por encima" if pct_diff > 0 else "por debajo"}</strong> de la mediana del vecindario
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gráfico de comparación
        st.markdown("")
        col_g1, col_g2 = st.columns([2, 1])

        with col_g1:
            neigh_data = {}
            for nb in neighbourhoods[:15]:
                inp_nb = dict(inputs)
                inp_nb['neighbourhood'] = nb
                X_nb = build_regression_input(inp_nb, reg_meta)
                neigh_data[nb] = float(np.exp(reg_model.predict(X_nb)[0]))

            df_comp = (pd.DataFrame.from_dict(neigh_data, orient='index', columns=['Precio'])
                       .sort_values('Precio', ascending=True))

            colors = ['#FF385C' if nb == neighbourhood else '#d1d5db' for nb in df_comp.index]
            fig_comp = go.Figure(go.Bar(
                x=df_comp['Precio'], y=df_comp.index,
                orientation='h', marker_color=colors,
                text=[f'${v:,.0f}' for v in df_comp['Precio']],
                textposition='outside',
            ))
            fig_comp.update_layout(
                title=f'Comparativa de precio estimado por alcaldía (primeras 15)',
                xaxis_title='Precio MXN/noche', height=420,
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=10, r=80, t=40, b=10),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_g2:
            st.markdown("**Características detectadas en el título:**")
            feat_flags = {
                'Studio/Estudio': int(bool(re.search(r'studio|estudio', name.lower()))),
                'Luxury/Premium': int(bool(re.search(r'luxury|lujo|deluxe|premium|vip|suite|penthouse', name.lower()))),
                'Loft':           int(bool(re.search(r'loft', name.lower()))),
                'Parking':        int(bool(re.search(r'parking|estacionamiento|garage', name.lower()))),
                'Pool/Alberca':   int(bool(re.search(r'pool|alberca|piscina', name.lower()))),
                'Terraza/Rooftop':int(bool(re.search(r'terrace|terraza|rooftop|balcon', name.lower()))),
            }
            for feat, val in feat_flags.items():
                icon = "✅" if val else "⬜"
                st.markdown(f"{icon} {feat}")

            m_beds_app = re.search(r'(\d+)\s*(?:bedroom|habitaci[oó]n|rec[aá]mara|bed\s*room|br\b)', name.lower())
            n_beds_app = min(int(m_beds_app.group(1)), 6) if m_beds_app else 0
            st.markdown(f"🛏️ Recámaras detectadas: **{n_beds_app}**")
            st.markdown(f"📅 Actividad reciente: **{'Sí' if inputs.get('last_review') else 'No'}**")


# ═══════════════════════════════════════════════════════════════════════════
#  CLASIFICACIÓN — Diabetes
# ═══════════════════════════════════════════════════════════════════════════
else:

    st.markdown("""
    <div class="header-clf">
        <h1>🩺 Predictor de Riesgo de Diabetes</h1>
        <p>Laboratorio 5 · Clasificación con Datos Desbalanceados · XGBoost + scale_pos_weight · BRFSS 2015</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ready('model_classification.joblib'):
        st.error("""
        **Modelos no encontrados.** Ejecuta primero el script de entrenamiento:
        ```bash
        cd ml-proyecto_analitica_datos
        python src/train_models.py
        ```
        """)
        st.stop()

    clf_model = load_model(MODELS_DIR / 'model_classification.joblib')
    clf_meta  = load_meta(MODELS_DIR / 'model_metadata_classification.joblib')

    # ── Métricas del modelo ──────────────────────────────────────────────────
    st.markdown('<div class="section-title st-blue">Desempeño del Modelo (Test Hold-Out 30%)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_clf = [
        (c1, "mc-red",  f"{clf_meta['recall_test']:.4f}",    "Recall",    "↑ prioridad clínica"),
        (c2, "mc-blue", f"{clf_meta['f1_test']:.4f}",        "F1-Score",  "balance R/P"),
        (c3, "mc-grn",  f"{clf_meta['roc_auc_test']:.4f}",   "ROC-AUC",   "discriminación"),
        (c4, "mc-ora",  f"{clf_meta['precision_test']:.4f}", "Precision", "exactitud positivos"),
        (c5, "mc-red",  f"{clf_meta['optimal_threshold']:.2f}", "Umbral óptimo", "índice de Youden"),
    ]
    for col, cls, val, lbl, sub in metrics_clf:
        col.markdown(f"""
        <div class="metric-card {cls}">
            <div class="lbl">{lbl}</div>
            <div class="val">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.info(
        "**Prioridad clínica:** Recall > F1 > AUC > Precision. Minimizar Falsos Negativos "
        "(diabéticos no detectados) es más crítico que evitar falsas alarmas."
    )

    # ── Formulario ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title st-blue">Ingresa los indicadores de salud del paciente</div>', unsafe_allow_html=True)

    ETIQUETAS_GENHLT = {1:"Excelente",2:"Muy buena",3:"Buena",4:"Regular",5:"Pobre"}
    ETIQUETAS_AGE    = {
        1:"18–24",2:"25–29",3:"30–34",4:"35–39",5:"40–44",6:"45–49",
        7:"50–54",8:"55–59",9:"60–64",10:"65–69",11:"70–74",12:"75–79",13:"80+"
    }
    ETIQUETAS_EDU = {
        1:"Sin educación formal",2:"Primaria incompleta",3:"Primaria completa",
        4:"Secundaria / Bachillerato",5:"Técnico / Tecnólogo",6:"Universidad o más"
    }
    ETIQUETAS_INC = {
        1:"< $10,000",2:"$10,000–$14,999",3:"$15,000–$19,999",
        4:"$20,000–$24,999",5:"$25,000–$34,999",6:"$35,000–$49,999",
        7:"$50,000–$74,999",8:"> $75,000"
    }

    with st.form("form_clf"):
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Métricas de Salud", "🏥 Historial Médico", "🏃 Estilo de Vida", "👤 Demografía"])

        with tab1:
            st.markdown("**Métricas cuantitativas**")
            col1, col2, col3 = st.columns(3)
            bmi      = col1.number_input("IMC (BMI)", min_value=10.0, max_value=100.0, value=27.0, step=0.5,
                                          help="Índice de Masa Corporal. Normal: 18.5–24.9 | Sobrepeso: 25–29.9 | Obeso: ≥30")
            ment_hlth= col2.number_input("Días con mala salud mental (último mes)", min_value=0, max_value=30, value=0,
                                          help="Días en el último mes en que la salud mental no fue buena")
            phys_hlth= col3.number_input("Días con mala salud física (último mes)", min_value=0, max_value=30, value=0,
                                          help="Días en el último mes en que hubo enfermedad o lesión física")

            st.markdown("**Estado de salud general y presión/colesterol**")
            col4, col5, col6, col7 = st.columns(4)
            gen_hlth  = col4.selectbox("Salud general (GenHlth)", options=list(ETIQUETAS_GENHLT.keys()),
                                        format_func=lambda x: ETIQUETAS_GENHLT[x], index=2)
            high_bp   = col5.selectbox("Presión arterial alta (HighBP)", [0, 1],
                                        format_func=lambda x: "Sí" if x else "No")
            high_chol = col6.selectbox("Colesterol alto (HighChol)", [0, 1],
                                        format_func=lambda x: "Sí" if x else "No")
            chol_check= col7.selectbox("Chequeo de colesterol en 5 años (CholCheck)", [0, 1],
                                        format_func=lambda x: "Sí" if x else "No", index=1)

        with tab2:
            col_m1, col_m2, col_m3 = st.columns(3)
            stroke     = col_m1.selectbox("Accidente cerebrovascular (Stroke)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No")
            heart_dis  = col_m2.selectbox("Cardiopatía coronaria / Infarto (HeartDiseaseorAttack)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No")
            diff_walk  = col_m3.selectbox("Dificultad para caminar (DiffWalk)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No")

            col_m4, col_m5 = st.columns(2)
            any_hc     = col_m4.selectbox("¿Tiene cobertura médica? (AnyHealthcare)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No", index=1)
            no_doc_cost= col_m5.selectbox("¿No fue al médico por costo? (NoDocbcCost)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No")

        with tab3:
            col_l1, col_l2, col_l3 = st.columns(3)
            smoker     = col_l1.selectbox("Fumador (Smoker) ≥100 cigarrillos en vida", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No")
            phys_act   = col_l2.selectbox("Actividad física en últimos 30 días (PhysActivity)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No", index=1)
            hvy_alc    = col_l3.selectbox("Consumo elevado de alcohol (HvyAlcoholConsump)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No")
            col_l4, col_l5 = st.columns(2)
            fruits     = col_l4.selectbox("Consume frutas diariamente (Fruits)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No", index=1)
            veggies    = col_l5.selectbox("Consume verduras diariamente (Veggies)", [0, 1],
                                           format_func=lambda x: "Sí" if x else "No", index=1)

        with tab4:
            col_d1, col_d2 = st.columns(2)
            sex        = col_d1.selectbox("Sexo (Sex)", [0, 1],
                                           format_func=lambda x: "Femenino (0)" if x == 0 else "Masculino (1)")
            age        = col_d2.selectbox("Grupo de edad (Age)", options=list(ETIQUETAS_AGE.keys()),
                                           format_func=lambda x: ETIQUETAS_AGE[x], index=6)

            col_d3, col_d4 = st.columns(2)
            education  = col_d3.selectbox("Nivel educativo (Education)", options=list(ETIQUETAS_EDU.keys()),
                                           format_func=lambda x: ETIQUETAS_EDU[x], index=4)
            income     = col_d4.selectbox("Nivel de ingresos anuales (Income, USD)", options=list(ETIQUETAS_INC.keys()),
                                           format_func=lambda x: ETIQUETAS_INC[x], index=4)

        submitted_clf = st.form_submit_button("🔬 Evaluar riesgo de diabetes", use_container_width=True)

    if submitted_clf:
        X_clf = pd.DataFrame([{
            'BMI'                 : bmi,
            'MentHlth'            : ment_hlth,
            'PhysHlth'            : phys_hlth,
            'HighBP'              : high_bp,
            'HighChol'            : high_chol,
            'CholCheck'           : chol_check,
            'Smoker'              : smoker,
            'Stroke'              : stroke,
            'HeartDiseaseorAttack': heart_dis,
            'PhysActivity'        : phys_act,
            'Fruits'              : fruits,
            'Veggies'             : veggies,
            'HvyAlcoholConsump'   : hvy_alc,
            'AnyHealthcare'       : any_hc,
            'NoDocbcCost'         : no_doc_cost,
            'DiffWalk'            : diff_walk,
            'Sex'                 : sex,
            'GenHlth'             : gen_hlth,
            'Age'                 : age,
            'Education'           : education,
            'Income'              : income,
        }])

        with st.spinner("Calculando riesgo..."):
            proba   = float(clf_model.predict_proba(X_clf)[0, 1])
            umbral  = clf_meta['optimal_threshold']
            pred    = int(proba >= umbral)

        pct = proba * 100

        if pct < 30:
            risk_class, risk_label, risk_color, box_class = "BAJO", "Bajo riesgo", "#10b981", "result-clf0"
        elif pct < 60:
            risk_class, risk_label, risk_color, box_class = "MODERADO", "Riesgo moderado", "#f59e0b", "result-clf1"
        else:
            risk_class, risk_label, risk_color, box_class = "ALTO", "Riesgo elevado", "#ef4444", "result-clf2"

        pred_label = "Positivo (diabetes/prediabetes)" if pred == 1 else "Negativo (sin diabetes)"

        st.markdown(f"""
        <div class="result-box {box_class}">
            <div style="font-size:.85rem;color:#666;margin-bottom:.3rem">Probabilidad de diabetes</div>
            <div class="result-label" style="color:{risk_color};">{pct:.1f}% — {risk_label}</div>
            <div class="result-sub">
                Clasificación con umbral óptimo ({umbral:.2f}): <strong>{pred_label}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Columnas resultado
        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            st.plotly_chart(gauge_chart(pct, "Probabilidad de Diabetes", risk_color), use_container_width=True)

            st.markdown(f"""
            | Métrica | Valor |
            |---|---|
            | Probabilidad | **{pct:.2f}%** |
            | Umbral usado | {umbral:.2f} (Youden) |
            | Clasificación | **{'Positivo' if pred else 'Negativo'}** |
            | Riesgo | **{risk_class}** |
            """)

        with col_r2:
            # Top factores de riesgo del paciente (según importancia del modelo)
            feat_imp = clf_meta.get('feature_importance', {})
            patient_factors = []

            risk_map = {
                'HighBP'              : ('Presión arterial alta', high_bp,    1, 'factor de riesgo'),
                'HighChol'            : ('Colesterol alto',       high_chol,  1, 'factor de riesgo'),
                'BMI'                 : ('IMC elevado (obeso)',    bmi,       30, 'IMC ≥ 30 es factor de riesgo'),
                'GenHlth'             : ('Salud general pobre',   gen_hlth,   4, 'GenHlth ≥ 4 = riesgo'),
                'Age'                 : ('Edad avanzada',          age,        9, 'Mayor riesgo a partir de 60–64'),
                'DiffWalk'            : ('Dificultad para caminar',diff_walk,  1, 'asociado a diabetes'),
                'HeartDiseaseorAttack': ('Cardiopatía',            heart_dis,  1, 'comorbilidad frecuente'),
                'Stroke'              : ('ACV previo',             stroke,     1, 'factor de riesgo'),
                'PhysActivity'        : ('Inactividad física',     phys_act,   0, 'actividad = protector'),
                'HvyAlcoholConsump'   : ('Consumo elevado alcohol',hvy_alc,   1, 'factor de riesgo'),
                'Smoker'              : ('Tabaquismo',              smoker,     1, 'factor de riesgo'),
            }

            for feat, (desc, val, threshold, note) in risk_map.items():
                imp = feat_imp.get(feat, 0)
                if feat == 'PhysActivity':
                    active_risk = (val == threshold)  # inactividad = riesgo
                elif feat in ('BMI', 'GenHlth', 'Age'):
                    active_risk = (val >= threshold)
                else:
                    active_risk = (val == threshold)

                patient_factors.append({
                    'Variable'    : feat,
                    'Descripción' : desc,
                    'Valor'       : val,
                    'Importancia' : imp,
                    'Activo'      : active_risk,
                    'Nota'        : note,
                })

            df_factors = (pd.DataFrame(patient_factors)
                          .sort_values('Importancia', ascending=False)
                          .head(8))

            st.markdown("**Factores de riesgo del paciente (top 8 por importancia del modelo):**")

            fig_bars = go.Figure()
            colors_f = ['#ef4444' if row['Activo'] else '#93c5fd' for _, row in df_factors.iterrows()]
            fig_bars.add_trace(go.Bar(
                x=df_factors['Importancia'],
                y=df_factors['Descripción'],
                orientation='h',
                marker_color=colors_f,
                text=[f"{'⚠️ Activo' if r['Activo'] else '✅ OK'}" for _, r in df_factors.iterrows()],
                textposition='outside',
            ))
            fig_bars.update_layout(
                title='Importancia × estado del paciente<br><sub>Rojo = factor activo | Azul = sin riesgo</sub>',
                xaxis_title='Importancia en el modelo',
                height=320, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=10, r=80, t=50, b=10),
            )
            st.plotly_chart(fig_bars, use_container_width=True)

        st.markdown("---")
        st.info(
            "⚕️ **Aviso médico:** Esta herramienta es un prototipo educativo de ML. "
            "Las predicciones NO constituyen diagnóstico médico. Consulta siempre a un profesional de la salud."
        )


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 Proyecto Final · Analítica de Datos · Universidad de Antioquia  "
    "| Lab 4: Regresión Airbnb CDMX  |  Lab 5: Clasificación Diabetes BRFSS 2015"
)
