"""
Analytix Intelligence — Plataforma de Modelos ML
NightPrice AI  ·  MediGuard
"""

import re
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="templates")

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

_cache: dict = {}

def _load(name: str):
    if name not in _cache:
        _cache[name] = joblib.load(MODELS_DIR / name)
    return _cache[name]


# ── Coordenadas de zonas clave CDMX (replicadas del notebook) ───────────────
LANDMARKS_V2 = {
    "dist_zocalo":    (19.4326, -99.1332),
    "dist_polanco":   (19.4338, -99.1918),
    "dist_condesa":   (19.4108, -99.1731),
    "dist_reforma":   (19.4226, -99.1676),
    "dist_airport":   (19.4363, -99.0727),
    "dist_santa_fe":  (19.3610, -99.2630),
    "dist_coyoacan":  (19.3502, -99.1624),
}


def _build_airbnb_row(data: dict, v1_meta: dict) -> pd.DataFrame:
    neigh   = data.get("neighbourhood", "Cuauhtémoc")
    lat_map = v1_meta["neighbourhood_latlon"]["latitude"]
    lon_map = v1_meta["neighbourhood_latlon"]["longitude"]
    lat     = float(lat_map.get(neigh, 19.42))
    lon     = float(lon_map.get(neigh, -99.16))

    rt      = data.get("room_type", "Entire home/apt")
    name    = str(data.get("name", "")).lower()
    mn      = max(1, int(data.get("minimum_nights", 2)))
    n_rev   = max(0, int(data.get("number_of_reviews", 0)))
    rpm     = max(0.0, float(data.get("reviews_per_month", 0.0)))
    h_list  = max(1, int(data.get("host_listings", 1)))
    avail   = max(0, min(365, int(data.get("availability_365", 180))))
    rev_ltm = max(0, int(data.get("reviews_ltm", 0)))

    ref_date = pd.Timestamp(v1_meta.get("ref_date", "2025-09-30"))
    lr_str   = data.get("last_review", "")
    if lr_str:
        try:
            lr_dt             = pd.Timestamp(lr_str)
            has_reviews       = 1
            days_since_review = max(0, (ref_date - lr_dt).days)
            is_recently_active= int(days_since_review < 180)
        except Exception:
            has_reviews = 0; days_since_review = 9999; is_recently_active = 0
    else:
        has_reviews = 0; days_since_review = 9999; is_recently_active = 0

    avail_ratio = avail / 365.0
    is_entire   = int(rt == "Entire home/apt")
    is_luxury_f = int(bool(re.search(r"luxury|lujo|deluxe|premium|vip|suite|penthouse", name)))

    m_beds = re.search(r"(\d+)\s*(?:bedroom|habitaci[oó]n|rec[aá]mara|bed\s*room|br\b)", name)

    # Distances
    dists: dict = {}
    for feat, (flat, flon) in LANDMARKS_V2.items():
        dists[feat] = float(np.sqrt((lat - flat) ** 2 + (lon - flon) ** 2) * 111.0)
    dists["dist_premium_zone"] = min(dists["dist_polanco"], dists["dist_condesa"], dists["dist_reforma"])

    row = {
        "latitude": lat, "longitude": lon,
        "minimum_nights": mn, "number_of_reviews": n_rev,
        "reviews_per_month": rpm, "calculated_host_listings_count": h_list,
        "availability_365": avail, "number_of_reviews_ltm": rev_ltm,
        "has_reviews": has_reviews, "days_since_review": days_since_review,
        "is_recently_active": is_recently_active,
        "log_minimum_nights":     float(np.log1p(mn)),
        "log_number_of_reviews":  float(np.log1p(n_rev)),
        "log_reviews_per_month":  float(np.log1p(rpm)),
        "log_host_listings":      float(np.log1p(h_list)),
        "log_reviews_ltm":        float(np.log1p(rev_ltm)),
        "availability_ratio":     avail_ratio,
        "reviews_per_listing":    n_rev / (h_list + 1),
        "high_min_nights":        int(mn > 7),
        "is_monthly_rental":      int(mn >= 28),
        "is_multi_listing_host":  int(h_list > 1),
        "is_entire_flag":         is_entire,
        "is_studio":   int(bool(re.search(r"studio|estudio", name))),
        "is_luxury":   is_luxury_f,
        "is_loft":     int(bool(re.search(r"loft", name))),
        "has_parking": int(bool(re.search(r"parking|estacionamiento|garage", name))),
        "has_pool":    int(bool(re.search(r"pool|alberca|piscina", name))),
        "has_terrace": int(bool(re.search(r"terrace|terraza|rooftop|balcon", name))),
        "n_bedrooms":  min(int(m_beds.group(1)), 6) if m_beds else 0,
        **dists,
        "has_wifi":     int(bool(re.search(r"wifi|wi-fi|internet", name))),
        "has_ac":       int(bool(re.search(r"\bac\b|a/c|aire\s*acond|air\s*cond", name))),
        "has_kitchen":  int(bool(re.search(r"kitchen|cocina|kitchenette", name))),
        "has_jacuzzi":  int(bool(re.search(r"jacuzzi|hidromasaje|whirlpool|tina", name))),
        "has_gym":      int(bool(re.search(r"\bgym\b|gimnasio|fitness", name))),
        "has_view":     int(bool(re.search(r"\bview\b|vista|panoram|skyline", name))),
        "is_apartment": int(bool(re.search(r"apartment|departamento|depto|dpto|apto", name))),
        "is_house":     int(bool(re.search(r"\bhouse\b|\bcasa\b|chalet|villa\b", name))),
        "luxury_score": min(len(re.findall(r"luxury|lujo|premium|deluxe|vip|penthouse|suite|exclusiv", name)), 5),
        "title_length": min(len(str(data.get("name", ""))), 120),
        "word_count":   min(len(str(data.get("name", "")).split()), 25),
        "is_professional_host": int(h_list >= 5),
        "review_momentum":   float(np.log1p(n_rev * rpm)),
        "luxury_entire":     is_luxury_f * is_entire,
        "active_and_booked": is_recently_active * (1.0 - avail_ratio),
        "log_avail_per_rev": float(np.log1p(avail / (n_rev + 1.0))),
        "room_type":    rt,
        "neighbourhood": neigh,
    }

    feat_v2  = _load("features_regression_v2.joblib")
    all_cols = feat_v2["numericas"] + feat_v2["categoricas"]
    return pd.DataFrame([row])[all_cols]


# ── Rutas ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    v1_meta  = _load("model_metadata_regression.joblib")
    v2_meta  = _load("model_metadata_regression_v2.joblib")
    clf_meta = _load("model_metadata_classification.joblib")

    neigh_medians = v1_meta.get("neighbourhood_stats", {}).get("mediana", {})

    return render_template(
        "index.html",
        neighbourhoods  = v1_meta["neighbourhoods"],
        room_types      = v1_meta["room_types"],
        neigh_medians   = neigh_medians,
        # Reg v2 metrics
        reg_r2_log      = round(v2_meta["r2_test_log"], 4),
        reg_r2_orig     = round(v2_meta["r2_test_original"], 4),
        reg_mae         = round(v2_meta["mae_test_mxn"], 0),
        reg_rmse        = round(v2_meta["rmse_test_mxn"], 0),
        reg_n_features  = v2_meta["n_features_total"],
        reg_n_new       = v2_meta["n_features_nuevas"],
        # Clf metrics
        clf_recall      = round(clf_meta["recall_test"], 4),
        clf_f1          = round(clf_meta["f1_test"], 4),
        clf_roc_auc     = round(clf_meta["roc_auc_test"], 4),
        clf_precision   = round(clf_meta["precision_test"], 4),
        clf_threshold   = round(clf_meta["optimal_threshold"], 3),
        clf_ratio       = round(clf_meta["ratio_desbalanceo"], 1),
    )


@app.route("/api/predict/airbnb", methods=["POST"])
def predict_airbnb():
    try:
        data    = request.get_json(force=True)
        v1_meta = _load("model_metadata_regression.joblib")
        model   = _load("model_regression_v2.joblib")

        X     = _build_airbnb_row(data, v1_meta)
        price = float(np.exp(model.predict(X)[0]))

        neigh      = data.get("neighbourhood", "")
        stats      = v1_meta.get("neighbourhood_stats", {})
        med_neigh  = float(stats.get("mediana", {}).get(neigh, v1_meta["price_median"]))
        pct_diff   = (price - med_neigh) / med_neigh * 100

        all_prices: dict = {}
        for nb in v1_meta["neighbourhoods"]:
            d2 = dict(data); d2["neighbourhood"] = nb
            try:
                X2 = _build_airbnb_row(d2, v1_meta)
                all_prices[nb] = round(float(np.exp(model.predict(X2)[0])), 0)
            except Exception:
                pass

        return jsonify({
            "ok": True,
            "price": round(price, 0),
            "median_neigh": round(med_neigh, 0),
            "pct_diff": round(pct_diff, 1),
            "all_prices": all_prices,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data      = request.get_json(force=True)
        clf_model = _load("model_classification.joblib")
        clf_meta  = _load("model_metadata_classification.joblib")

        features = clf_meta["all_features"]
        X        = pd.DataFrame([{f: float(data.get(f, 0)) for f in features}])[features]

        proba     = float(clf_model.predict_proba(X)[0, 1])
        threshold = clf_meta["optimal_threshold"]
        pred      = int(proba >= threshold)
        pct       = proba * 100

        risk = "high" if pct >= 60 else ("medium" if pct >= 30 else "low")

        return jsonify({
            "ok": True,
            "probability": round(proba, 4),
            "probability_pct": round(pct, 1),
            "prediction": pred,
            "threshold": round(threshold, 3),
            "risk": risk,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8501)
