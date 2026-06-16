import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
import plotly.express as px

from math import radians, sin, cos, sqrt, atan2

# =============================
# TITIK REFERENSI
# =============================

UGM = (-7.7705, 110.3772)

MALIOBORO = (-7.7926, 110.3658)

CITY_CENTER = (-7.7972, 110.3705)

# =============================
# HAVERSINE
# =============================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


# =============================
# KOORDINAT LOKASI
# =============================

location_coords = {

    "Ngaglik, Sleman": (-7.7060, 110.4010),
    "Depok, Sleman": (-7.7680, 110.4010),
    "Kalasan, Sleman": (-7.7670, 110.4720),
    "Mlati, Sleman": (-7.7420, 110.3600),
    "Sleman, Sleman": (-7.7150, 110.3550),
    "Ngemplak, Sleman": (-7.6940, 110.4300),
    "Gamping, Sleman": (-7.7990, 110.3210),
    "Godean, Sleman": (-7.7690, 110.2950),
    "Purwomartani, Sleman": (-7.7470, 110.4580),
    "Condong Catur, Sleman": (-7.7570, 110.4010),
    "Berbah, Sleman": (-7.8170, 110.4380),
    "Prambanan, Sleman": (-7.7520, 110.4920),
    "Kaliurang, Sleman": (-7.6000, 110.4200),
    "Sayegan, Sleman": (-7.7230, 110.2890),
    "Caturtunggal, Sleman": (-7.7560, 110.3850),
    "Pakem, Sleman": (-7.6640, 110.4210),
    "Moyudan, Sleman": (-7.7680, 110.2490),
    "Cebongan, Sleman": (-7.7310, 110.3420),
    "Tempel, Sleman": (-7.6500, 110.3230),
    "Turi, Sleman": (-7.6530, 110.3690),
    "Jombor, Sleman": (-7.7470, 110.3620),
    "Sidoarum, Sleman": (-7.7630, 110.3320)
}

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🏠 House Prediction App")

menu = st.sidebar.radio(
    "Menu",
    ["Prediksi Harga", "Evaluasi Model"]
)


# =========================
# LOAD MODEL & PREPROCESSOR
# =========================

@st.cache_resource
def load_assets():

    preprocessor = joblib.load("preprocessor.pkl")

    model = joblib.load("best_model.pkl")

    return preprocessor, model

preprocessor, best_model = load_assets()

def format_rupiah_adaptive(x):
    try:
        x = float(x)

        if x >= 1_000_000_000:
            return f"Rp {x/1_000_000_000:.2f} Miliar"
        elif x >= 1_000_000:
            return f"Rp {x/1_000_000:.2f} Juta"
        else:
            return f"Rp {x:,.0f}"

    except:
        return str(x)

# =========================
# LOAD EVALUATION RESULTS
# =========================

results_dict = {}

if os.path.exists("model_results.json"):
    with open("model_results.json", "r") as f:
        results_dict = json.load(f)
        
# =========================
# PREDICTION PAGE
# =========================
if menu == "Prediksi Harga":

    st.title("🏠 Prediksi Harga Rumah Sleman")
    st.write("Isi karakteristik rumah untuk memprediksi harga rumah")

    col1, col2 = st.columns(2)

    with col1:

        bed = st.number_input(
            "Jumlah Kamar Tidur",
            min_value=0,
            max_value=20,
            value=0
        )

        bath = st.number_input(
            "Jumlah Kamar Mandi",
            min_value=0,
            max_value=20,
            value=0
        )

        carport = st.number_input(
            "Jumlah Carport",
            min_value=0,
            max_value=10,
            value=0
        )

        surface_area = st.number_input(
            "Luas Tanah (m²)",
            min_value=0,
            value=0
        )

        building_area = st.number_input(
            "Luas Bangunan (m²)",
            min_value=0,
            value=0
        )

    with col2:

        location_options = ["Pilih Lokasi Rumah"] + list(location_coords.keys())

        listing_location = st.selectbox(
            "📍 Lokasi Rumah",
            location_options,
            index=0
        )

        if listing_location != "Pilih Lokasi Rumah":

            latitude, longitude = location_coords[listing_location]

            dist_ugm = haversine(
                latitude,
                longitude,
                UGM[0],
                UGM[1]
            )

            dist_malioboro = haversine(
                latitude,
                longitude,
                MALIOBORO[0],
                MALIOBORO[1]
            )

            dist_city_center = haversine(
                latitude,
                longitude,
                CITY_CENTER[0],
                CITY_CENTER[1]
            )

            st.info(
                f"""
    📍 Informasi Lokasi

    • Jarak ke UGM : ± {dist_ugm:.1f} km

    • Jarak ke Malioboro : ± {dist_malioboro:.1f} km

    • Jarak ke Pusat Kota : ± {dist_city_center:.1f} km
    """
                )
    if st.button("Predict", use_container_width=True):

        try:

            if (
                listing_location == "Pilih Lokasi Rumah"
                or bed <= 0
                or bath <= 0
                or surface_area <= 0
                or building_area <= 0
            ):
                st.warning(
                    "⚠️ Silakan lengkapi seluruh data rumah terlebih dahulu sebelum melakukan prediksi."
                )
                st.stop()

            latitude, longitude = location_coords[
                listing_location
            ]

            dist_ugm = haversine(
                latitude,
                longitude,
                UGM[0],
                UGM[1]
            )

            dist_malioboro = haversine(
                latitude,
                longitude,
                MALIOBORO[0],
                MALIOBORO[1]
            )

            dist_city_center = haversine(
                latitude,
                longitude,
                CITY_CENTER[0],
                CITY_CENTER[1]
            )

            input_df = pd.DataFrame([{
                "listing-location": listing_location,
                "bed": bed,
                "bath": bath,
                "carport": carport,
                "surface_area": surface_area,
                "building_area": building_area,
                "latitude": latitude,
                "longitude": longitude,
                "dist_ugm": dist_ugm,
                "dist_malioboro": dist_malioboro,
                "dist_city_center": dist_city_center
            }])

            st.subheader("📋 Data Input")
            st.dataframe(input_df)

            if isinstance(best_model, tf.keras.Model):

                X_processed = preprocessor.transform(input_df)

                if hasattr(X_processed, "toarray"):
                    X_processed = X_processed.toarray()

                pred = best_model.predict(
                    X_processed,
                    verbose=0
                ).flatten()[0]

            else:

                pred = best_model.predict(
                    input_df
                )[0]

            pred = np.expm1(pred)
            pred = max(float(pred), 0)

            st.markdown(
                f"""
                <div style="
                    padding:25px;
                    border-radius:15px;
                    text-align:center;
                    background:linear-gradient(135deg,#667eea,#764ba2);
                    color:white;
                ">
                    <h2>💰 Predicted House Price</h2>
                    <h1>{format_rupiah_adaptive(pred)}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

# =========================
# EVALUATION PAGE 
# =========================
elif menu == "Evaluasi Model":

    st.title("📊 Evaluasi Model")

    if not results_dict:
        st.warning("⚠️ model_results.json tidak ditemukan")
        st.stop()

    # =========================
    # LOAD DATAFRAME
    # =========================
    metrics_df = pd.DataFrame(results_dict).T.reset_index()
    metrics_df.rename(columns={"index": "Model"}, inplace=True)

    # =========================
    # NORMALISASI NAMA KOLOM
    # =========================
    metrics_df.rename(columns={
        "test_r2": "R²",
        "R2": "R²",
        "r2": "R²",

        "test_mae": "MAE",
        "mae": "MAE",

        "test_rmse": "RMSE",
        "rmse": "RMSE",

        "test_mape": "MAPE",
        "mape": "MAPE",
        "MAPE (%)": "MAPE"
    }, inplace=True)

    # =========================
    # KONVERSI NUMERIK
    # =========================
    for col in ["R²", "MAE", "RMSE", "MAPE"]:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(
                metrics_df[col],
                errors="coerce"
            )

    # =========================
    # FILTER KOLOM
    # =========================
    required_cols = [
        "Model",
        "R²",
        "MAE",
        "RMSE",
        "MAPE"
    ]

    available_cols = [
        col for col in required_cols
        if col in metrics_df.columns
    ]

    metrics_df = metrics_df[available_cols]

    # =========================
    # TABEL EVALUASI
    # =========================
    st.subheader("📌 Tabel Evaluasi Model")

    metrics_show = metrics_df.copy()

    if "R²" in metrics_show.columns:
        metrics_show["R²"] = metrics_show["R²"].round(4)

    if "MAE" in metrics_show.columns:
        metrics_show["MAE"] = metrics_show["MAE"].apply(
            format_rupiah_adaptive
        )

    if "RMSE" in metrics_show.columns:
        metrics_show["RMSE"] = metrics_show["RMSE"].apply(
            format_rupiah_adaptive
        )

    if "MAPE" in metrics_show.columns:
        metrics_show["MAPE"] = metrics_show["MAPE"].apply(
            lambda x: f"{x:.2f}%"
            if pd.notnull(x)
            else "-"
        )

    st.dataframe(
        metrics_show,
        use_container_width=True
    )

    # =========================
    # VISUALISASI ERROR METRICS
    # =========================
    st.subheader("📊 Perbandingan Error Model")

    error_metrics = [
        c for c in ["MAE", "RMSE"]
        if c in metrics_df.columns
    ]

    if error_metrics:

        fig = px.bar(
            metrics_df.melt(
                id_vars="Model",
                value_vars=error_metrics
            ),
            x="Model",
            y="value",
            color="variable",
            barmode="group",
            text_auto=True,
            title="Perbandingan MAE, RMSE"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("Data visualisasi tidak lengkap")

    # =========================
    # VISUALISASI R²
    # =========================
    if "R²" in metrics_df.columns:

        st.subheader("📈 Perbandingan R²")

        fig_r2 = px.bar(
            metrics_df,
            x="Model",
            y="R²",
            text_auto=".4f",
            title="Perbandingan Nilai R²"
        )

        st.plotly_chart(
            fig_r2,
            use_container_width=True
        )

    # =========================
    # MODEL TERBAIK
    # =========================
    if "R²" in metrics_df.columns:

        best_row = metrics_df.sort_values(
            "R²",
            ascending=False
        ).iloc[0]

        mape_text = "-"

        if "MAPE" in metrics_df.columns:
            mape_text = f"{best_row['MAPE']:.2f}%"

        st.success(
            f"""
🏆 Model Terbaik: {best_row['Model']}

R²   : {best_row['R²']:.4f}

MAE  : {format_rupiah_adaptive(best_row.get('MAE', 0))}

RMSE : {format_rupiah_adaptive(best_row.get('RMSE', 0))}

MAPE : {mape_text}
"""
        )

    else:
        st.warning(
            "Kolom R² tidak ditemukan, tidak bisa menentukan model terbaik"
        )

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    if hasattr(best_model, "feature_importances_"):

        st.subheader("🔍 Feature Importance")

        fi_df = pd.DataFrame({
            "Feature": range(
                len(best_model.feature_importances_)
            ),
            "Importance": best_model.feature_importances_
        })

        fig_fi = px.bar(
            fi_df,
            x="Feature",
            y="Importance",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig_fi,
            use_container_width=True
        )
