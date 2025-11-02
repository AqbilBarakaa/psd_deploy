import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objs as go
from datetime import datetime, timedelta

# CONFIG
MODEL_PATH = "models/knn_model.pkl"
SCALER_PATH = "models/minmax_scaler.pkl"
MAX_DAYS = 30

st.set_page_config(page_title="NO₂ Multi-step Forecast (KNN)", layout="wide")
st.title("Prediksi Kadar NO₂ — Multi-step Forecast (KNN + MinMaxScaler)")

# CEK MODEL DAN SCALER
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("File model atau scaler tidak ditemukan. Pastikan ada di folder `models/`:\n- knn_model.pkl\n- minmax_scaler.pkl")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

data_min = scaler.data_min_
data_max = scaler.data_max_
data_range = data_max - data_min

# INPUT SECTION
st.markdown("### Pengaturan Prediksi")

days_to_predict = st.number_input("Berapa hari ke depan (H)?", min_value=1, max_value=MAX_DAYS, value=3, step=1)
threshold = 8.347e-05  # mol/m²
show_ugm3 = st.checkbox("Tampilkan juga hasil konversi ke µg/m³", value=True)

st.markdown(f"**Threshold Kadar NO₂:** {threshold:.8e} mol/m²")
st.markdown("---")

st.subheader("Input 2 Hari Terakhir (lag=2)")
st.info("Masukkan 2 nilai NO₂ (mol/m²) secara manual:")

col1, col2 = st.columns([2, 1])
with col1:
    input_t2 = st.number_input("NO₂(t-2) [mol/m²]", min_value=0.0, format="%.10f", value=0.00008951)
    input_t1 = st.number_input("NO₂(t-1) [mol/m²]", min_value=0.0, format="%.10f", value=0.00009666)
with col2:
    st.markdown("**Preview Input:**")
    st.metric("NO₂(t-2)", f"{input_t2:.8f}")
    st.metric("NO₂(t-1)", f"{input_t1:.8f}")

# HELPER FUNCTIONS
def normalize_pair(arr_orig):
    norm0 = (arr_orig[0] - data_min[0]) / (data_range[0] if data_range[0] != 0 else 1.0)
    norm1 = (arr_orig[1] - data_min[1]) / (data_range[1] if data_range[1] != 0 else 1.0)
    return np.array([norm0, norm1])

def normalize_value_col1(val):
    return (val - data_min[1]) / (data_range[1] if data_range[1] != 0 else 1.0)

def to_ugm3(mol_m2):
    return mol_m2 * 46010

def multi_step_recursive(last2_orig, n_steps, model, scaler):
    preds = []
    w_norm = normalize_pair(last2_orig)
    w_orig = np.array(last2_orig, dtype=float)
    for step in range(1, n_steps + 1):
        X_in = w_norm.reshape(1, -1)
        pred_orig = model.predict(X_in)[0]
        preds.append(float(pred_orig))
        norm_pred_col1 = normalize_value_col1(pred_orig)
        w_norm = np.array([w_norm[1], norm_pred_col1])
        w_orig = np.array([w_orig[1], pred_orig])
    return preds

# RUN PREDIKSI
run_btn = st.button("Jalankan Multi-step Forecast")
if run_btn:
    last2_raw = np.array([input_t2, input_t1], dtype=float)
    last2_norm = normalize_pair(last2_raw)
    preds = multi_step_recursive(last2_raw, int(days_to_predict), model, scaler)

    # OUTPUT DETAIL TEKS
    st.markdown("### **Detail Perhitungan Prediksi per Hari**")

    st.markdown(f"""
**Data 2 hari terakhir (Original):**
 - NO₂(t-2): {last2_raw[0]:.8f} mol/m²
 - NO₂(t-1): {last2_raw[1]:.8f} mol/m²

**Data 2 hari terakhir (Normalized - MinMaxScaler):**
 - NO₂(t-2): {last2_norm[0]:.4f}
 - NO₂(t-1): {last2_norm[1]:.4f}

**HASIL PREDIKSI NO₂ PER HARI:**
──────────────────────────────
""")

    for i, pred in enumerate(preds, start=1):
        diff = pred - threshold
        pct = (diff / threshold) * 100
        status = "BERBAHAYA" if pred > threshold else "AMAN"

        st.markdown(f"""
**Hari ke-{i}:**
 - Predicted NO₂: {pred:.8f} mol/m²
 - Threshold: {threshold:.8f}
 - Status: {status}
 - Difference: {diff:+.8f}
 - Percentage: {pct:+.2f}% dari threshold
""")

        # Tampilkan warning sesuai status
        if status == 'BERBAHAYA':
            if pct > 20:
                st.error(f"**PERINGATAN TINGGI:** NO₂ {pct:.1f}% di atas threshold → **SANGAT BERBAHAYA!**")
            elif pct > 10:
                st.warning(f"**PERINGATAN SEDANG:** {pct:.1f}% di atas threshold → **Waspada!**")
            else:
                st.error(f"**PERINGATAN:** {pct:.1f}% di atas threshold → **Perlu monitoring.**")
        else:
            if abs(pct) < 5:
                st.warning(f"**HATI-HATI:** hanya {abs(pct):.1f}% di bawah threshold → **Masih aman, tapi mendekati batas.**")
            else:
                st.success(f"**AMAN:** {abs(pct):.1f}% di bawah threshold → **Kualitas udara baik.**")

        st.markdown("---")

    # VISUALISASI
    base_date = datetime.utcnow()
    # Hari ini (t) = prediksi pertama
    past_dates = [
        (base_date - timedelta(days=2)).strftime("%Y-%m-%d"),
        (base_date - timedelta(days=1)).strftime("%Y-%m-%d"),
        base_date.strftime("%Y-%m-%d")  # hari ini!
    ]
    future_dates = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, len(preds))]
    full_dates = past_dates + future_dates

    # Nilai hari ini = preds[0]
    full_values = [input_t2, input_t1, preds[0]] + preds[1:]

    df_out = pd.DataFrame({
        "Tanggal": full_dates,
        "NO2(mol/m²)": full_values
    })
    if show_ugm3:
        df_out["NO2(µg/m³)"] = df_out["NO2(mol/m²)"].apply(to_ugm3)
    df_out["Status"] = np.where(df_out["NO2(mol/m²)"] > threshold, "BERBAHAYA", "AMAN")

    st.markdown("### Visualisasi Nilai NO₂ (2 Hari Sebelumnya + Prediksi)")
    colors = ["#2E86AB", "#2E86AB"] + ["red" if s == "BERBAHAYA" else "green" for s in df_out["Status"][2:]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_out["Tanggal"],
        y=df_out["NO2(mol/m²)"],
        marker_color=colors,
        name="NO₂ (mol/m²)",
        hovertemplate="Tanggal: %{x}<br>NO₂: %{y:.8f}"
    ))
    fig.add_trace(go.Scatter(
        x=df_out["Tanggal"],
        y=[threshold]*len(df_out),
        mode="lines",
        name="Threshold",
        line=dict(dash="dash", color="orange"),
        hovertemplate="Threshold: %{y:.8f}"
    ))
    fig.update_layout(
        title="Kadar NO₂ per Hari",
        xaxis_title="Tanggal",
        yaxis_title="NO₂ (mol/m²)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # TABEL + DOWNLOAD
    st.markdown("### Data Lengkap")
    st.dataframe(df_out.style.format({
        "NO2(mol/m²)": "{:.8f}",
        "NO2(µg/m³)": "{:.4f}",
    }), use_container_width=True)

    csv_out = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Unduh hasil (CSV)", csv_out, "no2_multi_forecast.csv", "text/csv")
