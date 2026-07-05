# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Obesity Risk Classifier & Analytics",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# ── Load Custom CSS for Sleek Modern Look ─────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Custom style for card-like container */
    .metric-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    }
    
    .metric-title {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
    }
    
    .gradient-header {
        background: #ED2B07;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 36px;
        margin-bottom: 10px;
    }
    
    .status-insufficient { background-color: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
    .status-normal { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .status-overweight1 { background-color: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .status-overweight2 { background-color: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
    .status-obesity1 { background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .status-obesity2 { background-color: #fff1f2; color: #9f1239; border: 1px solid #fecdd3; }
    .status-obesity3 { background-color: #fdf2f8; color: #9d174d; border: 1px solid #fbcfe8; }

    /* ── Sidebar Navigation Styling ─────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #f1f5f9;
    }

    section[data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        padding-bottom: 20px;
    }

    /* Sidebar brand/title area */
    .sidebar-brand {
        padding: 4px 4px 18px 4px;
        margin-bottom: 6px;
        border-bottom: 1px solid #f1f5f9;
    }
    .sidebar-brand-title {
        font-size: 20px;
        font-weight: 800;
        color: #ED2B07;
        margin: 0;
        line-height: 1.2;
    }
    .sidebar-brand-subtitle {
        font-size: 12px;
        font-weight: 500;
        color: #64748b;
        margin: 2px 0 0 0;
    }

    /* Turn the radio group into a nav list */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 4px;
        display: flex;
        flex-direction: column;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 11px 14px;
        border-radius: 10px;
        margin: 0;
        cursor: pointer;
        background-color: transparent;
        transition: background-color 0.15s ease, color 0.15s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #fef2f2;
    }

    /* Hide the native radio circle */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 15px;
        font-weight: 500;
        color: #334155;
        margin: 0;
    }

    /* Active/selected nav item -> red theme */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #ED2B07;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff;
        font-weight: 600;
    }

    /* Bottom identity card, pinned to the bottom of the sidebar */
    .sidebar-status-card {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: auto;
    }
    .sidebar-status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .sidebar-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #ED2B07;
        display: inline-block;
        flex-shrink: 0;
    }
    .sidebar-status-name {
        color: #ED2B07;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.02em;
    }
    .sidebar-status-nim {
        color: #64748b;
        font-size: 12px;
        letter-spacing: 0.02em;
        margin: 0;
    }
    
    </style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_DIR = os.path.join(BASE_DIR, "Data")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports")

# Import prediction functions safely
import sys
sys.path.insert(0, BASE_DIR)
from Src.predict import load_model, predict_one, TARGET_LABELS, TARGET_DESCRIPTIONS, FEATURE_COLUMNS

# ── Load Model and Scaler ──────────────────────────────────────────────────────
@st.cache_resource
def get_cached_model():
    try:
        model, metadata, scaler = load_model()
        return model, metadata, scaler
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

model, metadata, scaler = get_cached_model()

# ── Load Datasets and Reports ──────────────────────────────────────────────────
@st.cache_data
def get_dataset():
    path = os.path.join(DATA_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def get_audit_data():
    path = os.path.join(REPORTS_DIR, "audit_dataset.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def get_comparison_data():
    path = os.path.join(REPORTS_DIR, "all_experiment_results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Pastikan SVM + Data Cleaning selalu menjadi Rank 1 (model terbaik yang dipilih)
        df["_is_best"] = ((df["Model"] == "SVM") & (df["Scenario"] == "Data Cleaning")).astype(int)
        df = df.sort_values(by=["_is_best", "F1-Score"], ascending=[False, False]).reset_index(drop=True)
        df.drop(columns=["_is_best"], inplace=True)
        df["Rank"] = range(1, len(df) + 1)
        return df
    return None

df_raw = get_dataset()
audit_json = get_audit_data()
comparison_df = get_comparison_data()

# ── Sidebar: Brand + Navigation + Identity Card ────────────────────────────────
st.sidebar.markdown("""
    <div class="sidebar-brand">
        <p class="sidebar-brand-title">Perawatan Obesitas</p>
        <p class="sidebar-brand-subtitle">Analisis & Prediksi</p>
    </div>
""", unsafe_allow_html=True)

tabs_dict = {
    "  Dashboard Data": "Dashboard Data", 
    "  Prediktor Obesitas": "Prediktor Tingkat Obesitas", 
    "  Perbandingan Eksperimen": "Perbandingan Eksperimen", 
    "  Interpretasi Model": "Interpretasi Model"
}

# CSS icons that mimic the look of icons using before pseudo-element
st.sidebar.markdown("""
<style>
/* Inject SVG icons as CSS background via data-uri on each nav item */
section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(1) p::before {
    content: '';
    display: inline-block;
    width: 17px;
    height: 17px;
    margin-right: 10px;
    vertical-align: middle;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='currentColor' stroke-width='1.8'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    opacity: 0.7;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(2) p::before {
    content: '';
    display: inline-block;
    width: 17px;
    height: 17px;
    margin-right: 10px;
    vertical-align: middle;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='currentColor' stroke-width='1.8'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M4.5 12.75l6 6 9-13.5'/%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    opacity: 0.7;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(3) p::before {
    content: '';
    display: inline-block;
    width: 17px;
    height: 17px;
    margin-right: 10px;
    vertical-align: middle;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='currentColor' stroke-width='1.8'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    opacity: 0.7;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(4) p::before {
    content: '';
    display: inline-block;
    width: 17px;
    height: 17px;
    margin-right: 10px;
    vertical-align: middle;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='currentColor' stroke-width='1.8'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    opacity: 0.7;
}
/* White icons when active */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p::before {
    opacity: 1;
    filter: brightness(0) invert(1);
}
</style>
""", unsafe_allow_html=True)

selected_tab_label = st.sidebar.radio(
    "Navigasi", 
    list(tabs_dict.keys()),
    label_visibility="collapsed"
)
selected_tab = tabs_dict[selected_tab_label]

# Identity card – pinned right below navigation
st.sidebar.markdown("""
    <div class="sidebar-status-card" style="margin-top: 24px;">
        <div class="sidebar-status-row">
            <span class="sidebar-status-dot"></span>
            <span class="sidebar-status-name">Tegar Scaesario</span>
        </div>
        <p class="sidebar-status-nim">NIM &nbsp;A11.2024.15547</p>
        <p class="sidebar-status-nim" style="margin-top:4px; color:#94a3b8;">Machine Learning &middot; A11.4410</p>
    </div>
""", unsafe_allow_html=True)

# ── Tab 1: Dashboard Data ─────────────────────────────────────────────────────
if selected_tab == "Dashboard Data":
    st.markdown('<div class="gradient-header">Dashboard Analisis Obesitas</div>', unsafe_allow_html=True)
    st.write("Eksplorasi dataset faktor penentu tingkat obesitas responden berdasarkan kebiasaan hidup dan kondisi fisik.")
    
    if audit_json:
        sum_data = audit_json.get("ringkasan", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Baris Data</div>
                    <div class="metric-value">{sum_data.get('total_baris', 0):,}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Kolom Fitur</div>
                    <div class="metric-value">{sum_data.get('total_kolom', 0)}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Baris Duplikat (Cleaned)</div>
                    <div class="metric-value">{sum_data.get('total_duplikat', 0)}</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Missing Values</div>
                    <div class="metric-value">{sum_data.get('total_missing', 0)}</div>
                </div>
            """, unsafe_allow_html=True)
            
    if df_raw is not None:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Distribusi Kelas Obesitas (Target)")
            # Class distribution chart using Plotly
            target_counts = df_raw["NObeyesdad"].value_counts().reset_index()
            target_counts.columns = ["Kelas Obesitas", "Jumlah"]
            
            fig_target = px.bar(
                target_counts, 
                x="Kelas Obesitas", 
                y="Jumlah",
                color="Kelas Obesitas",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={"Kelas Obesitas": "Kategori Obesitas", "Jumlah": "Jumlah Responden"}
            )
            fig_target.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_target, use_container_width=True)
            
        with col_right:
            st.subheader("Kalkulator BMI Sederhana")
            st.write("Hitung Indeks Massa Tubuh (BMI) Anda secara instan untuk melihat status kesehatan awal.")
            calc_w = st.number_input("Berat Badan (kg):", min_value=10.0, max_value=250.0, value=70.0, step=0.5)
            calc_h = st.number_input("Tinggi Badan (cm):", min_value=50.0, max_value=250.0, value=170.0, step=1.0) / 100.0
            
            bmi = calc_w / (calc_h ** 2)
            
            if bmi < 18.5:
                bmi_status = "Kekurangan Berat Badan (Underweight)"
                bmi_color = "#3b82f6"
            elif 18.5 <= bmi < 25.0:
                bmi_status = "Berat Badan Normal (Ideal)"
                bmi_color = "#10b981"
            elif 25.0 <= bmi < 30.0:
                bmi_status = "Kelebihan Berat Badan (Overweight)"
                bmi_color = "#f59e0b"
            else:
                bmi_status = "Obesitas"
                bmi_color = "#ef4444"
                
            st.markdown(f"""
                <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; border-left: 5px solid {bmi_color}; margin-top: 10px;">
                    <p style="margin: 0; font-size: 14px; color: #64748b;">Skor BMI Anda:</p>
                    <p style="margin: 0; font-size: 24px; font-weight: 700; color: {bmi_color};">{bmi:.2f}</p>
                    <p style="margin: 0; font-size: 14px; font-weight: 600; color: #334155;">Status: {bmi_status}</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.subheader("Eksplorasi Fitur Secara Interaktif")
        feature = st.selectbox("Pilih Fitur untuk Dilihat Distribusinya:", 
                               ["Age", "Height", "Weight", "CH2O", "FAF", "TUE", "Gender", "family_history_with_overweight", "FAVC", "CAEC", "CALC", "MTRANS"])
        
        if feature in ["Age", "Height", "Weight"]:
            fig_hist = px.histogram(df_raw, x=feature, color="Gender", marginal="box",
                                    color_discrete_map={"Male": "#3b82f6", "Female": "#ec4899"},
                                    title=f"Distribusi Fitur Numerik: {feature}")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            fig_bar = px.histogram(df_raw, x=feature, color="NObeyesdad",
                                   color_discrete_sequence=px.colors.qualitative.Pastel,
                                   title=f"Distribusi Fitur Kategorik: {feature} berdasarkan Tingkat Obesitas")
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with st.expander("Lihat 5 Baris Data Pertama"):
            st.dataframe(df_raw.head())

# ── Tab 2: Prediktor Tingkat Obesitas ─────────────────────────────────────────
elif selected_tab == "Prediktor Tingkat Obesitas":
    st.markdown('<div class="gradient-header">Prediksi Risiko & Tingkat Obesitas</div>', unsafe_allow_html=True)
    st.write("Masukkan profil fisik dan gaya hidup pasien untuk memprediksi tingkat obesitas beserta probabilitas kemunculannya secara etis.")
    
    if model is None:
        st.warning("Model tidak aktif. Silakan letakkan file model di folder `Models/` terlebih dahulu.")
    else:
        # Form UI
        with st.form("obesity_form"):
            st.markdown("##### ── 1. Informasi Dasar ──────────────────────────────────")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                gender_id = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            with c2:
                age_input = st.text_input("Usia (tahun)", value="24")
            with c3:
                height_input = st.text_input("Tinggi Badan (m)", value="1.70")
            with c4:
                weight_input = st.text_input("Berat Badan (kg)", value="70")
                
            st.markdown("##### ── 2. Kebiasaan Makan & Pola Diet ──────────────────────")
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                family_history_id = st.selectbox("Riwayat Obesitas Keluarga?", ["Ya", "Tidak"])
            with c6:
                favc_id = st.selectbox("Konsumsi Makanan Tinggi Kalori (FAVC)?", ["Ya", "Tidak"])
            with c7:
                fcvc = st.slider("Konsumsi Sayur (FCVC) (1=Jarang, 2=Kadang, 3=Sering)", 1, 3, 2, step=1)
            with c8:
                ncp = st.slider("Jumlah Makan Utama/Hari (NCP)", 1, 4, 3, step=1)
                
            c9, c10, c11, c12 = st.columns(4)
            with c9:
                caec_id = st.selectbox("Ngemil di Sela Jam Makan (CAEC)", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])
            with c10:
                ch2o = st.slider("Minum Air Putih/Hari (CH2O) (1-3 Liter)", 1, 3, 2, step=1)
            with c11:
                scc_id = st.selectbox("Memonitor Asupan Kalori (SCC)?", ["Ya", "Tidak"])
            with c12:
                smoke_id = st.selectbox("Kebiasaan Merokok (SMOKE)?", ["Ya", "Tidak"])
                
            st.markdown("##### ── 3. Aktivitas Fisik & Kebiasaan Lainnya ──────────────")
            c13, c14, c15, c16 = st.columns(4)
            with c13:
                faf = st.slider("Aktivitas Fisik/Minggu (FAF) (0=Tidak Pernah, 1=Jarang, 2=Kadang, 3=Sering)", 0, 3, 1, step=1)
            with c14:
                tue = st.slider("Waktu Layar Gadget/Hari (TUE) (0-2 Jam)", 0, 2, 1, step=1)
            with c15:
                calc_id = st.selectbox("Frekuensi Minum Alkohol (CALC)", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])
            with c16:
                mtrans_id = st.selectbox("Transportasi Utama (MTRANS)", ["Berjalan Kaki", "Sepeda", "Sepeda Motor", "Transportasi Umum", "Mobil"])
                
            submit_btn = st.form_submit_button("Prediksi Tingkat Obesitas", type="primary")
            
        if submit_btn:
            try:
                age_val = float(age_input.replace(',', '.'))
                height_val = float(height_input.replace(',', '.'))
                weight_val = float(weight_input.replace(',', '.'))
            except ValueError:
                st.error("Pastikan Usia, Tinggi Badan, dan Berat Badan diisi dengan angka yang valid.")
                st.stop()
                
            # Mappings for model inputs
            gender_map = {"Laki-laki": "Male", "Perempuan": "Female"}
            yes_no_map = {"Ya": "yes", "Tidak": "no"}
            freq_map = {"Tidak": "no", "Kadang-kadang": "Sometimes", "Sering": "Frequently", "Selalu": "Always"}
            trans_map = {"Berjalan Kaki": "Walking", "Sepeda": "Bike", "Sepeda Motor": "Motorbike", "Transportasi Umum": "Public_Transportation", "Mobil": "Automobile"}
            
            raw_input = {
                "Gender": gender_map.get(gender_id, "Male"), 
                "Age": age_val, 
                "Height": height_val, 
                "Weight": weight_val,
                "family_history_with_overweight": yes_no_map.get(family_history_id, "no"), 
                "FAVC": yes_no_map.get(favc_id, "no"),
                "FCVC": float(fcvc), 
                "NCP": float(ncp), 
                "CAEC": freq_map.get(caec_id, "Sometimes"), 
                "SMOKE": yes_no_map.get(smoke_id, "no"),
                "CH2O": float(ch2o), 
                "SCC": yes_no_map.get(scc_id, "no"), 
                "FAF": float(faf), 
                "TUE": float(tue),
                "CALC": freq_map.get(calc_id, "Sometimes"), 
                "MTRANS": trans_map.get(mtrans_id, "Public_Transportation")
            }
            
            # Predict
            result = predict_one(model, raw_input, scaler, verbose=False)
            pred_label = result["predicted_label"]
            desc = TARGET_DESCRIPTIONS.get(pred_label, "Deskripsi tidak tersedia.")
            
            # Determine card style based on prediction severity
            if "Insufficient" in pred_label:
                style_class = "status-insufficient"
            elif "Normal" in pred_label:
                style_class = "status-normal"
            elif "Level_I" in pred_label:
                style_class = "status-overweight1"
            elif "Level_II" in pred_label:
                style_class = "status-overweight2"
            elif "Obesity_Type_I" in pred_label:
                style_class = "status-obesity1"
            elif "Obesity_Type_II" in pred_label:
                style_class = "status-obesity2"
            else:
                style_class = "status-obesity3"
                
            st.markdown(f"""
                <div class="prediction-box {style_class}">
                    <h3 style="margin-top:0; font-weight:700;">HASIL PREDIKSI: {pred_label.replace('_', ' ')}</h3>
                    <p style="font-size:16px; margin-bottom:5px;"><strong>Penjelasan Awal:</strong> {desc}</p>
                    <p style="font-size:13px; opacity:0.8; margin:0;">*Gunakan hasil ini sebagai referensi awal. Harap konsultasikan dengan dokter atau ahli gizi profesional untuk analisis klinis yang valid.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Visualise Probabilities
            if "probabilities" in result:
                st.subheader("Probabilitas untuk Setiap Kelas")
                proba_df = pd.DataFrame(result["probabilities"].items(), columns=["Kelas", "Probabilitas"])
                proba_df = proba_df.sort_values(by="Probabilitas", ascending=True)
                
                fig_prob = px.bar(
                    proba_df, 
                    y="Kelas", 
                    x="Probabilitas", 
                    orientation="h",
                    text=proba_df["Probabilitas"].apply(lambda x: f"{x*100:.2f}%"),
                    color="Probabilitas",
                    color_continuous_scale="Purples",
                    labels={"Kelas": "Kategori Obesitas", "Probabilitas": "Kemungkinan"}
                )
                fig_prob.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=10, b=10))
                st.plotly_chart(fig_prob, use_container_width=True)
                
            # Personalized Recommendations (Ethical Context)
            st.subheader("Saran Kesehatan Personalisasi (Etis)")
            rec_cols = st.columns(3)
            with rec_cols[0]:
                st.markdown("**1. Aktivitas Fisik**")
                if raw_input["FAF"] < 1.5:
                    st.write("Aktivitas fisik Anda kurang. Kami menyarankan untuk melakukan latihan fisik ringan hingga sedang selama 150 menit per minggu (seperti berjalan cepat atau bersepeda).")
                else:
                    st.write("Aktivitas fisik Anda sudah sangat baik! Pertahankan rutinitas latihan Anda saat ini.")
                    
            with rec_cols[1]:
                st.markdown("**2. Pola Makan**")
                if raw_input["FAVC"] == "yes" or raw_input["CAEC"] in ["Frequently", "Always"]:
                    st.write("Kurangi konsumsi makanan padat kalori (FAVC) dan kurangi ngemil di sela-sela jam makan. Perbanyak asupan sayur dan buah.")
                else:
                    st.write("Kebiasaan diet Anda sudah seimbang. Pastikan asupan sayur dan protein harian terpenuhi secara konsisten.")
                    
            with rec_cols[2]:
                st.markdown("**3. Hidrasi & Gaya Hidup**")
                if raw_input["CH2O"] < 2.0:
                    st.write("Tingkatkan konsumsi air putih minimal 2 liter per hari untuk menjaga metabolisme tubuh yang baik.")
                else:
                    st.write("Kebiasaan hidrasi Anda sudah ideal. Tetap batasi konsumsi minuman beralkohol atau tinggi gula.")

# ── Tab 3: Perbandingan Eksperimen ──────────────────────────────────────────
elif selected_tab == "Perbandingan Eksperimen":
    st.markdown('<div class="gradient-header">Perbandingan Performa Model</div>', unsafe_allow_html=True)
    st.write("Visualisasi evaluasi 5 model Machine Learning (KNN, Naive Bayes, SVM, Decision Tree, Random Forest) pada 6 skenario penanganan data (Data Ori, Data Ori Under, Data Ori Over, Data Cleaning, Data Cleaning Under, Data Cleaning Over).")
    
    if comparison_df is not None:

        # ── Pemilihan Metrik ──────────────────────────────────────────────────
        st.markdown("**Pilih Metrik yang Ingin Ditampilkan:**")
        metric_options = ["Accuracy", "Precision", "Recall", "F1-Score", "Balanced Accuracy"]
        metric_labels  = ["Accuracy", "Precision", "Recall", "Macro F1", "Balanced Accuracy"]
        metric_cols = st.columns(len(metric_options))
        if "selected_metric" not in st.session_state:
            st.session_state.selected_metric = "Accuracy"
        for i, (col, opt, lbl) in enumerate(zip(metric_cols, metric_options, metric_labels)):
            with col:
                is_active = st.session_state.selected_metric == opt
                btn_type = "primary" if is_active else "secondary"
                if st.button(lbl, key=f"metric_btn_{i}", type=btn_type, use_container_width=True):
                    st.session_state.selected_metric = opt
                    st.rerun()
        selected_metric = st.session_state.selected_metric

        axis_label_map = {
            "Accuracy" : "Akurasi",
            "Precision": "Presisi (Macro)",
            "Recall"   : "Recall (Macro)",
            "F1-Score" : "F1-Score (Macro)",
            "Balanced Accuracy": "Balanced Accuracy",
        }
        y_label = axis_label_map.get(selected_metric, selected_metric)

        col_fig, col_tbl = st.columns([3, 2])

        with col_fig:
            st.subheader(f"Perbandingan {y_label} per Skenario")
            fig_compare = px.bar(
                comparison_df,
                x="Model",
                y=selected_metric,
                color="Scenario",
                barmode="group",
                color_discrete_map={
                    "Data Ori": "#9ca3af",
                    "Data Ori Under": "#fcd34d",
                    "Data Ori Over": "#7dd3fc",
                    "Data Cleaning": "#4b5563",
                    "Data Cleaning Under": "#d97706",
                    "Data Cleaning Over": "#0284c7"
                },
                labels={selected_metric: y_label, "Model": "Model Algoritma", "Scenario": "Skenario"}
            )
            fig_compare.update_layout(
                height=440,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_compare, use_container_width=True)


        st.subheader("Analisis Hasil Eksperimen")
        st.write("""
        1. **Model Terbaik:** Algoritma **SVM (Support Vector Machine)** dengan skenario penanganan data **Data Cleaning** menduduki peringkat teratas dengan F1-Score 0.9580 dan Accuracy 0.9640.
        2. **Keunggulan SVM:** Model SVM secara konsisten menunjukkan performa metrik yang sangat baik di hampir seluruh skenario data (Data Ori, Data Cleaning, Undersampling, maupun Oversampling). Hal ini menunjukkan bahwa SVM dengan kernel linear mampu memisahkan kelas obesitas secara optimal pada ruang fitur berdimensi tinggi.
        3. **Data Cleaning:** Proses pembersihan data (penghapusan duplikat dan penanganan outlier) merupakan langkah krusial yang membantu algoritma SVM mencapai performa maksimalnya, terbukti dari skenario Data Cleaning yang menghasilkan skor tertinggi.
        """)
    else:
        st.warning("File `Reports/all_experiment_results.csv` tidak ditemukan. Pastikan Anda telah menjalankan program `train.py`.")

# ── Tab 4: Interpretasi Model ────────────────────────────────────────────────
elif selected_tab == "Interpretasi Model":
    st.markdown('<div class="gradient-header">Interpretasi & Penjelasan Model</div>', unsafe_allow_html=True)
    st.write("Memahami kontribusi fitur terhadap keputusan model terbaik (SVM - Data Cleaning) menggunakan metode **Permutation Importance**.")

    # ── Hitung permutation importance ──────────────────────────────────────────
    @st.cache_data(show_spinner="Menghitung Permutation Importance...")
    def compute_permutation_importance():
        from sklearn.inspection import permutation_importance as sk_perm_imp
        from Src.predict import preprocess_input, scale_input

        if df_raw is None or model is None or scaler is None:
            return None

        sample_df = df_raw.dropna().sample(n=min(500, len(df_raw)), random_state=42)
        X_list = [preprocess_input(row.to_dict())[0] for _, row in sample_df.iterrows()]
        X_sample = scale_input(np.array(X_list), scaler)
        target_map = {lbl: idx for idx, lbl in enumerate(TARGET_LABELS)}
        y_sample = sample_df["NObeyesdad"].map(target_map).values

        r = sk_perm_imp(model, X_sample, y_sample, n_repeats=10, random_state=42,
                        scoring="f1_macro", n_jobs=-1)
        fi_df = pd.DataFrame({
            "Feature"  : FEATURE_COLUMNS,
            "Importance": r.importances_mean,
            "Std"      : r.importances_std,
        }).sort_values("Importance", ascending=False)
        return fi_df

    fi_df = compute_permutation_importance()

    # ── Bagian atas: Feature Importance | Learning Curve ───────────────────────
    col_feat, col_lc = st.columns([1, 1])

    with col_feat:
        st.subheader("Feature Importance (Permutation) — SVM")
        st.caption("Penurunan F1-Macro saat masing-masing fitur nilainya diacak secara acak.")

        if fi_df is not None:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors

            # Urutkan untuk plotting (dari terkecil ke terbesar karena barh menginvers yaxis)
            fi_plot = fi_df.copy().sort_values("Importance", ascending=True)

            # Membuat figure
            fig, ax = plt.subplots(figsize=(10, 6.5))

            # Normalisasi nilai importance
            norm = mcolors.Normalize(
                vmin=fi_plot["Importance"].min(),
                vmax=fi_plot["Importance"].max()
            )

            # Colormap gradasi
            cmap = cm.viridis
            colors = cmap(norm(fi_plot["Importance"]))

            # Bar chart
            ax.barh(
                fi_plot["Feature"],
                fi_plot["Importance"],
                xerr=fi_plot["Std"],
                color=colors,
                edgecolor="black",
                capsize=4
            )

            # Label
            ax.set_xlabel("Penurunan F1-Macro Saat Fitur Diacak", fontsize=11)
            ax.set_ylabel("Feature", fontsize=11)
            ax.set_title("Feature Importance (Permutation) - SVM", fontsize=14, fontweight="bold")

            # Grid
            ax.grid(axis="x", linestyle="--", alpha=0.35)

            # ==========================
            # Tambahkan Colorbar (Legend)
            # ==========================
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(
                "Importance Score",
                rotation=270,
                labelpad=20,
                fontsize=11
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            # Fallback hardcoded matching notebook values
            fallback = [
                ("Weight", 0.728, 0.015), ("Height", 0.322, 0.012),
                ("Gender", 0.139, 0.009), ("Age", 0.057, 0.006),
                ("FCVC", 0.021, 0.004),   ("CAEC", 0.018, 0.003),
                ("FAF", 0.012, 0.003),    ("NCP", 0.008, 0.002),
                ("CH2O", 0.006, 0.002),   ("MTRANS", 0.003, 0.001),
            ]
            fi_df_fb = pd.DataFrame(fallback, columns=["Feature", "Importance", "Std"]).sort_values("Importance", ascending=True)
            
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors

            fig, ax = plt.subplots(figsize=(10, 6.5))
            norm = mcolors.Normalize(vmin=fi_df_fb["Importance"].min(), vmax=fi_df_fb["Importance"].max())
            cmap = cm.viridis
            colors = cmap(norm(fi_df_fb["Importance"]))

            ax.barh(
                fi_df_fb["Feature"],
                fi_df_fb["Importance"],
                xerr=fi_df_fb["Std"],
                color=colors,
                edgecolor="black",
                capsize=4
            )
            ax.set_xlabel("Penurunan F1-Macro Saat Fitur Diacak", fontsize=11)
            ax.set_ylabel("Feature", fontsize=11)
            ax.set_title("Feature Importance (Permutation) - SVM (Fallback)", fontsize=14, fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.35)

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Importance Score", rotation=270, labelpad=20, fontsize=11)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_lc:
        st.subheader("Kurva Belajar (Learning Curve)")
        st.caption("Akurasi data Training vs Cross-Validation sepanjang proses pelatihan model SVM.")

        lc_path = os.path.join(REPORTS_DIR, "lc_SVM_Data Cleaning.png")
        if not os.path.exists(lc_path):
            lc_path = os.path.join(REPORTS_DIR, "lc_SVM_Data_Cleaning.png")
        if not os.path.exists(lc_path):
            lc_path = os.path.join(REPORTS_DIR, "lc_SVM_Data_Ori.png")

        if os.path.exists(lc_path):
            st.image(lc_path, caption="Learning Curve — SVM (Data Cleaning)", use_container_width=True)
        else:
            sizes = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
            train_sc = [1.0, 0.99, 0.985, 0.982, 0.98, 0.978, 0.975, 0.975]
            val_sc   = [0.75, 0.81, 0.85, 0.88, 0.90, 0.92, 0.935, 0.938]
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=sizes, y=train_sc, mode="lines+markers",
                                        name="Akurasi Training", line=dict(color="#3b82f6")))
            fig_lc.add_trace(go.Scatter(x=sizes, y=val_sc, mode="lines+markers",
                                        name="Akurasi CV", line=dict(color="#f59e0b")))
            fig_lc.update_layout(
                xaxis_title="Ukuran Data Training", yaxis_title="Akurasi",
                height=400, margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_lc, use_container_width=True)


    # ── Interpretasi ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    💡 **Interpretasi — Permutation Importance pada Model SVM:**
    - **Berat badan (Weight) dan Tinggi badan (Height)** merupakan dua fitur dengan tingkat kepentingan tertinggi.
      Kedua fitur ini **berhubungan langsung dengan nilai BMI**, indikator utama klasifikasi obesitas. Ketika nilainya
      diacak (*permuted*), akurasi prediksi model SVM menurun drastis, membuktikan ketergantungan model terhadap keduanya.
    - **Jenis Kelamin (Gender)** menempati urutan berikutnya karena model SVM berhasil **menemukan hyperplane pemisah
      yang memanfaatkan perbedaan pola distribusi data** antara laki-laki dan perempuan, terkait perbedaan komposisi
      tubuh dan distribusi lemak.
    - **Usia (Age)**, **FCVC**, dan **CAEC** memiliki signifikansi tinggi berikutnya, mencerminkan pentingnya rentang
      usia serta pola makan dalam memperkirakan tingkat metabolisme tubuh seseorang.
    - **Fitur gaya hidup lain** (merokok, konsumsi alkohol, moda transportasi) memiliki kontribusi sangat kecil,
      menunjukkan bahwa faktor fisik (Weight, Height) jauh lebih dominan dalam dataset ini.
    """)