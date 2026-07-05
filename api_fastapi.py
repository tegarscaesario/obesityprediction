from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import sys

# ── Paths & Imports ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports")
sys.path.insert(0, BASE_DIR)

from Src.predict import load_model, predict_one, TARGET_LABELS, TARGET_DESCRIPTIONS

# Initialize FastAPI app
app = FastAPI(
    title="Obesity Risk Prediction API",
    description="Backend API untuk mendeteksi tingkat obesitas responden berdasarkan profil fisik dan kebiasaan hidup harian.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model and Scaler ──────────────────────────────────────────────────────
try:
    model, metadata, scaler = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model, metadata, scaler = None, None, None

# ── Pydantic Request Models ───────────────────────────────────────────────────
class ObesityInput(BaseModel):
    Gender: str = Field(..., description="Jenis Kelamin ('Male' atau 'Female')", json_schema_extra={"example": "Male"})
    Age: float = Field(..., description="Usia responden (tahun)", json_schema_extra={"example": 24.0})
    Height: float = Field(..., description="Tinggi badan responden (meter)", json_schema_extra={"example": 1.70})
    Weight: float = Field(..., description="Berat badan responden (kg)", json_schema_extra={"example": 72.0})
    family_history_with_overweight: str = Field(..., description="Riwayat keluarga obesitas ('yes' atau 'no')", json_schema_extra={"example": "no"})
    FAVC: str = Field(..., description="Konsumsi makanan kalori tinggi ('yes' atau 'no')", json_schema_extra={"example": "no"})
    FCVC: float = Field(..., description="Frekuensi makan sayur (1 = Jarang, 3 = Selalu)", json_schema_extra={"example": 2.0})
    NCP: float = Field(..., description="Jumlah makan utama per hari", json_schema_extra={"example": 3.0})
    CAEC: str = Field(..., description="Konsumsi cemilan ('no', 'Sometimes', 'Frequently', 'Always')", json_schema_extra={"example": "Sometimes"})
    SMOKE: str = Field(..., description="Kebiasaan merokok ('yes' atau 'no')", json_schema_extra={"example": "no"})
    CH2O: float = Field(..., description="Konsumsi air putih harian (liter)", json_schema_extra={"example": 2.0})
    SCC: str = Field(..., description="Memonitor asupan kalori ('yes' atau 'no')", json_schema_extra={"example": "no"})
    FAF: float = Field(..., description="Aktivitas fisik mingguan (0 = Jarang, 3 = Sering)", json_schema_extra={"example": 1.0})
    TUE: float = Field(..., description="Lama penggunaan gadget harian (jam)", json_schema_extra={"example": 1.0})
    CALC: str = Field(..., description="Konsumsi alkohol ('no', 'Sometimes', 'Frequently', 'Always')", json_schema_extra={"example": "no"})
    MTRANS: str = Field(..., description="Moda transportasi ('Walking', 'Bike', 'Motorbike', 'Public_Transportation', 'Automobile')", json_schema_extra={"example": "Public_Transportation"})

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    """Mengembalikan metadata mahasiswa, status model, dan konfigurasi API."""
    return {
        "status": "online",
        "project": "Klasifikasi Tingkat Obesitas dengan Machine Learning",
        "mahasiswa": {
            "nama": "Tegar Scaesario",
            "nim": "A11.2024.15547",
            "mata_kuliah": "Machine Learning – A11.4410",
            "institusi": "Universitas Dian Nuswantoro"
        },
        "model_status": {
            "loaded": model is not None,
            "model_name": metadata.get("model_name", "Random Forest") if metadata else "N/A",
            "scenario": metadata.get("scenario", "SMOTE") if metadata else "N/A",
            "accuracy": metadata.get("accuracy", 0.9568) if metadata else 0.0,
            "f1_score": metadata.get("f1_score", 0.9568) if metadata else 0.0,
        }
    }

@app.post("/predict")
def predict_obesity(payload: ObesityInput):
    """Menerima data pasien dan menghasilkan prediksi tingkat obesitas beserta probabilitas per kelas."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model Machine Learning tidak aktif di backend server.")
        
    try:
        # Convert Pydantic object to dictionary
        raw_dict = payload.model_dump()
        
        # Predict using utility function
        result = predict_one(model, raw_dict, scaler, verbose=False)
        pred_label = result["predicted_label"]
        desc = TARGET_DESCRIPTIONS.get(pred_label, "Deskripsi tidak tersedia.")
        
        # Calculate BMI
        bmi = payload.Weight / (payload.Height ** 2)
        
        return {
            "status": "success",
            "bmi": round(bmi, 2),
            "prediction": {
                "label": pred_label,
                "display_name": pred_label.replace("_", " "),
                "description": desc
            },
            "probabilities": result.get("probabilities", {}),
            "recommendations": {
                "physical_activity": (
                    "Tingkatkan olahraga mingguan minimal 150 menit per minggu." 
                    if payload.FAF < 1.5 else "Tingkat aktivitas fisik sudah sangat baik. Lanjutkan rutin!"
                ),
                "nutrition": (
                    "Batasi konsumsi makanan padat kalori (FAVC) dan ngemil manis." 
                    if payload.FAVC == "yes" or payload.CAEC in ["Frequently", "Always"] 
                    else "Kebiasaan diet Anda sudah seimbang."
                ),
                "hydration": (
                    "Tingkatkan konsumsi air putih minimal 2 liter per hari." 
                    if payload.CH2O < 2.0 else "Kebiasaan hidrasi tubuh sudah ideal."
                )
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat inferensi model: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Mengembalikan rekap perbandingan hasil seluruh eksperimen (model x skenario)."""
    path_csv = os.path.join(REPORTS_DIR, "all_experiment_results.csv")
    if not os.path.exists(path_csv):
        raise HTTPException(status_code=404, detail="File comparison metrics belum tersedia. Harap jalankan training pipeline.")
        
    try:
        df = pd.read_csv(path_csv)
        return {
            "status": "success",
            "total_experiments": len(df),
            "experiments": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membaca metrik: {str(e)}")
