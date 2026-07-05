import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_DIR   = os.path.join(BASE_DIR, "Data")
sys.path.insert(0, BASE_DIR)

# ── Mapping ───────────────────────────────────────────────────────────────────
BINARY_MAP = {"yes": 1, "no": 0, "Yes": 1, "No": 0, "1": 1, "0": 0}
GENDER_MAP = {"Female": 0, "Male": 1, "female": 0, "male": 1, "F": 0, "M": 1}
CAEC_MAP   = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
CALC_MAP   = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
MTRANS_MAP = {
    "Walking": 0, "Bike": 1, "Public_Transportation": 2,
    "Motorbike": 3, "Automobile": 4,
}

TARGET_LABELS = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

TARGET_DESCRIPTIONS = {
    "Insufficient_Weight" : "  Berat Badan Kurang – Perlu peningkatan asupan nutrisi.",
    "Normal_Weight"       : "  Berat Badan Normal – Pertahankan pola hidup sehat!",
    "Overweight_Level_I"  : "  Kelebihan Berat Badan Tk. I – Perhatikan pola makan.",
    "Overweight_Level_II" : "  Kelebihan Berat Badan Tk. II – Konsultasikan dengan dokter.",
    "Obesity_Type_I"      : "  Obesitas Tipe I – Segera ubah gaya hidup.",
    "Obesity_Type_II"     : "  Obesitas Tipe II – Diperlukan intervensi medis.",
    "Obesity_Type_III"    : "  Obesitas Tipe III – Kondisi serius, konsultasi dokter segera.",
}

# Urutan kolom fitur (sesuai encoding)
FEATURE_COLUMNS = [
    "Gender", "Age", "Height", "Weight",
    "family_history_with_overweight", "FAVC",
    "FCVC", "NCP", "CAEC", "SMOKE", "CH2O",
    "SCC", "FAF", "TUE", "CALC", "MTRANS",
]


# 1. LOAD MODEL

def load_model(model_path: str = None) -> tuple:

    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "model_terbaik.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}\n"
            "Jalankan terlebih dahulu: python Src/train.py"
        )

    payload = joblib.load(model_path)

    # Jika payload berupa dict (model_terbaik.joblib)
    if isinstance(payload, dict) and "model" in payload:
        model    = payload["model"]
        metadata = payload.get("metadata", {})
    else:
        model    = payload
        metadata = {}

    # Muat scaler (wajib — model dilatih di atas data yang sudah di-scale)
    scaler_path = os.path.join(MODELS_DIR, "scaler_minmax.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler tidak ditemukan: {scaler_path}\n"
            "Scaler wajib ada karena model dilatih di atas data yang sudah dinormalisasi."
        )
    scaler = joblib.load(scaler_path)

    print(f"  ✔ Model dimuat dari: {model_path}")
    print(f"  ✔ Scaler dimuat dari: {scaler_path}")
    if metadata:
        print(f"  • Nama model  : {metadata.get('model_name', 'N/A')}")
        print(f"  • Skenario    : {metadata.get('scenario', 'N/A')}")
        print(f"  • Accuracy    : {metadata.get('accuracy', 'N/A'):.4f}")
        print(f"  • F1-Score    : {metadata.get('f1_score', 'N/A'):.4f}")

    return model, metadata, scaler


# 2. PREPROCESSING INPUT

def preprocess_input(raw: dict) -> np.ndarray:
    processed = {}

    # Gender
    processed["Gender"] = GENDER_MAP.get(str(raw.get("Gender", "Male")), 0)

    # Numeric
    for col in ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        processed[col] = float(raw.get(col, 0))

    # Binary
    for col in ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]:
        val = str(raw.get(col, "no"))
        processed[col] = BINARY_MAP.get(val, 0)

    # Ordinal
    processed["CAEC"]   = CAEC_MAP.get(str(raw.get("CAEC", "Sometimes")), 1)
    processed["CALC"]   = CALC_MAP.get(str(raw.get("CALC", "Sometimes")), 1)
    processed["MTRANS"] = MTRANS_MAP.get(str(raw.get("MTRANS", "Public_Transportation")), 3)

    # Susun sesuai urutan fitur
    arr = np.array([[processed[col] for col in FEATURE_COLUMNS]], dtype=float)
    return arr


def scale_input(arr: np.ndarray, scaler) -> np.ndarray:
    """Menerapkan MinMaxScaler yang sama seperti saat training."""
    return scaler.transform(arr)


# 3. PREDICT ONE SAMPLE

def predict_one(model, input_data: dict, scaler, verbose: bool = True) -> dict:

    X_raw    = preprocess_input(input_data)
    X_scaled = scale_input(X_raw, scaler)

    pred_idx   = model.predict(X_scaled)[0]
    pred_label = TARGET_LABELS[pred_idx]

    result = {
        "predicted_index": int(pred_idx),
        "predicted_label": pred_label,
    }

    # Probabilitas (jika model mendukung predict_proba)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        result["probabilities"] = {
            TARGET_LABELS[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

    if verbose:
        _print_prediction_result(result, input_data)

    return result


def _print_prediction_result(result: dict, input_data: dict) -> None:
    pred_label = result["predicted_label"]
    desc       = TARGET_DESCRIPTIONS.get(pred_label, "")

    print("\n" + "─" * 55)
    print("  HASIL PREDIKSI TINGKAT OBESITAS")
    print("─" * 55)
    print(f"  Data Input:")
    for k, v in input_data.items():
        print(f"    {k:<35} : {v}")
    print("─" * 55)
    print(f"  Prediksi  : {pred_label}")
    print(f"  Deskripsi : {desc}")

    if "probabilities" in result:
        print("\n  Probabilitas per kelas:")
        sorted_proba = sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for label, prob in sorted_proba:
            bar = "█" * int(prob * 30)
            print(f"    {label:<25} : {prob:.4f} |{bar}")

    print("─" * 55)


# 4. PREDICT BATCH (dari CSV)

def predict_batch(
    model,
    csv_path: str,
    scaler,
    output_path: str = None,
) -> pd.DataFrame:

    print(f"\n  [BATCH] Memuat data dari: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  • Jumlah sampel: {len(df)}")

    # Encode setiap baris
    X_list = []
    for _, row in df.iterrows():
        X_row = preprocess_input(row.to_dict())
        X_list.append(X_row[0])

    X_batch = np.array(X_list)
    X_batch_scaled = scale_input(X_batch, scaler)
    preds   = model.predict(X_batch_scaled)

    df["predicted_index"] = preds
    df["predicted_label"] = [TARGET_LABELS[i] for i in preds]
    df["description"]     = df["predicted_label"].map(TARGET_DESCRIPTIONS)

    # Tampilkan distribusi prediksi
    print("\n  Distribusi Prediksi:")
    vc = df["predicted_label"].value_counts()
    for label, cnt in vc.items():
        bar = "█" * cnt
        print(f"  {label:<25} : {cnt:>4} {bar}")

    # Simpan output
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "predictions_output.csv")

    df.to_csv(output_path, index=False)
    print(f"\n  ✔ Hasil prediksi disimpan: {output_path}")

    return df



# 5. INPUT INTERAKTIF


def interactive_input() -> dict:

    print("\n" + "=" * 55)
    print("  INPUT DATA PASIEN")
    print("  (Tekan Enter untuk menggunakan nilai default)")
    print("=" * 55)

    def ask(prompt, default, dtype=str):
        val = input(f"  {prompt} [{default}]: ").strip()
        if val == "":
            return default
        try:
            return dtype(val)
        except ValueError:
            print(f"    ⚠ Input tidak valid, menggunakan default: {default}")
            return default

    data = {}

    print("\n  ── Informasi Dasar ──────────────────────────────────")
    data["Gender"]  = ask("Jenis Kelamin (Male/Female)", "Male")
    data["Age"]     = ask("Usia (tahun)", 25, float)
    data["Height"]  = ask("Tinggi Badan (meter, contoh: 1.70)", 1.70, float)
    data["Weight"]  = ask("Berat Badan (kg)", 70, float)

    print("\n  ── Riwayat & Kebiasaan Makan ─────────────────────────")
    data["family_history_with_overweight"] = ask(
        "Riwayat keluarga obesitas (yes/no)", "no"
    )
    data["FAVC"] = ask("Sering makan tinggi kalori? (yes/no)", "no")
    data["FCVC"] = ask("Frekuensi makan sayuran (1=Tidak/2=Kadang/3=Selalu)", 2, float)
    data["NCP"]  = ask("Jumlah makan utama per hari (1–4)", 3, float)
    data["CAEC"] = ask(
        "Ngemil di luar jam makan (no/Sometimes/Frequently/Always)", "Sometimes"
    )
    data["CH2O"] = ask("Konsumsi air putih per hari (liter, 1–3)", 2, float)

    print("\n  ── Kebiasaan Lainnya ─────────────────────────────────")
    data["SMOKE"] = ask("Merokok? (yes/no)", "no")
    data["SCC"]   = ask("Monitor kalori? (yes/no)", "no")
    data["FAF"]   = ask("Frekuensi aktivitas fisik per minggu (0–3)", 1, float)
    data["TUE"]   = ask("Waktu penggunaan gadget per hari (jam, 0–2)", 1, float)
    data["CALC"]  = ask(
        "Frekuensi minum alkohol (no/Sometimes/Frequently/Always)", "no"
    )
    data["MTRANS"] = ask(
        "Transportasi utama (Walking/Bike/Motorbike/Public_Transportation/Automobile)",
        "Public_Transportation",
    )

    return data



# 6. CONTOH PREDIKSI DEMO

def run_demo_predictions(model, scaler) -> None:
    print("\n" + "=" * 65)
    print("  DEMO PREDIKSI – BEBERAPA KASUS CONTOH")
    print("=" * 65)

    demo_cases = [
        {
            "name"    : "Pria Muda, BMI Normal",
            "Gender"  : "Male",   "Age": 22, "Height": 1.78, "Weight": 72,
            "family_history_with_overweight": "no",
            "FAVC"    : "no",   "FCVC": 2, "NCP": 3,
            "CAEC"    : "Sometimes", "SMOKE": "no", "CH2O": 2,
            "SCC"     : "no",   "FAF": 2, "TUE": 1,
            "CALC"    : "no",   "MTRANS": "Public_Transportation",
        },
        {
            "name"    : "Wanita Dewasa, Riwayat Keluarga Obesitas",
            "Gender"  : "Female", "Age": 35, "Height": 1.62, "Weight": 85,
            "family_history_with_overweight": "yes",
            "FAVC"    : "yes",  "FCVC": 1, "NCP": 3,
            "CAEC"    : "Frequently", "SMOKE": "no", "CH2O": 2,
            "SCC"     : "no",   "FAF": 0, "TUE": 2,
            "CALC"    : "Sometimes", "MTRANS": "Automobile",
        },
        {
            "name"    : "Pria Paruh Baya, Aktif Olahraga",
            "Gender"  : "Male",   "Age": 45, "Height": 1.80, "Weight": 78,
            "family_history_with_overweight": "yes",
            "FAVC"    : "no",   "FCVC": 3, "NCP": 3,
            "CAEC"    : "Sometimes", "SMOKE": "no", "CH2O": 3,
            "SCC"     : "yes",  "FAF": 3, "TUE": 0,
            "CALC"    : "no",   "MTRANS": "Walking",
        },
        {
            "name"    : "Remaja, Kurang Aktivitas",
            "Gender"  : "Female", "Age": 17, "Height": 1.58, "Weight": 95,
            "family_history_with_overweight": "yes",
            "FAVC"    : "yes",  "FCVC": 1, "NCP": 4,
            "CAEC"    : "Always", "SMOKE": "no", "CH2O": 1,
            "SCC"     : "no",   "FAF": 0, "TUE": 2,
            "CALC"    : "no",   "MTRANS": "Automobile",
        },
    ]

    for i, case in enumerate(demo_cases, 1):
        name = case.pop("name")
        print(f"\n  ─── Demo [{i}]: {name} ───")
        predict_one(model, case, scaler, verbose=True)



# MAIN ENTRY POINT

def main() -> None:
    """
    Fungsi utama untuk menjalankan prediksi.

    Mode yang tersedia:
        1. Interaktif – input manual dari user
        2. Demo       – contoh prediksi otomatis
        3. Batch      – prediksi dari file CSV (via --batch <path>)
    """
    parser = argparse.ArgumentParser(
        description="Prediksi Tingkat Obesitas – A11.2024.15547"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path ke file model (.joblib). Default: Models/model_terbaik.joblib"
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Path ke file CSV untuk prediksi batch."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Jalankan demo prediksi dengan data contoh."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path output untuk hasil prediksi batch."
    )
    args = parser.parse_args()

    print("\n" + "█" * 65)
    print("  PREDIKSI TINGKAT OBESITAS")
    print("  Nama : Tegar Scaesario  |  NIM : A11.2024.15547")
    print("  Mata Kuliah : Machine Learning – A11.4410")
    print("█" * 65)

    # ── Muat model ────────────────────────────────────────────────────────
    print("\n  [LOAD] Memuat Model...")
    model, metadata, scaler = load_model(args.model)

    # ── Mode Batch ────────────────────────────────────────────────────────
    if args.batch:
        predict_batch(model, args.batch, scaler, args.output)
        return

    # ── Mode Demo ─────────────────────────────────────────────────────────
    if args.demo:
        run_demo_predictions(model, scaler)
        return

    # ── Mode Interaktif ───────────────────────────────────────────────────
    print("\n  Pilih mode:")
    print("  [1] Input manual (interaktif)")
    print("  [2] Demo prediksi otomatis")
    print("  [3] Keluar")

    choice = input("\n  Pilihan [1/2/3]: ").strip()

    if choice == "1":
        while True:
            input_data = interactive_input()
            predict_one(model, input_data, scaler, verbose=True)

            again = input("\n  Lakukan prediksi lagi? (y/n): ").strip().lower()
            if again != "y":
                break

    elif choice == "2":
        run_demo_predictions(model, scaler)

    elif choice == "3":
        print("\n  Terima kasih!\n")
        return

    else:
        print("\n  Pilihan tidak valid. Menjalankan demo...")
        run_demo_predictions(model, scaler)

    print("\n  Prediksi selesai. Terima kasih!\n")


if __name__ == "__main__":
    main()