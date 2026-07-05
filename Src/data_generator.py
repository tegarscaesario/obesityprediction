import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
REPORT_DIR = os.path.join(BASE_DIR, "Reports")
DATASET_PATH = os.path.join(DATA_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")

os.makedirs(REPORT_DIR, exist_ok=True)

# ── Mapping Enkoding Manual ────────────────────────────────────────────────────
BINARY_MAP = {"yes": 1, "no": 0}

CAEC_MAP   = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
CALC_MAP   = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

GENDER_MAP = {"Female": 0, "Male": 1}

MTRANS_MAP = {
    "Walking": 0,
    "Bike": 1,
    "Public_Transportation": 2,
    "Motorbike": 3,
    "Automobile": 4,
}

TARGET_MAP = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6,
}

TARGET_LABELS = list(TARGET_MAP.keys())



# 1. LOAD DATA

def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    print("=" * 65)
    print("  [LOAD] Memuat Dataset")
    print("=" * 65)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)
    print(f"  ✔ Dataset berhasil dimuat")
    print(f"  • Jumlah baris  : {df.shape[0]:,}")
    print(f"  • Jumlah kolom  : {df.shape[1]}")
    print(f"  • Kolom         : {list(df.columns)}")
    return df



# 2. EKSPLORASI DATA 

def explore_data(df: pd.DataFrame, save_plots: bool = True) -> None:
    print("\n" + "=" * 65)
    print("Eksplorasi Data Awal")
    print("=" * 65)

    # ── Info Dasar ─────────────────────────────────────────────────────────
    print("\n  --- Tipe Data ---")
    print(df.dtypes.to_string())

    print("\n  --- Statistik Deskriptif (Numerik) ---")
    print(df.describe().to_string())

    # ── Missing Values ──────────────────────────────────────────────────────
    missing = df.isnull().sum()
    print("\n  --- Missing Values ---")
    if missing.sum() == 0:
        print("  ✔ Tidak ada missing value")
    else:
        print(missing[missing > 0])

    # ── Duplikat ────────────────────────────────────────────────────────────
    dupes = df.duplicated().sum()
    print(f"\n  --- Duplikat: {dupes} baris ---")

    # ── Distribusi Target ───────────────────────────────────────────────────
    print("\n  --- Distribusi Kelas Target ---")
    vc = df["NObeyesdad"].value_counts()
    for cls, cnt in vc.items():
        bar = "█" * (cnt // 30)
        print(f"  {cls:<25} | {cnt:>4} | {bar}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if save_plots:
        _plot_class_distribution(df)
        _plot_numeric_distributions(df)
        _plot_correlation_heatmap(df)


def _plot_class_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df["NObeyesdad"].value_counts().index.tolist()
    sns.countplot(data=df, x="NObeyesdad", order=order,
                  palette="viridis", ax=ax)
    ax.set_title("Distribusi Kelas Target (NObeyesdad)", fontsize=14)
    ax.set_xlabel("Kelas Obesitas")
    ax.set_ylabel("Jumlah")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "eda_class_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✔ Plot disimpan: {out}")


def _plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribusi Fitur Numerik", fontsize=15, y=1.01)
    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "eda_numeric_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Plot disimpan: {out}")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Heatmap Korelasi Fitur Numerik", fontsize=14)
    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "eda_correlation_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✔ Plot disimpan: {out}")



# 3. CLEANING DATA

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  [CLEAN] Pembersihan Data")
    print("=" * 65)

    n_before = len(df)

    # ── Hapus duplikat ──────────────────────────────────────────────────────
    df = df.drop_duplicates()
    n_after_dup = len(df)
    print(f"  • Duplikat dihapus    : {n_before - n_after_dup} baris")

    # ── Hapus outlier (IQR) pada kolom numerik ──────────────────────────────
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    mask = pd.Series([True] * len(df), index=df.index)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        col_mask = (df[col] >= lower) & (df[col] <= upper)
        removed = (~col_mask).sum()
        if removed:
            print(f"  • Outlier [{col:<8}]: {removed} baris dihapus "
                  f"(batas: [{lower:.2f}, {upper:.2f}])")
        mask = mask & col_mask

    df = df[mask].reset_index(drop=True)
    n_after_iqr = len(df)
    print(f"\n  ✔ Total baris setelah cleaning : {n_after_iqr:,} "
          f"(dari {n_before:,})")
    return df



# 4. ENCODING

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("\n" + "=" * 65)
    print("  [ENCODE] Encoding Fitur Kategorik")
    print("=" * 65)

    df = df.copy()

    # Binary
    for col in ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]:
        df[col] = df[col].map(BINARY_MAP)
        print(f"  ✔ Binary encode  : {col}")

    # Ordinal
    df["CAEC"]   = df["CAEC"].map(CAEC_MAP)
    df["CALC"]   = df["CALC"].map(CALC_MAP)
    df["Gender"] = df["Gender"].map(GENDER_MAP)
    df["MTRANS"] = df["MTRANS"].map(MTRANS_MAP)
    print("  ✔ Ordinal encode : CAEC, CALC, Gender, MTRANS")

    # Target
    y = df["NObeyesdad"].map(TARGET_MAP)
    X = df.drop(columns=["NObeyesdad"])
    print(f"  ✔ Target encode  : NObeyesdad → 7 kelas (0–6)")
    print(f"  • Shape X: {X.shape},  Shape y: {y.shape}")

    return X, y



# 5. NORMALISASI

def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "minmax",
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Normalisasi fitur menggunakan MinMaxScaler (default) atau RobustScaler.
    Mengembalikan (X_train_scaled, X_test_scaled, scaler) agar scaler
    dapat disimpan dan digunakan kembali saat prediksi.
    """
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler



# 6. GENERATE DATASETS (3 SKENARIO)

def generate_datasets(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_type: str = "minmax",
    save_scaler: bool = True,
) -> dict:
    """Membuat 3 skenario dataset (original, undersampling, SMOTE).
    Scaler yang di-fit pada data training disimpan ke Models/ secara default
    agar dapat digunakan kembali saat prediksi (predict.py / Streamlit).
    """
    print("\n" + "=" * 65)
    print("  [DATASETS] Membuat 3 Skenario Dataset")
    print("=" * 65)

    # ── Split ───────────────────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── Normalisasi (sesuai notebook: MinMaxScaler fit pada data clean train)
    X_train_scaled, X_test_scaled, scaler = normalize_features(
        X_train_raw, X_test_raw, scaler_type
    )

    # ── Simpan scaler ke Models/ agar predict.py / Streamlit bisa memuat ────
    if save_scaler:
        import joblib
        models_dir = os.path.join(BASE_DIR, "Models")
        os.makedirs(models_dir, exist_ok=True)
        scaler_name = f"scaler_{scaler_type}.joblib"
        scaler_path = os.path.join(models_dir, scaler_name)
        joblib.dump(scaler, scaler_path)
        print(f"  ✔ Scaler disimpan: {scaler_path}")

    datasets = {}

    # ── 1. Original ─────────────────────────────────────────────────────────
    datasets["original"] = (
        X_train_scaled, X_test_scaled,
        y_train.values, y_test.values,
    )
    dist = pd.Series(y_train.values).value_counts().sort_index()
    print(f"\n  [1] Original   – Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"      Distribusi kelas: {dict(dist)}")

    # ── 2. Undersampling ────────────────────────────────────────────────────
    rus = RandomUnderSampler(random_state=random_state)
    X_rus, y_rus = rus.fit_resample(X_train_scaled, y_train.values)
    dist_rus = pd.Series(y_rus).value_counts().sort_index()
    datasets["undersample"] = (X_rus, X_test_scaled, y_rus, y_test.values)
    print(f"\n  [2] Undersample – Train: {len(y_rus)}, Test: {len(y_test)}")
    print(f"      Distribusi kelas: {dict(dist_rus)}")

    # ── 3. SMOTE ────────────────────────────────────────────────────────────
    smote = SMOTE(random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train.values)
    dist_smt = pd.Series(y_smote).value_counts().sort_index()
    datasets["smote"] = (X_smote, X_test_scaled, y_smote, y_test.values)
    print(f"\n  [3] SMOTE       – Train: {len(y_smote)}, Test: {len(y_test)}")
    print(f"      Distribusi kelas: {dict(dist_smt)}")

    print("\n  ✔ Semua skenario dataset berhasil dibuat")
    return datasets



# 7. SAVE DATASETS

def save_datasets(datasets: dict, out_dir: str = DATA_DIR) -> None:
    """
    Menyimpan dataset yang telah diproses ke file CSV di folder Data.

    Parameters
    ----------
    datasets : dict – Output dari generate_datasets().
    out_dir  : str  – Direktori output.
    """
    print("\n" + "=" * 65)
    print("  [SAVE] Menyimpan Dataset yang Telah Diproses")
    print("=" * 65)

    os.makedirs(out_dir, exist_ok=True)

    for scenario, (X_train, X_test, y_train, y_test) in datasets.items():
        # Train
        df_train = pd.DataFrame(X_train)
        df_train["target"] = y_train
        path_train = os.path.join(out_dir, f"{scenario}_train.csv")
        df_train.to_csv(path_train, index=False)
        print(f"  ✔ Saved: {path_train}")

        # Test
        df_test = pd.DataFrame(X_test)
        df_test["target"] = y_test
        path_test = os.path.join(out_dir, f"{scenario}_test.csv")
        df_test.to_csv(path_test, index=False)
        print(f"  ✔ Saved: {path_test}")


# MAIN – Jalankan seluruh pipeline preprocessing

def run_pipeline(save: bool = True) -> dict:
    print("\n" + "█" * 65)
    print("  DATA GENERATOR PIPELINE")
    print("  Klasifikasi Tingkat Obesitas – A11.2024.15547")
    print("█" * 65)

    # 1. Load
    df = load_data()

    # 2. Explore (EDA)
    explore_data(df, save_plots=save)

    # 3. Clean
    df_clean = clean_data(df)

    # 4. Encode Original Data (Tanpa Cleaning)
    print("\n  [ENCODE] Memproses Data Original (Tanpa Cleaning)...")
    X_ori, y_ori = encode_features(df)

    # 5. Encode Cleaned Data (Dengan Cleaning)
    print("\n  [ENCODE] Memproses Data Cleaned (Dengan Cleaning)...")
    X_clean, y_clean = encode_features(df_clean)

    # 6. Generate datasets for Original (Tanpa simpan scaler utama)
    print("\n  [DATASETS] Membuat Skenario Data Original...")
    datasets_ori = generate_datasets(X_ori, y_ori, save_scaler=False)

    # 7. Generate datasets for Cleaned (Simpan scaler utama untuk modeling)
    print("\n  [DATASETS] Membuat Skenario Data Cleaned...")
    datasets_clean = generate_datasets(X_clean, y_clean, save_scaler=True)

    # Combine datasets
    combined_datasets = {
        "Data Ori"           : datasets_ori["original"],
        "Data Ori Under"     : datasets_ori["undersample"],
        "Data Ori Over"      : datasets_ori["smote"],
        "Data Cleaning"      : datasets_clean["original"],
        "Data Cleaning Under": datasets_clean["undersample"],
        "Data Cleaning Over" : datasets_clean["smote"],
    }

    # 8. (Optional) Save processed cleaned datasets for audit/reference
    if save:
        save_datasets(datasets_clean)

    print("\n" + "█" * 65)
    print("  ✔ PIPELINE SELESAI")
    print("█" * 65 + "\n")

    return combined_datasets


if __name__ == "__main__":
    datasets = run_pipeline(save=True)
