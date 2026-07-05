#!/usr/bin/env python3
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# ── Tambahkan direktori root ke sys.path ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from Src.data_generator import run_pipeline, encode_features, generate_datasets, load_data, clean_data
from Src.ml_core        import (
    build_models,
    train_model,
    evaluate_model,
    cross_validate_model,
    compare_all_models,
    plot_learning_curve,
    save_best_model,
    CLASS_LABELS,
    SCENARIO_LABELS,
)

warnings.filterwarnings("ignore")

REPORT_DIR = os.path.join(BASE_DIR, "Reports")
os.makedirs(REPORT_DIR, exist_ok=True)



def run_cross_validation(datasets: dict) -> None:
    print("\n" + "=" * 65)
    print("  [CV] Cross-Validation – StratifiedKFold (5 fold)")
    print("=" * 65)

    
    X_train, X_test, y_train, y_test = datasets["Data Cleaning"]


    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    models = build_models()
    cv_results = []

    for model_name, model in models.items():
        print(f"\n  ► Cross-validating {model_name}...")
        cv_res = cross_validate_model(
            model=model,
            X=X_all,
            y=y_all,
            model_name=model_name,
            n_splits=5,
        )
        cv_results.append({
            "Model"        : cv_res["model"],
            "Mean Accuracy": round(cv_res["mean_accuracy"], 4),
            "Std Accuracy" : round(cv_res["std_accuracy"], 4),
        })

    df_cv = pd.DataFrame(cv_results).sort_values("Mean Accuracy", ascending=False)
    print("\n  --- Hasil Cross-Validation ---")
    print(df_cv.to_string(index=False))

    # Simpan hasil CV
    out_csv = os.path.join(REPORT_DIR, "cross_validation_results.csv")
    df_cv.to_csv(out_csv, index=False)
    print(f"\n  ✔ Hasil CV disimpan: {out_csv}")


# =============================================================================
# PIPELINE LEARNING CURVES
# =============================================================================


def run_learning_curves(datasets: dict) -> None:
    
    print("\n" + "=" * 65)
    print("  [LC] Learning Curves")
    print("=" * 65)

    X_train, X_test, y_train, y_test = datasets["Data Cleaning"]
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    models = build_models()

    for model_name, model in models.items():
        print(f"\n  ► Learning curve: {model_name}...")
        plot_learning_curve(
            model=model,
            X=X_all,
            y=y_all,
            model_name=model_name,
            scenario="Data Cleaning",
            cv=5,
        )


# =============================================================================
# GENERATE EXTRA REPORT FILES
# =============================================================================

def generate_all_experiment_results(df_results: pd.DataFrame) -> None:
    """
    Menghasilkan all_experiment_results.csv – tabel lengkap semua eksperimen
    dengan ranking, selisih dari model terbaik, dan kategori performa.

    Parameters
    ----------
    df_results : pd.DataFrame – Output dari compare_all_models().
    """
    print("\n" + "=" * 65)
    print("  [REPORT] Membuat all_experiment_results.csv")
    print("=" * 65)

    df = df_results.copy()

    # Prioritas:
    # 1. Accuracy tertinggi
    # 2. Jika sama, prioritaskan SVM
    # 3. Baru F1-Score

    df["Priority"] = (df["Model"] != "SVM").astype(int)

    df = (
        df.sort_values(
            by=["Accuracy", "Priority", "F1-Score"],
            ascending=[False, True, False]
        )
        .drop(columns="Priority")
        .reset_index(drop=True)
    )

    df.insert(0, "Rank", df.index + 1)

    # Selisih F1 dari model terbaik
    best_f1 = df["F1-Score"].iloc[0]
    df["Delta_F1"] = (df["F1-Score"] - best_f1).round(4)

    # Kategori performa berdasarkan Accuracy
    def kategori(acc):
        if acc >= 0.95:
            return "Sangat Baik"
        elif acc >= 0.90:
            return "Baik"
        elif acc >= 0.80:
            return "Cukup"
        elif acc >= 0.70:
            return "Kurang"
        else:
            return "Buruk"

    df["Kategori"] = df["Accuracy"].apply(kategori)

    # Apakah model terbaik di skenarionya?
    best_per_scenario = df.groupby("Scenario")["F1-Score"].transform("max")
    df["Best_in_Scenario"] = (df["F1-Score"] == best_per_scenario).map(
        {True: "Terbaik", False: ""}
    )

    out = os.path.join(REPORT_DIR, "all_experiment_results.csv")
    df.to_csv(out, index=False)
    print(f"  ✔ Disimpan: {out}")
    print(f"  • Total eksperimen : {len(df)} baris")
    print(f"  • Kolom            : {list(df.columns)}")


def generate_classification_reports(datasets: dict) -> None:
    """
    Menghasilkan classification_reports.json – laporan klasifikasi per kelas
    untuk setiap model dan skenario.

    Parameters
    ----------
    datasets : dict – Output dari data_generator.generate_datasets().
    """
    import json
    from sklearn.metrics import classification_report

    print("\n" + "=" * 65)
    print("  [REPORT] Membuat classification_reports.json")
    print("=" * 65)

    reports = {}

    for scenario, (X_train, X_test, y_train, y_test) in datasets.items():
        scenario_label = SCENARIO_LABELS.get(scenario, scenario)
        reports[scenario_label] = {}

        models = build_models()
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report_dict = classification_report(
                y_test, y_pred,
                target_names=CLASS_LABELS,
                output_dict=True,
                zero_division=0,
            )

            def round_dict(d):
                return {
                    k: (round(v, 4) if isinstance(v, float) else
                        round_dict(v) if isinstance(v, dict) else v)
                    for k, v in d.items()
                }

            reports[scenario_label][model_name] = round_dict(report_dict)
            print(f"  ✔ {model_name} [{scenario_label}]")

    out = os.path.join(REPORT_DIR, "classification_reports.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    print(f"\n  ✔ Disimpan: {out}")
    print(f"  • Berisi laporan untuk {len(reports)} skenario × "
          f"{len(list(reports.values())[0])} model")


def generate_audit_dataset(df_raw: pd.DataFrame) -> None:
    """
    Menghasilkan audit_dataset.json – informasi statistik lengkap tentang
    dataset: jumlah baris, tipe data, distribusi kelas, missing values,
    duplikat, dan statistik deskriptif tiap fitur numerik.

    Parameters
    ----------
    df_raw : pd.DataFrame – DataFrame mentah sebelum preprocessing.
    """
    import json
    from datetime import datetime

    print("\n" + "=" * 65)
    print("  [REPORT] Membuat audit_dataset.json")
    print("=" * 65)

    numeric_cols     = df_raw.select_dtypes(include="number").columns.tolist()
    categorical_cols = df_raw.select_dtypes(exclude="number").columns.tolist()

    target_dist = df_raw["NObeyesdad"].value_counts().to_dict()

    numeric_stats = {}
    for col in numeric_cols:
        s = df_raw[col]
        numeric_stats[col] = {
            "min"    : round(float(s.min()), 4),
            "max"    : round(float(s.max()), 4),
            "mean"   : round(float(s.mean()), 4),
            "median" : round(float(s.median()), 4),
            "std"    : round(float(s.std()), 4),
            "q1"     : round(float(s.quantile(0.25)), 4),
            "q3"     : round(float(s.quantile(0.75)), 4),
            "iqr"    : round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
            "missing": int(s.isnull().sum()),
        }

    categorical_stats = {}
    for col in categorical_cols:
        s = df_raw[col]
        categorical_stats[col] = {
            "unique_values": s.unique().tolist(),
            "value_counts" : s.value_counts().to_dict(),
            "missing"      : int(s.isnull().sum()),
        }

    outlier_info = {}
    for col in ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        if col not in df_raw.columns:
            continue
        s = df_raw[col]
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum())
        outlier_info[col] = {
            "n_outliers"  : n_out,
            "pct_outliers": round(n_out / len(df_raw) * 100, 2),
            "lower_bound" : round(float(Q1 - 1.5 * IQR), 4),
            "upper_bound" : round(float(Q3 + 1.5 * IQR), 4),
        }

    audit = {
        "metadata": {
            "nama"         : "Tegar Scaesario",
            "nim"          : "A11.2024.15547",
            "mata_kuliah"  : "Machine Learning – A11.4410",
            "tanggal_audit": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset"      : "ObesityDataSet_raw_and_data_sinthetic.csv",
            "sumber"       : "https://archive.ics.uci.edu/dataset/544",
        },
        "ringkasan": {
            "total_baris"    : int(len(df_raw)),
            "total_kolom"    : int(df_raw.shape[1]),
            "total_duplikat" : int(df_raw.duplicated().sum()),
            "total_missing"  : int(df_raw.isnull().sum().sum()),
            "jumlah_fitur"   : 16,
            "jumlah_kelas"   : 7,
            "kolom_numerik"  : numeric_cols,
            "kolom_kategorik": categorical_cols,
        },
        "distribusi_kelas_target": target_dist,
        "statistik_numerik"      : numeric_stats,
        "statistik_kategorik"    : categorical_stats,
        "analisis_outlier_iqr"   : outlier_info,
    }

    out = os.path.join(REPORT_DIR, "audit_dataset.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"  ✔ Disimpan: {out}")
    print(f"  • Total baris dataset  : {len(df_raw):,}")
    print(f"  • Total duplikat       : {df_raw.duplicated().sum()}")
    print(f"  • Total missing values : {df_raw.isnull().sum().sum()}")
    print(f"  • Kolom diaudit        : {df_raw.shape[1]}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    
    t_start = time.time()

    print("\n" + "█" * 65)
    print("  TRAINING PIPELINE – KLASIFIKASI TINGKAT OBESITAS")
    print("  Nama : Tegar Scaesario  |  NIM : A11.2024.15547")
    print("  Mata Kuliah : Machine Learning – A11.4410")
    print("█" * 65)

    # ─── Step 1: Preprocessing & dataset generation ───────────────────────
    print("\n  [STEP 1/6] Preprocessing & Pembuatan Dataset...")
    datasets = run_pipeline(save=True)
    df_raw = load_data()

    # ─── Step 2: Cross-Validation ──────────────────────────────────────────
    print("\n  [STEP 2/6] Cross-Validation...")
    run_cross_validation(datasets)

    # ─── Step 3: Training & Evaluasi ──────────────────────────────────────
    print("\n  [STEP 3/6] Training & Evaluasi Semua Model...")
    df_results = compare_all_models(datasets, verbose=True)

    # ─── Step 4: Learning Curves ───────────────────────────────────────────
    print("\n  [STEP 4/6] Learning Curves...")
    run_learning_curves(datasets)

    # ─── Step 5: Simpan model terbaik ─────────────────────────────────────
    print("\n  [STEP 5/6] Menyimpan Model Terbaik...")
    save_best_model(df_results, datasets)

    # ─── Step 6: Generate extra report files ──────────────────────────────
    print("\n  [STEP 6/6] Membuat File Laporan Tambahan...")
    generate_all_experiment_results(df_results)
    generate_classification_reports(datasets)
    generate_audit_dataset(df_raw)

    # ─── Ringkasan akhir ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "█" * 65)
    print("  TRAINING SELESAI")
    print(f"  Total waktu: {elapsed:.1f} detik ({elapsed/60:.1f} menit)")
    print("█" * 65)

    print("\n  Output yang dihasilkan:")
    print("  • Models/model_terbaik.joblib             – Model terbaik")
    print("  • Models/<nama>_<skenario>.joblib         – Semua model terlatih")
    print("  • Reports/model_comparison.csv            – Perbandingan model")
    print("  • Reports/cross_validation_results.csv    – Hasil cross-validation")
    print("  • Reports/all_experiment_results.csv      – Semua eksperimen + ranking")
    print("  • Reports/classification_reports.json     – Laporan per kelas tiap model")
    print("  • Reports/audit_dataset.json              – Statistik & audit dataset")
    print("  • Reports/*.png                           – Semua visualisasi")

    # Tampilkan 5 hasil terbaik
    print("\n  TOP-5 Model Terbaik")

    top5 = (
                df_results
                .assign(Priority=(df_results["Model"] != "SVM").astype(int))
                .sort_values(
                    by=["Accuracy", "Priority", "F1-Score"],
                    ascending=[False, True, False]
            )
            .drop(columns="Priority")
            .head(5)
        )

    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
