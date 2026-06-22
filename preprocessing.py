# =============================================================================
# preprocessing.py
# Berisi fungsi untuk memuat dan memproses dataset sebelum dilatih.
#
# Fungsi yang tersedia:
#   - load_and_preprocess() : membaca CSV, menangani missing values,
#                             encoding label, menghapus kolom data leakage,
#                             dan melakukan split manual 80:20.
# =============================================================================

import pandas as pd
import numpy as np
from collections import Counter


def load_and_preprocess(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
        1.1  Baca file CSV dengan pandas
        1.2  Tampilkan 5 baris pertama sebagai gambaran awal
        1.3  Periksa dan tangani missing values
        1.4  Encode label target string -> integer
        1.5  Hapus kolom yang menyebabkan data leakage (exam_score)
        1.6  Pisahkan fitur (X) dan target (y)
        1.7  Split manual 80:20 tanpa sklearn.train_test_split
    """
    print("=" * 65)
    print("  MEMUAT DAN MEMPROSES DATASET")
    print("=" * 65)

    # ── 1.1  Baca file CSV ────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"\n[OK] Dataset berhasil dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")

    # ── 1.2  Tampilkan 5 baris pertama sebagai gambaran awal ─────────────
    print("\n--- Sampel Data (5 baris pertama) ---")
    print(df.head().to_string(index=False))

    # ── 1.3  Periksa dan tangani missing values ───────────────────────────
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("\n[OK] Tidak ada missing values ditemukan.")
    else:
        print(f"\n[!] Ditemukan missing values:\n{missing[missing > 0]}")
        # Isi missing value: kolom numerik dengan median, kategorik dengan modus
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                    print(f"    -> Kolom '{col}' diisi dengan nilai median.")
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    print(f"    -> Kolom '{col}' diisi dengan nilai modus.")

    # ── 1.4  Encoding label target ────────────────────────────────────────
    target_col  = "placement_status"
    class_names = sorted(df[target_col].unique())   # ['Not Placed', 'Placed']
    label_map   = {cls: idx for idx, cls in enumerate(class_names)}
    df[target_col] = df[target_col].map(label_map)
    print(f"\n[OK] Encoding label target : {label_map}")

    # ── 1.5  Hapus kolom yang menyebabkan data leakage ───────────────────
    KOLOM_DIHAPUS = ["exam_score"]
    df = df.drop(columns=KOLOM_DIHAPUS, errors="ignore")
    print(f"[OK] Kolom dihapus (data leakage) : {KOLOM_DIHAPUS}")

    # ── 1.6  Pisahkan fitur (X) dan target (y) ───────────────────────────
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values.astype(float)   # bentuk: (N, n_fitur)
    y = df[target_col].values.astype(int)         # bentuk: (N,)

    # ── 1.7  Split manual 80:20 ───────────────────────────────────────────
    np.random.seed(random_state)              # seed untuk reproduktivitas
    indices = np.arange(len(y))
    np.random.shuffle(indices)               # acak urutan indeks

    batas = int(len(y) * (1 - test_size))   # titik potong antara train dan test
    train_idx, test_idx = indices[:batas], indices[batas:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"[OK] Split data  -> Training: {len(y_train)} | Testing: {len(y_test)}")
    print(f"     Kelas training : {dict(Counter(y_train))}")
    print(f"     Kelas testing  : {dict(Counter(y_test))}")

    return X_train, X_test, y_train, y_test, feature_names, class_names
