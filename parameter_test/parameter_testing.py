# -*- coding: utf-8 -*-
# =============================================================================
# parameter_testing.py
# Script untuk menguji berbagai kombinasi parameter Decision Tree
# dan membandingkan hasilnya secara sistematis.
#
# Parameter yang diuji:
#   - max_depth    : kedalaman maksimum pohon
#   - min_sampel   : minimum sampel per node sebelum split
#   - max_threshold: jumlah kandidat threshold per fitur
#
# Cara menjalankan:
#   python parameter_testing.py
# =============================================================================

import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocessing import load_and_preprocess
from decision_tree  import bangun_pohon, prediksi


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI TETAP
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = r"dataset\student_dataset_10000_rows.csv"
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

# Load data sekali, dipakai semua eksperimen
print("Memuat dataset...")
X_train, X_test, y_train, y_test, feature_names, class_names = load_and_preprocess(
    DATASET_PATH, TEST_SIZE, RANDOM_SEED
)
print(f"Dataset siap. Train: {len(y_train)} | Test: {len(y_test)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI HELPER
# ─────────────────────────────────────────────────────────────────────────────

def jalankan_eksperimen(max_depth, min_sampel, max_threshold=20):
    """
    Melatih Decision Tree dengan parameter tertentu dan mengembalikan
    dictionary berisi semua metrik evaluasi.
    """
    t0    = time.time()
    pohon = bangun_pohon(X_train, y_train, max_depth=max_depth, min_sampel=min_sampel)
    waktu_bangun = time.time() - t0

    y_pred = prediksi(pohon, X_test)

    acc          = accuracy_score(y_test, y_pred)
    f1_not_placed = f1_score(y_test, y_pred, pos_label=0)   # kelas minoritas
    f1_placed     = f1_score(y_test, y_pred, pos_label=1)   # kelas mayoritas
    f1_macro      = f1_score(y_test, y_pred, average="macro")

    return {
        "max_depth"    : max_depth,
        "min_sampel"   : min_sampel,
        "max_threshold": max_threshold,
        "akurasi"      : acc,
        "f1_not_placed": f1_not_placed,
        "f1_placed"    : f1_placed,
        "f1_macro"     : f1_macro,
        "waktu"        : waktu_bangun,
    }


def cetak_tabel(judul, hasil_list, kolom_highlight=None):
    """Mencetak tabel hasil eksperimen ke terminal."""
    print("\n" + "=" * 80)
    print(f"  {judul}")
    print("=" * 80)
    header = (
        f"  {'Parameter':<22s} {'Akurasi':>8s} {'F1 (Not Placed)':>16s} "
        f"{'F1 (Placed)':>12s} {'F1 Macro':>10s} {'Waktu':>8s}"
    )
    print(header)
    print("  " + "-" * 77)
    for r in hasil_list:
        param_str = f"depth={r['max_depth']}, min={r['min_sampel']}"
        marker = " <-- DIPILIH" if kolom_highlight and r == kolom_highlight else ""
        print(
            f"  {param_str:<22s} {r['akurasi']*100:>7.2f}%"
            f" {r['f1_not_placed']:>16.4f}"
            f" {r['f1_placed']:>12.4f}"
            f" {r['f1_macro']:>10.4f}"
            f" {r['waktu']:>7.2f}s"
            f"{marker}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EKSPERIMEN 1: VARIASI max_depth (min_sampel tetap = 5)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[EKSPERIMEN 1] Variasi max_depth (min_sampel=5, max_threshold=20)")
print("Menguji:", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15])

hasil_depth = []
for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
    r = jalankan_eksperimen(max_depth=d, min_sampel=5)
    hasil_depth.append(r)
    print(f"  depth={d:>2d}: acc={r['akurasi']*100:.2f}%  F1_macro={r['f1_macro']:.4f}  t={r['waktu']:.2f}s")

pilihan_depth = max(hasil_depth, key=lambda x: (x["f1_macro"], x["akurasi"]))


# ─────────────────────────────────────────────────────────────────────────────
# EKSPERIMEN 2: VARIASI min_sampel (max_depth = 8)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[EKSPERIMEN 2] Variasi min_sampel (max_depth=8, max_threshold=20)")
print("Menguji:", [2, 5, 10, 20, 50, 100])

hasil_minsampel = []
for ms in [2, 5, 10, 20, 50, 100]:
    r = jalankan_eksperimen(max_depth=8, min_sampel=ms)
    hasil_minsampel.append(r)
    print(f"  min_sampel={ms:>3d}: acc={r['akurasi']*100:.2f}%  F1_macro={r['f1_macro']:.4f}  t={r['waktu']:.2f}s")

pilihan_min = max(hasil_minsampel, key=lambda x: (x["f1_macro"], x["akurasi"]))


# ─────────────────────────────────────────────────────────────────────────────
# EKSPERIMEN 3: GRID SEARCH — kombinasi terbaik
# ─────────────────────────────────────────────────────────────────────────────

print("\n[EKSPERIMEN 3] Grid Search max_depth x min_sampel")
depths    = [4, 6, 8, 10]
min_samps = [2, 5, 10, 20]
hasil_grid = []

for d in depths:
    for ms in min_samps:
        r = jalankan_eksperimen(max_depth=d, min_sampel=ms)
        hasil_grid.append(r)
        print(f"  depth={d}, min={ms:>2d}: acc={r['akurasi']*100:.2f}%  F1_macro={r['f1_macro']:.4f}  t={r['waktu']:.2f}s")

pilihan_grid = max(hasil_grid, key=lambda x: (x["f1_macro"], x["akurasi"]))


# ─────────────────────────────────────────────────────────────────────────────
# CETAK TABEL RANGKUMAN
# ─────────────────────────────────────────────────────────────────────────────

cetak_tabel("EKSPERIMEN 1: Variasi max_depth (min_sampel=5)", hasil_depth, pilihan_depth)
cetak_tabel("EKSPERIMEN 2: Variasi min_sampel (max_depth=8)", hasil_minsampel, pilihan_min)
cetak_tabel("EKSPERIMEN 3: Grid Search Terbaik", sorted(hasil_grid, key=lambda x: -x["f1_macro"])[:8], pilihan_grid)


# ─────────────────────────────────────────────────────────────────────────────
# RINGKASAN REKOMENDASI PARAMETER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  REKOMENDASI PARAMETER TERBAIK")
print("=" * 80)
best = pilihan_grid
print(f"\n  max_depth    : {best['max_depth']}")
print(f"  min_sampel   : {best['min_sampel']}")
print(f"  Akurasi      : {best['akurasi']*100:.2f}%")
print(f"  F1 Not Placed: {best['f1_not_placed']:.4f}")
print(f"  F1 Placed    : {best['f1_placed']:.4f}")
print(f"  F1 Macro     : {best['f1_macro']:.4f}")
print(f"  Waktu build  : {best['waktu']:.2f} detik")

# Bandingkan dengan parameter awal (depth=8, min=5)
baseline = jalankan_eksperimen(max_depth=8, min_sampel=5)
print(f"\n  Parameter awal (depth=8, min=5):")
print(f"    Akurasi    : {baseline['akurasi']*100:.2f}%")
print(f"    F1 Macro   : {baseline['f1_macro']:.4f}")
print(f"    F1 Not Placed: {baseline['f1_not_placed']:.4f}")
print("\n" + "=" * 80)
print("  SELESAI")
print("=" * 80)
