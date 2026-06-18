# =============================================================================
# main.py
# Titik masuk utama program Decision Tree.
#
# Mengimpor modul-modul yang sudah dipisah:
#   - preprocessing  : load_and_preprocess()
#   - decision_tree  : bangun_pohon(), prediksi(), tampilkan_pohon(),
#                      hitung_feature_importance()
#   - evaluasi       : evaluasi_model()
#   - visualisasi    : visualisasi_confusion_matrix(),
#                      visualisasi_feature_importance(),
#                      visualisasi_distribusi_kelas()
# =============================================================================

import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
import numpy as np

# Import modul-modul yang sudah dipisah
from preprocessing  import load_and_preprocess
from decision_tree  import (
    bangun_pohon,
    prediksi,
    tampilkan_pohon,
    hitung_feature_importance,
)
from evaluasi       import evaluasi_model
from visualisasi    import (
    visualisasi_confusion_matrix,
    visualisasi_feature_importance,
    visualisasi_distribusi_kelas,
)


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH  = r"dataset\student_dataset_10000_rows.csv"
OUTPUT_DIR    = "./output"      # direktori penyimpanan file PNG

# Parameter Decision Tree
# Nilai optimal dipilih berdasarkan hasil grid search (parameter_testing.py):
#   Eksperimen menguji depth [1,2,3,4,5,6,8,10,12,15] x min_sampel [2,5,10,20,50,100]
#   Kombinasi depth=10, min_sampel=20 menghasilkan F1 macro tertinggi (0.7680)
MAX_DEPTH     = 10   # kedalaman maksimum pohon (optimal: lebih dalam = lebih ekspresif)
MIN_SAMPEL    = 20   # minimum sampel per node (optimal: mencegah split noise)
MAX_THRESHOLD = 20   # kandidat threshold per fitur (optimasi kecepatan)
TEST_SIZE     = 0.2  # proporsi data uji (20%)
RANDOM_SEED   = 42   # seed untuk reproduktivitas


# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 1 : LOAD DAN PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test, feature_names, class_names = load_and_preprocess(
    filepath     = DATASET_PATH,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 2 : VISUALISASI DISTRIBUSI KELAS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[INFO] Membuat visualisasi distribusi kelas...")
visualisasi_distribusi_kelas(y_train, y_test, class_names, OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 3 : BANGUN POHON KEPUTUSAN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"  MEMBANGUN DECISION TREE  (max_depth={MAX_DEPTH}, max_threshold={MAX_THRESHOLD})")
print("=" * 65)
print("\n[INFO] Membangun pohon... harap tunggu.")

t0 = time.time()

# Panggil fungsi rekursif utama untuk membangun pohon dari data training
pohon = bangun_pohon(
    X          = X_train,
    y          = y_train,
    kedalaman  = 0,
    max_depth  = MAX_DEPTH,
    min_sampel = MIN_SAMPEL,
)

print(f"[OK] Pohon berhasil dibangun dalam {time.time() - t0:.1f} detik.")


# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 4 : TAMPILKAN STRUKTUR POHON
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STRUKTUR POHON KEPUTUSAN (3 Level Pertama)")
print("=" * 65)
tampilkan_pohon(pohon, feature_names, class_names, batas_depth=3)


# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 5 : PREDIKSI PADA DATA UJI
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[INFO] Memprediksi {len(y_test)} sampel data uji...")
t1         = time.time()
y_prediksi = prediksi(pohon, X_test)
print(f"[OK] Prediksi selesai dalam {time.time() - t1:.1f} detik.")


# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 6 : EVALUASI MODEL
# ─────────────────────────────────────────────────────────────────────────────

akurasi, cm = evaluasi_model(y_test, y_prediksi, class_names)


# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 7 : FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  FEATURE IMPORTANCE")
print("=" * 65)

importance = hitung_feature_importance(pohon, len(feature_names))

print(f"\n  {'Fitur':<25s} {'Importance':>12s}")
print("  " + "-" * 38)
urutan = np.argsort(importance)[::-1]
for i in urutan:
    bar = "#" * int(importance[i] * 40)    # mini bar chart di terminal
    print(f"  {feature_names[i]:<25s} {importance[i]:>10.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# LANGKAH 8 : VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  MEMBUAT VISUALISASI")
print("=" * 65)
visualisasi_confusion_matrix(cm, class_names, OUTPUT_DIR)
visualisasi_feature_importance(importance, feature_names, OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# RINGKASAN AKHIR
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  RINGKASAN AKHIR")
print("=" * 65)
print(f"\n  Dataset          : {DATASET_PATH}")
print(f"  Jumlah Sampel    : {len(y_train)+len(y_test):,} (train={len(y_train):,} | test={len(y_test):,})")
print(f"  Jumlah Fitur     : {len(feature_names)}")
print(f"  Kedalaman Maks   : {MAX_DEPTH}")
print(f"  Akurasi Testing  : {akurasi * 100:.2f}%")
print(f"\n  File output:")
print(f"    - confusion_matrix.png")
print(f"    - feature_importance.png")
print(f"    - distribusi_kelas.png")
print("\n" + "=" * 65)
print("  SELESAI")
print("=" * 65)
