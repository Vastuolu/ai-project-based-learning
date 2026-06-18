# =============================================================================
# demo_data_leakage.py
# Script DEMONSTRASI data leakage.
#
# Script ini SENGAJA menyertakan kolom exam_score sebagai fitur
# untuk menunjukkan efek data leakage:
#   - Akurasi artifisial 100%
#   - Pohon hanya menggunakan 1 fitur (exam_score)
#   - Model tidak realistis untuk digunakan di dunia nyata
# =============================================================================

import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Tambahkan folder root project ke sys.path agar bisa import modul
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score
)

# Import algoritma DT dari modul root
from decision_tree import bangun_pohon, prediksi, tampilkan_pohon, hitung_feature_importance


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(ROOT_DIR, r"dataset\student_dataset_10000_rows.csv")
OUTPUT_DIR   = os.path.join(ROOT_DIR, r"")
MAX_DEPTH    = 10
MIN_SAMPEL   = 20
TEST_SIZE    = 0.2
RANDOM_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — DENGAN exam_score (DATA LEAKAGE DISENGAJA)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  DEMO DATA LEAKAGE — exam_score DISERTAKAN SEBAGAI FITUR")
print("=" * 65)

df = pd.read_csv(DATASET_PATH)
print(f"\n[OK] Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")

# Encoding label target
target_col  = "placement_status"
class_names = sorted(df[target_col].unique())
label_map   = {cls: idx for idx, cls in enumerate(class_names)}
df[target_col] = df[target_col].map(label_map)
print(f"[OK] Encoding label: {label_map}")

# TIDAK menghapus exam_score
feature_names = [c for c in df.columns if c != target_col]
X = df[feature_names].values.astype(float)
y = df[target_col].values.astype(int)

print(f"\n[!] Fitur yang digunakan ({len(feature_names)} fitur, TERMASUK exam_score):")
for i, f in enumerate(feature_names, 1):
    marker = "  <-- DATA LEAKAGE" if f == "exam_score" else ""
    print(f"    {i}. {f}{marker}")

# Split manual 80:20
np.random.seed(RANDOM_SEED)
indices = np.arange(len(y))
np.random.shuffle(indices)
batas = int(len(y) * (1 - TEST_SIZE))
X_train, X_test = X[indices[:batas]], X[indices[batas:]]
y_train, y_test = y[indices[:batas]], y[indices[batas:]]
print(f"\n[OK] Split -> Training: {len(y_train)} | Testing: {len(y_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# BANGUN POHON
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"  MEMBANGUN POHON (max_depth={MAX_DEPTH}, DENGAN exam_score)")
print("=" * 65)
t0    = time.time()
pohon = bangun_pohon(X_train, y_train, max_depth=MAX_DEPTH, min_sampel=MIN_SAMPEL)
waktu_bangun = time.time() - t0
print(f"[OK] Pohon dibangun dalam {waktu_bangun:.2f} detik")


# ─────────────────────────────────────────────────────────────────────────────
# TAMPILKAN STRUKTUR POHON
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STRUKTUR POHON (3 Level Pertama) — DENGAN exam_score")
print("=" * 65)
tampilkan_pohon(pohon, feature_names, class_names, batas_depth=3)


# ─────────────────────────────────────────────────────────────────────────────
# PREDIKSI & EVALUASI LENGKAP
# ─────────────────────────────────────────────────────────────────────────────

y_pred = prediksi(pohon, X_test)
cm     = confusion_matrix(y_test, y_pred)

# ── Hitung semua metrik secara dinamis ────────────────────────────────────────
akurasi         = accuracy_score(y_test, y_pred)

prec_not_placed = precision_score(y_test, y_pred, pos_label=0)
rec_not_placed  = recall_score(y_test, y_pred, pos_label=0)
f1_not_placed   = f1_score(y_test, y_pred, pos_label=0)

prec_placed     = precision_score(y_test, y_pred, pos_label=1)
rec_placed      = recall_score(y_test, y_pred, pos_label=1)
f1_placed       = f1_score(y_test, y_pred, pos_label=1)

f1_macro        = f1_score(y_test, y_pred, average="macro")
f1_weighted     = f1_score(y_test, y_pred, average="weighted")

print("\n" + "=" * 65)
print("  HASIL EVALUASI — DENGAN exam_score (DATA LEAKAGE)")
print("=" * 65)
print(f"\n  [!] Akurasi Model : {akurasi * 100:.2f}%  <-- artifisial, tidak realistis")

# ── Classification Report ─────────────────────────────────────────────────────
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── Tabel metrik per kelas ────────────────────────────────────────────────────
print("--- Metrik Per Kelas ---")
print(f"  {'Kelas':<18s} {'Precision':>10s} {'Recall':>8s} {'F1-Score':>10s} {'Support':>9s}")
print("  " + "-" * 58)
print(f"  {'Not Placed':<18s} {prec_not_placed:>10.4f} {rec_not_placed:>8.4f} {f1_not_placed:>10.4f} {cm[0].sum():>9d}")
print(f"  {'Placed':<18s} {prec_placed:>10.4f} {rec_placed:>8.4f} {f1_placed:>10.4f} {cm[1].sum():>9d}")
print("  " + "-" * 58)
print(f"  {'F1 Macro':<18s} {'':>10s} {'':>8s} {f1_macro:>10.4f}")
print(f"  {'F1 Weighted':<18s} {'':>10s} {'':>8s} {f1_weighted:>10.4f}")
print(f"  {'Akurasi':<18s} {'':>10s} {'':>8s} {akurasi:>10.4f}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
print("\n--- Confusion Matrix ---")
header = f"{'':20s}" + "".join(f"{'Pred: '+cn:>16s}" for cn in class_names)
print(header)
for i, cn in enumerate(class_names):
    baris = f"{'Aktual: '+cn:20s}" + "".join(f"{v:>16d}" for v in cm[i])
    print(baris)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  FEATURE IMPORTANCE — BUKTI DATA LEAKAGE")
print("=" * 65)
importance = hitung_feature_importance(pohon, len(feature_names))
print(f"\n  {'Fitur':<25s} {'Importance':>12s}  {'Bar':}")
print("  " + "-" * 55)
urutan = np.argsort(importance)[::-1]
for i in urutan:
    bar    = "#" * int(importance[i] * 40)
    marker = "  <-- LEAKAGE (100%)" if feature_names[i] == "exam_score" else ""
    print(f"  {feature_names[i]:<25s} {importance[i]:>10.4f}  {bar}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI — CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    cmap="Reds", ax=ax, colorbar=True
)
ax.set_title(
    f"Confusion Matrix — DENGAN exam_score (Data Leakage)\n"
    f"Akurasi: {akurasi*100:.2f}%  |  F1 Macro: {f1_macro:.4f}  (Tidak Realistis)",
    fontsize=11, fontweight="bold", pad=10
)
plt.tight_layout()
path_cm = os.path.join(OUTPUT_DIR, "leakage_confusion_matrix.png")
plt.savefig(path_cm, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[OK] Confusion matrix disimpan ke '{path_cm}'")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

urutan_imp    = np.argsort(importance)[::-1]
nama_terurut  = [feature_names[i] for i in urutan_imp]
nilai_terurut = importance[urutan_imp]
colors        = ["#e74c3c" if n == "exam_score" else "#3498db" for n in nama_terurut]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(nama_terurut[::-1], nilai_terurut[::-1],
               color=colors[::-1], edgecolor="white")
for bar, val in zip(bars, nilai_terurut[::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)

ax.set_xlabel("Importance Score (dinormalisasi)", fontsize=11)
ax.set_title(
    "Feature Importance — DENGAN exam_score\n"
    "(exam_score mendominasi 100% → Data Leakage)",
    fontsize=12, fontweight="bold"
)
ax.set_xlim(0, max(nilai_terurut) * 1.25)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.legend(
    handles=[
        mpatches.Patch(color="#e74c3c", label="exam_score (DATA LEAKAGE)"),
        mpatches.Patch(color="#3498db", label="Fitur normal"),
    ],
    loc="lower right"
)
plt.tight_layout()
path_fi = os.path.join(OUTPUT_DIR, "leakage_feature_importance.png")
plt.savefig(path_fi, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Feature importance disimpan ke '{path_fi}'")


# ─────────────────────────────────────────────────────────────────────────────
# RINGKASAN PERBANDINGAN (dihitung dinamis)
# ─────────────────────────────────────────────────────────────────────────────

# Nilai model tanpa leakage (parameter optimal)
NO_LEAK = {
    "akurasi"      : 0.8740,
    "f1_not_placed": 0.6111,
    "f1_placed"    : 0.9248,
    "f1_macro"     : 0.7680,
    "f1_weighted"  : 0.8681,
    "fitur"        : "6 (semua fitur)",
    "waktu"        : "~1.0 detik",
}

print("\n" + "=" * 65)
print("  PERBANDINGAN LENGKAP: DENGAN vs TANPA exam_score")
print("=" * 65)
print(f"\n  {'Metrik':<22s} {'DENGAN exam_score':>20s} {'TANPA exam_score':>18s}")
print("  " + "-" * 62)
print(f"  {'Akurasi':<22s} {akurasi*100:>19.2f}% {NO_LEAK['akurasi']*100:>17.2f}%")
print(f"  {'Precision (Not Placed)':<22s} {prec_not_placed:>20.4f} {'-':>18s}")
print(f"  {'Recall    (Not Placed)':<22s} {rec_not_placed:>20.4f} {'-':>18s}")
print(f"  {'F1-Score  (Not Placed)':<22s} {f1_not_placed:>20.4f} {NO_LEAK['f1_not_placed']:>18.4f}")
print(f"  {'Precision (Placed)':<22s} {prec_placed:>20.4f} {'-':>18s}")
print(f"  {'Recall    (Placed)':<22s} {rec_placed:>20.4f} {'-':>18s}")
print(f"  {'F1-Score  (Placed)':<22s} {f1_placed:>20.4f} {NO_LEAK['f1_placed']:>18.4f}")
print(f"  {'F1 Macro':<22s} {f1_macro:>20.4f} {NO_LEAK['f1_macro']:>18.4f}")
print(f"  {'F1 Weighted':<22s} {f1_weighted:>20.4f} {NO_LEAK['f1_weighted']:>18.4f}")
print("  " + "-" * 62)
print(f"  {'Fitur dipakai pohon':<22s} {'1 (exam_score)':>20s} {NO_LEAK['fitur']:>18s}")
print(f"  {'Waktu bangun pohon':<22s} {f'{waktu_bangun:.2f} detik':>20s} {NO_LEAK['waktu']:>18s}")
print(f"  {'Berguna secara praktis':<22s} {'TIDAK':>20s} {'YA':>18s}")
print(f"  {'Realistis?':<22s} {'TIDAK (LEAKAGE)':>20s} {'YA':>18s}")

print("\n" + "=" * 65)
print("  FILE OUTPUT")
print("=" * 65)
print(f"  - {path_cm}")
print(f"  - {path_fi}")
print("\n" + "=" * 65)
print("  SELESAI")
print("=" * 65)
