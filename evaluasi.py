# =============================================================================
# evaluasi.py
# Berisi fungsi untuk mengevaluasi performa model Decision Tree.
#
# Fungsi yang tersedia:
#   - evaluasi_model() : menghitung dan menampilkan akurasi,
#                        confusion matrix, dan classification report.
#
# Library yang digunakan: sklearn.metrics (HANYA untuk evaluasi)
# =============================================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def evaluasi_model(
    y_asli:      np.ndarray,
    y_prediksi:  np.ndarray,
    class_names: list,
):
    print("\n" + "=" * 65)
    print("  HASIL EVALUASI MODEL DECISION TREE")
    print("=" * 65)

    # ── Akurasi ───────────────────────────────────────────────────────────
    # Akurasi = jumlah prediksi benar / total prediksi
    akurasi = accuracy_score(y_asli, y_prediksi)
    print(f"\n[OK] Akurasi Model : {akurasi * 100:.2f}%\n")

    # ── Classification Report ─────────────────────────────────────────────
    # Menampilkan precision, recall, F1-score, dan support per kelas
    print("--- Classification Report ---")
    print(classification_report(y_asli, y_prediksi, target_names=class_names))

    # ── Confusion Matrix (format teks) ────────────────────────────────────
    # Baris = kelas aktual, Kolom = kelas prediksi
    cm = confusion_matrix(y_asli, y_prediksi)
    print("--- Confusion Matrix ---")
    header = f"{'':20s}" + "".join(f"{'Pred: '+cn:>16s}" for cn in class_names)
    print(header)
    for i, cn in enumerate(class_names):
        baris = f"{'Aktual: '+cn:20s}" + "".join(f"{v:>16d}" for v in cm[i])
        print(baris)

    return akurasi, cm
