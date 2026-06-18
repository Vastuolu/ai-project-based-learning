# -*- coding: utf-8 -*-
# =============================================================================
# visualisasi.py
# Berisi fungsi-fungsi visualisasi untuk Decision Tree.
#
# Fungsi yang tersedia:
#   - visualisasi_confusion_matrix()    : heatmap confusion matrix (PNG)
#   - visualisasi_feature_importance()  : bar chart horizontal (PNG)
#   - visualisasi_distribusi_kelas()    : pie chart distribusi kelas (PNG)
#
# Library yang digunakan: matplotlib, numpy
# Semua grafik disimpan ke file PNG (backend Agg, tanpa membuka jendela GUI).
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend non-GUI: simpan ke file tanpa membuka jendela
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def visualisasi_confusion_matrix(
    cm:          "np.ndarray",
    class_names: list,
    output_dir:  str = "./output",
):
    """
    Menampilkan confusion matrix sebagai heatmap berwarna dan menyimpannya
    sebagai file PNG.

    Parameter:
        cm          : confusion matrix (numpy ndarray, shape NxN)
        class_names : daftar nama kelas (sesuai urutan encoding)
        output_dir  : direktori penyimpanan file PNG (default: direktori saat ini)
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Render heatmap confusion matrix menggunakan ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)

    ax.set_title("Confusion Matrix -- Decision Tree", fontsize=14,
                 fontweight="bold", pad=12)
    plt.tight_layout()

    # Simpan ke file PNG
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix disimpan ke '{path}'")


def visualisasi_feature_importance(
    importance:    "np.ndarray",
    feature_names: list,
    output_dir:    str = "./output",
):
    """
    Menampilkan bar chart horizontal feature importance dan menyimpannya
    sebagai file PNG. Fitur diurutkan dari importance tertinggi ke terendah.

    Parameter:
        importance    : array importance yang sudah dinormalisasi (total = 1.0)
        feature_names : daftar nama fitur (urutan sesuai kolom dataset)
        output_dir    : direktori penyimpanan file PNG
    """
    # Urutkan indeks fitur dari importance terbesar ke terkecil
    urutan        = np.argsort(importance)[::-1]
    nama_terurut  = [feature_names[i] for i in urutan]
    nilai_terurut = importance[urutan]

    # Buat warna gradasi biru tua -> biru muda
    n      = len(nama_terurut)
    colors = plt.cm.Blues(np.linspace(0.85, 0.30, n))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        nama_terurut[::-1], nilai_terurut[::-1],
        color=colors[::-1], edgecolor="white",
    )

    # Tambahkan label nilai di ujung setiap batang
    for bar, val in zip(bars, nilai_terurut[::-1]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Importance Score (dinormalisasi)", fontsize=11)
    ax.set_title("Feature Importance -- Decision Tree", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(nilai_terurut) * 1.22)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Simpan ke file PNG
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Feature importance disimpan ke '{path}'")


def visualisasi_distribusi_kelas(
    y_train:     "np.ndarray",
    y_test:      "np.ndarray",
    class_names: list,
    output_dir:  str = "./output",
):
    """
    Menampilkan distribusi kelas training dan testing sebagai dua pie chart
    berdampingan dan menyimpannya sebagai file PNG.

    Parameter:
        y_train     : array label kelas data training
        y_test      : array label kelas data testing
        class_names : daftar nama kelas
        output_dir  : direktori penyimpanan file PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    warna   = ["#4C72B0", "#DD8452"]          # biru dan oranye
    explode = [0.05] * len(class_names)       # sedikit pisahkan tiap irisan

    for ax, y_arr, judul in zip(axes, [y_train, y_test], ["Training Set", "Testing Set"]):
        # Hitung jumlah sampel tiap kelas untuk pie chart
        counts = [np.sum(y_arr == i) for i in range(len(class_names))]
        ax.pie(
            counts,
            labels     = class_names,
            autopct    = "%1.1f%%",
            colors     = warna,
            explode    = explode,
            startangle = 90,
            textprops  = {"fontsize": 11},
        )
        ax.set_title(judul, fontsize=13, fontweight="bold")

    plt.suptitle(
        "Distribusi Kelas Target (placement_status)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    # Simpan ke file PNG
    path = os.path.join(output_dir, "distribusi_kelas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Distribusi kelas disimpan ke '{path}'")
