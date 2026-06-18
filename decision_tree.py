# =============================================================================
# decision_tree.py
# Berisi implementasi algoritma Decision Tree dari.
#
# Kelas dan fungsi yang tersedia:
#   - Node                    : representasi satu simpul dalam pohon
#   - hitung_entropy()        : menghitung nilai Shannon Entropy
#   - hitung_information_gain(): menghitung Information Gain setelah split
#   - pilih_fitur_terbaik()   : memilih (fitur, threshold) dengan IG tertinggi
#   - bangun_pohon()          : membangun pohon secara rekursif
#   - prediksi_satu()         : prediksi satu sampel (rekursif)
#   - prediksi()              : prediksi banyak sampel sekaligus
#   - hitung_feature_importance(): menghitung kontribusi IG per fitur
#   - tampilkan_pohon()       : menampilkan struktur pohon dalam format teks
#
# Library yang digunakan: numpy, collections.Counter
# =============================================================================

import numpy as np
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# 1. KELAS NODE
# ─────────────────────────────────────────────────────────────────────────────

class Node:
    """
    Merepresentasikan satu simpul (node) dalam pohon keputusan.

    Atribut:
        feature_index : indeks kolom fitur yang digunakan untuk split
        threshold     : nilai ambang batas pemisahan
        left          : anak kiri  (sampel dengan nilai <= threshold)
        right         : anak kanan (sampel dengan nilai >  threshold)
        value         : label kelas mayoritas (hanya untuk leaf / daun)
        info_gain     : information gain dari split terbaik node ini
    """
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        info_gain=None,
    ):
        self.feature_index = feature_index   # fitur pembagi pada node ini
        self.threshold     = threshold       # batas nilai pemisahan
        self.left          = left            # subtree anak kiri
        self.right         = right           # subtree anak kanan
        self.value         = value           # prediksi kelas (leaf node)
        self.info_gain     = info_gain       # information gain split ini

    def is_leaf(self) -> bool:
        """Mengembalikan True jika node ini adalah daun (leaf)."""
        return self.value is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. FUNGSI ENTROPY
# ─────────────────────────────────────────────────────────────────────────────

def hitung_entropy(y: np.ndarray) -> float:
    """
    Menghitung nilai Entropy dari sekumpulan label kelas.

    Rumus Shannon Entropy:
        H(S) = -sum( p_i * log2(p_i) )
        p_i  = proporsi kelas ke-i dalam S

    Sifat:
        H = 0   -> semua sampel satu kelas (sangat murni / pure)
        H maks  -> semua kelas terdistribusi merata (sangat tidak murni)

    Parameter:
        y : array label kelas (integer), shape (n,)

    Return:
        float : nilai entropy
    """
    n = len(y)
    if n == 0:
        return 0.0   # tidak ada sampel -> entropy nol

    # Hitung frekuensi setiap kelas yang unik
    _, counts = np.unique(y, return_counts=True)
    # Hitung proporsi (probabilitas) setiap kelas
    proporsi  = counts / n
    # Hitung entropy; tambahkan 1e-12 untuk menghindari log(0) = -inf
    entropy   = -np.sum(proporsi * np.log2(proporsi + 1e-12))
    return float(entropy)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FUNGSI INFORMATION GAIN
# ─────────────────────────────────────────────────────────────────────────────

def hitung_information_gain(
    y_parent: np.ndarray,
    y_kiri:   np.ndarray,
    y_kanan:  np.ndarray,
) -> float:
    """
    Menghitung Information Gain setelah melakukan satu split.

    Rumus:
        IG = H(parent) - [ (|kiri|/|parent|) * H(kiri)
                         + (|kanan|/|parent|) * H(kanan) ]

    Semakin besar IG, semakin baik split tersebut dalam
    memisahkan kelas-kelas yang berbeda.

    Parameter:
        y_parent : label sampel sebelum split (simpul induk)
        y_kiri   : label sampel anak kiri  (nilai <= threshold)
        y_kanan  : label sampel anak kanan (nilai >  threshold)

    Return:
        float : nilai information gain
    """
    n  = len(y_parent)
    nL = len(y_kiri)
    nR = len(y_kanan)

    # Entropy simpul induk (sebelum split)
    H_parent = hitung_entropy(y_parent)

    # Entropy tertimbang dari kedua anak setelah split
    H_anak = (nL / n) * hitung_entropy(y_kiri) + \
             (nR / n) * hitung_entropy(y_kanan)

    return H_parent - H_anak


# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNGSI PILIH FITUR TERBAIK
# ─────────────────────────────────────────────────────────────────────────────

def pilih_fitur_terbaik(
    X:             np.ndarray,
    y:             np.ndarray,
    max_threshold: int = 20,
) -> dict:
    """
    Mencari fitur dan nilai threshold terbaik untuk melakukan split,
    berdasarkan Information Gain tertinggi.

    Optimasi kecepatan:
        Untuk setiap fitur, jika jumlah nilai unik > max_threshold,
        dipilih max_threshold kandidat threshold secara merata (linspace)
        dari nilai minimum ke maksimum. Ini mempercepat proses secara
        signifikan tanpa mengorbankan kualitas split secara berarti.

    Parameter:
        X             : matriks fitur  (n_sampel x n_fitur)
        y             : array label    (n_sampel,)
        max_threshold : batas maksimum kandidat threshold per fitur

    Return:
        dict berisi kunci:
            'feature_index'  : indeks fitur terbaik
            'threshold'      : nilai ambang batas terbaik
            'info_gain'      : nilai IG terbaik
            'dataset_kiri'   : tuple (X_kiri, y_kiri)
            'dataset_kanan'  : tuple (X_kanan, y_kanan)
        None jika tidak ada split yang menghasilkan IG > 0
    """
    n_fitur    = X.shape[1]
    ig_terbaik = -1         # information gain terbaik (inisialisasi negatif)
    split_terbaik = None    # menyimpan detail split terbaik

    for fi in range(n_fitur):
        kolom      = X[:, fi]
        nilai_unik = np.unique(kolom)

        # Buat kandidat threshold sebagai nilai tengah antar nilai unik berurutan
        kandidat = (nilai_unik[:-1] + nilai_unik[1:]) / 2.0

        # Batasi jumlah kandidat jika terlalu banyak (optimasi kecepatan)
        if len(kandidat) > max_threshold:
            # Ambil max_threshold titik yang terdistribusi merata
            idx_pilih = np.linspace(0, len(kandidat) - 1, max_threshold, dtype=int)
            kandidat  = kandidat[idx_pilih]

        for threshold in kandidat:
            # Bagi dataset berdasarkan threshold
            mask_kiri  = kolom <= threshold
            mask_kanan = ~mask_kiri

            y_kiri  = y[mask_kiri]
            y_kanan = y[mask_kanan]

            # Lewati jika salah satu sisi kosong (split tidak valid)
            if len(y_kiri) == 0 or len(y_kanan) == 0:
                continue

            # Hitung information gain untuk split ini
            ig = hitung_information_gain(y, y_kiri, y_kanan)

            # Perbarui split terbaik jika IG lebih tinggi
            if ig > ig_terbaik:
                ig_terbaik    = ig
                split_terbaik = {
                    "feature_index" : fi,
                    "threshold"     : threshold,
                    "info_gain"     : ig,
                    "dataset_kiri"  : (X[mask_kiri],  y_kiri),
                    "dataset_kanan" : (X[mask_kanan], y_kanan),
                }

    return split_terbaik


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNGSI BANGUN POHON (REKURSIF)
# ─────────────────────────────────────────────────────────────────────────────

def bangun_pohon(
    X:          np.ndarray,
    y:          np.ndarray,
    kedalaman:  int = 0,
    max_depth:  int = 5,
    min_sampel: int = 2,
) -> Node:
    """
    Membangun pohon keputusan secara rekursif (algoritma ID3/CART sederhana).

    Strategi Divide-and-Conquer:
        1. Cari split terbaik (fitur + threshold) berdasarkan IG
        2. Bagi dataset menjadi dua subset (kiri dan kanan)
        3. Rekursif bangun subtree untuk masing-masing subset
        4. Berhenti jika kondisi penghentian terpenuhi

    Kondisi berhenti (base case):
        a. Kedalaman sudah mencapai max_depth
        b. Semua sampel di node ini berasal dari satu kelas (entropy = 0)
        c. Jumlah sampel di bawah min_sampel

    Parameter:
        X          : matriks fitur (n_sampel x n_fitur)
        y          : array label kelas (n_sampel,)
        kedalaman  : kedalaman node saat ini dalam pohon
        max_depth  : batas maksimum kedalaman pohon
        min_sampel : jumlah minimum sampel agar node bisa di-split

    Return:
        Node -- simpul hasil (internal node atau leaf node)
    """
    n_sampel = len(y)
    n_kelas  = len(np.unique(y))

    # ── Kondisi berhenti: buat leaf node ──────────────────────────────────
    if (
        kedalaman >= max_depth   # sudah mencapai kedalaman maksimum
        or n_kelas == 1          # semua sampel satu kelas (murni)
        or n_sampel < min_sampel # terlalu sedikit sampel untuk split
    ):
        # Nilai leaf = kelas yang paling banyak muncul (majority vote)
        nilai_daun = int(Counter(y).most_common(1)[0][0])
        return Node(value=nilai_daun)

    # ── Cari split terbaik ────────────────────────────────────────────────
    split = pilih_fitur_terbaik(X, y)

    # Jika tidak ada split yang berguna (IG = 0 atau None), buat leaf node
    if split is None or split["info_gain"] <= 0:
        nilai_daun = int(Counter(y).most_common(1)[0][0])
        return Node(value=nilai_daun)

    # ── Rekursi: bangun subtree kiri dan kanan ────────────────────────────
    X_kiri,  y_kiri  = split["dataset_kiri"]
    X_kanan, y_kanan = split["dataset_kanan"]

    anak_kiri  = bangun_pohon(X_kiri,  y_kiri,  kedalaman + 1, max_depth, min_sampel)
    anak_kanan = bangun_pohon(X_kanan, y_kanan, kedalaman + 1, max_depth, min_sampel)

    # Kembalikan internal node dengan informasi split terbaik
    return Node(
        feature_index = split["feature_index"],
        threshold     = split["threshold"],
        left          = anak_kiri,
        right         = anak_kanan,
        info_gain     = split["info_gain"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. FUNGSI PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────

def prediksi_satu(node: Node, x: np.ndarray) -> int:
    """
    Memprediksi kelas untuk SATU sampel dengan cara menelusuri pohon
    dari root hingga leaf.

    Cara kerja:
        - Jika node adalah leaf -> kembalikan nilai prediksi
        - Jika nilai fitur[node.feature_index] <= threshold
          -> telusuri anak kiri
        - Jika nilai fitur[node.feature_index] >  threshold
          -> telusuri anak kanan

    Parameter:
        node : simpul awal penelusuran (biasanya root)
        x    : satu vektor fitur, shape (n_fitur,)

    Return:
        int : label kelas prediksi
    """
    # Base case: sudah mencapai leaf node
    if node.is_leaf():
        return node.value

    # Tentukan arah berdasarkan nilai fitur vs threshold
    if x[node.feature_index] <= node.threshold:
        return prediksi_satu(node.left,  x)   # ke anak kiri
    else:
        return prediksi_satu(node.right, x)   # ke anak kanan


def prediksi(pohon: Node, X: np.ndarray) -> np.ndarray:
    """
    Memprediksi kelas untuk BANYAK sampel sekaligus.

    Parameter:
        pohon : simpul akar pohon keputusan
        X     : matriks fitur, shape (n_sampel, n_fitur)

    Return:
        numpy array shape (n_sampel,) berisi label kelas prediksi
    """
    # Terapkan prediksi_satu ke setiap baris (sampel) dalam X
    return np.array([prediksi_satu(pohon, baris) for baris in X])


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def hitung_feature_importance(
    node:       Node,
    n_fitur:    int,
    importance: np.ndarray = None,
    bobot:      float = 1.0,
) -> np.ndarray:
    """
    Menghitung feature importance berdasarkan akumulasi Information Gain
    yang dikontribusikan setiap fitur di seluruh node pohon.

    Setiap node internal berkontribusi:
        importance[fi] += bobot_node * info_gain_node
    dimana bobot_node dihitung secara rekursif (dikecilkan tiap level).

    Parameter:
        node       : simpul pohon yang sedang diproses
        n_fitur    : jumlah total fitur
        importance : array akumulasi (diinisialisasi otomatis pada root)
        bobot      : bobot kontribusi node ini (dikurangi x0.5 tiap level)

    Return:
        numpy array shape (n_fitur,) dinormalisasi sehingga total = 1.0
    """
    # Inisialisasi array importance pada pemanggilan pertama (root)
    if importance is None:
        importance = np.zeros(n_fitur)

    # Leaf node tidak memiliki split -> tidak berkontribusi
    if node.is_leaf():
        return importance

    fi = node.feature_index
    # Tambahkan kontribusi node ini ke fitur yang digunakan sebagai splitter
    importance[fi] += bobot * node.info_gain

    # Rekursi ke subtree kiri dan kanan dengan bobot dikurangi setengah
    hitung_feature_importance(node.left,  n_fitur, importance, bobot * 0.5)
    hitung_feature_importance(node.right, n_fitur, importance, bobot * 0.5)

    # Normalisasi agar semua importance berjumlah 1 (dilakukan di root saja)
    total = importance.sum()
    if total > 0:
        importance = importance / total

    return importance


# ─────────────────────────────────────────────────────────────────────────────
# 8. TAMPILKAN STRUKTUR POHON (TEKS)
# ─────────────────────────────────────────────────────────────────────────────

def tampilkan_pohon(
    node:          Node,
    feature_names: list,
    class_names:   list,
    indent:        str  = "",
    branch:        str  = "ROOT",
    batas_depth:   int  = 3,
    kedalaman:     int  = 0,
):
    """
    Menampilkan struktur pohon keputusan dalam format teks hierarkis.
    Hanya menampilkan hingga batas_depth level dari root.

    Parameter:
        node          : simpul yang sedang ditampilkan
        feature_names : daftar nama kolom fitur
        class_names   : daftar nama kelas target
        indent        : string indentasi untuk tampilan hierarki
        branch        : label cabang ('ROOT', 'KIRI', 'KANAN')
        batas_depth   : kedalaman maksimum yang ditampilkan
        kedalaman     : kedalaman simpul saat ini
    """
    if node is None or kedalaman > batas_depth:
        return

    if node.is_leaf():
        # Leaf node: tampilkan prediksi kelas
        print(f"{indent}[{branch}] DAUN -> Prediksi: {class_names[node.value]}")
    else:
        # Internal node: tampilkan fitur, threshold, dan IG
        nama = feature_names[node.feature_index]
        print(
            f"{indent}[{branch}] SPLIT: {nama} <= {node.threshold:.4f}"
            f"  (IG={node.info_gain:.4f})"
        )
        # Rekursi ke subtree kiri (nilai <= threshold)
        tampilkan_pohon(node.left,  feature_names, class_names,
                        indent + "  |  ", "KIRI",  batas_depth, kedalaman + 1)
        # Rekursi ke subtree kanan (nilai > threshold)
        tampilkan_pohon(node.right, feature_names, class_names,
                        indent + "  |  ", "KANAN", batas_depth, kedalaman + 1)
