"""Verifikasi otomatis kontrak protokol Tim 26 pada ``src/protokol/data.py``.

Memeriksa:
1. ``muat()`` mengembalikan (X_train, y_train, X_val, y_val, X_test, y_test) dari folder sesi simulasi.
2. Tidak ada satuan pemisahan (``blok``) yang muncul di latih/validasi dan uji sekaligus.
3. Tidak ada ``kondisi`` yang bocor antar-split.
4. Aturan buang 1 detik pertama, filter ``imu = kipas`` & ``kanal = kipas``, dan
   normalisasi yang dihitung dari data latih saja benar-benar berlaku.

Jalankan:
    python -m src.protokol.uji_protokol
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.protokol import data


def uji_muat_dan_kebocoran() -> None:
    X_tr, y_tr, X_va, y_va, X_te, y_te = data.muat()
    assert data.TUGAS == "klasifikasi", f"TUGAS seharusnya 'klasifikasi', dapat {data.TUGAS!r}"
    assert X_tr.ndim == 3 and X_tr.shape[1:] == (400, 7), f"Bentuk X_tr tidak sesuai: {X_tr.shape}"
    assert X_va.ndim == 3 and X_va.shape[1:] == (400, 7), f"Bentuk X_va tidak sesuai: {X_va.shape}"
    assert X_te.ndim == 3 and X_te.shape[1:] == (400, 7), f"Bentuk X_te tidak sesuai: {X_te.shape}"
    assert X_tr.dtype == np.float32 and X_va.dtype == np.float32 and X_te.dtype == np.float32
    assert y_tr.dtype == np.int64 and y_va.dtype == np.int64 and y_te.dtype == np.int64
    assert len(X_tr) + len(X_va) + len(X_te) == 96 * 19

    for nama, y_split in (("y_tr", y_tr), ("y_va", y_va), ("y_te", y_te)):
        kelas_ada = set(np.unique(y_split).tolist())
        assert kelas_ada == {0, 1, 2, 3}, f"{nama} tidak memuat seluruh 4 kelas: {kelas_ada}"

    _, _, _, _, _, _, meta = data.muat_dengan_meta()
    blok_tr = set(meta["blok_train"])
    blok_va = set(meta["blok_val"])
    blok_te = set(meta["blok_test"])
    assert (blok_tr & blok_te) == set(), f"Blok muncul di latih dan uji: {blok_tr & blok_te}"
    assert (blok_va & blok_te) == set(), f"Blok muncul di validasi dan uji: {blok_va & blok_te}"
    assert len(blok_te) == 4 and len(blok_tr | blok_va) == 4

    k_tr = set(meta["kondisi_train"])
    k_va = set(meta["kondisi_val"])
    k_te = set(meta["kondisi_test"])
    assert (k_tr & k_va) == set() and (k_tr & k_te) == set() and (k_va & k_te) == set()

    # Pastikan fungsi pemeriksaan kebocoran menolak irisan blok
    gagal_seperti_harapan = False
    try:
        data.periksa_tanpa_kebocoran(np.array([1, 2]), np.array([1]), np.array([2, 3]))
    except AssertionError:
        gagal_seperti_harapan = True
    assert gagal_seperti_harapan, "periksa_tanpa_kebocoran gagal mendeteksi irisan blok!"

    print(
        f"[OK] muat() & cek kebocoran blok lulus: "
        f"train={X_tr.shape} blok={sorted(blok_tr)}, "
        f"val={X_va.shape} blok={sorted(blok_va)}, "
        f"test={X_te.shape} blok={sorted(blok_te)}"
    )


def uji_invariansi_aturan_protokol() -> None:
    """Uji buang 1 detik awal, filter kipas, dan normalisasi latih-saja pada salinan sesi."""
    X_tr_ref, _, X_va_ref, _, _, _, meta_ref = data.muat_dengan_meta()
    assert np.allclose(X_tr_ref.mean(axis=(0, 1)), 0.0, atol=1e-5)
    assert np.allclose(X_tr_ref.std(axis=(0, 1)), 1.0, atol=1e-4)

    with tempfile.TemporaryDirectory() as tmp:
        folder_tmp = Path(tmp) / "sesi_uji"
        shutil.copytree(data.FOLDER_SESI_BAWAAN, folder_tmp)

        df_des = pd.read_csv(folder_tmp / "desain.csv")
        df_imu = pd.read_csv(folder_tmp / "imu.csv")
        df_daya = pd.read_csv(folder_tmp / "daya.csv")

        # 1. Rusak sampel 1 detik pertama setiap kondisi -> hasil muat() harus tetap identik
        for row in df_des.itertuples(index=False):
            m_imu = (df_imu["t"] >= row.t_mulai) & (df_imu["t"] < row.t_mulai + 1.0)
            df_imu.loc[m_imu, ["ax", "ay", "az", "gx", "gy", "gz"]] += 9999.0
            m_daya = (df_daya["t"] >= row.t_mulai) & (df_daya["t"] < row.t_mulai + 1.0)
            df_daya.loc[m_daya, "arus"] += 9999.0

        # 2. Rusak baris distraktor imu != 'kipas' dan kanal != 'kipas' -> harus terabaikan
        df_imu.loc[df_imu["imu"] != "kipas", ["ax", "ay", "az"]] = -8888.0
        df_daya.loc[df_daya["kanal"] != "kipas", "arus"] = -8888.0

        # 3. Ubah skala besar pada blok uji -> statistik normalisasi, X_train, X_val tidak boleh berubah
        blok_uji = set(meta_ref["blok_test"])
        for row in df_des[df_des["blok"].isin(blok_uji)].itertuples(index=False):
            m_imu = (df_imu["t"] >= row.t_mulai) & (df_imu["t"] <= row.t_selesai)
            df_imu.loc[m_imu, ["ax", "ay", "az", "gx", "gy", "gz"]] *= 50.0
            m_daya = (df_daya["t"] >= row.t_mulai) & (df_daya["t"] <= row.t_selesai)
            df_daya.loc[m_daya, "arus"] += 5000.0

        df_imu.to_csv(folder_tmp / "imu.csv", index=False)
        df_daya.to_csv(folder_tmp / "daya.csv", index=False)

        X_tr_new, _, X_va_new, _, X_te_new, _, meta_new = data.muat_dengan_meta(folder_tmp)

        assert np.allclose(meta_ref["norm_mean"], meta_new["norm_mean"], atol=1e-6), "Normalisasi bocor dari data uji!"
        assert np.allclose(meta_ref["norm_std"], meta_new["norm_std"], atol=1e-6), "Normalisasi std bocor dari data uji!"
        assert np.array_equal(X_tr_ref, X_tr_new), "X_train berubah akibat 1 detik awal / non-kipas / data uji!"
        assert np.array_equal(X_va_ref, X_va_new), "X_val berubah akibat 1 detik awal / non-kipas / data uji!"
        assert X_te_new.shape[0] == 4 * 12 * 19

    print("[OK] Invariansi buang-1-detik-awal, filter kipas, dan normalisasi latih-saja lulus!")


def main() -> None:
    uji_muat_dan_kebocoran()
    uji_invariansi_aturan_protokol()


if __name__ == "__main__":
    main()
