"""Pemuat dan pemotong jendela data Tim 26 (Deteksi penghalang aliran udara kipas).

Kontrak protokol tim (dibekukan setelah Sprint 0 dengan persetujuan pembimbing):
    TUGAS = "klasifikasi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)

Aturan yang diterapkan:
1. Masukan berasal dari ``imu.csv`` baris ``imu == "kipas"`` (400 Hz: ax, ay, az, gx, gy, gz)
   dan ``daya.csv`` baris ``kanal == "kipas"`` (200 Hz: arus), dipotong menjadi jendela 1 detik
   pada grid 400 Hz (``arus`` diinterpolasi linear ke grid 400 Hz -> bentuk ``[N, 400, 7]``).
2. Membuang 1 detik pertama setiap kondisi (fase peralihan alat).
3. Data uji dipisah per blok (seluruh kondisi & jendela dari blok uji tidak pernah muncul
   pada data latih maupun validasi).
4. Split latih/validasi memakai ``src.common.split.indeks`` (seed 2026) pada level kondisi
   di dalam blok latih/validasi agar tidak ada kebocoran jendela antarkondisi yang sama.
5. Normalisasi z-score per kanal dihitung hanya dari ``X_train``, lalu diterapkan ke
   ``X_train``, ``X_val``, dan ``X_test``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common import manifest, split

TUGAS = "klasifikasi"

FOLDER_SESI_BAWAAN = manifest.AKAR / "data" / "sesi_simulasi"

KELAS_PENGHALANG = (
    "tanpa penghalang",
    "kertas menutup 25%",
    "kertas menutup 50%",
    "kertas menutup 75%",
)
PETA_LABEL = {nama: idx for idx, nama in enumerate(KELAS_PENGHALANG)}

BUANG_AWAL_DETIK = 1.0
JENDELA_DETIK = 1.0
FS_IMU = 400
FS_DAYA = 200
PORSI_VAL = 0.2

KOLOM_IMU = ("ax", "ay", "az", "gx", "gy", "gz")
NAMA_FITUR = (*KOLOM_IMU, "arus")


def periksa_tanpa_kebocoran(
    blok_train: np.ndarray,
    blok_val: np.ndarray,
    blok_test: np.ndarray,
    kondisi_train: np.ndarray | None = None,
    kondisi_val: np.ndarray | None = None,
    kondisi_test: np.ndarray | None = None,
) -> None:
    """Pastikan tidak ada satuan pemisahan (blok) yang muncul di latih/val dan uji sekaligus."""
    set_tr_blok = set(np.asarray(blok_train).tolist())
    set_va_blok = set(np.asarray(blok_val).tolist())
    set_te_blok = set(np.asarray(blok_test).tolist())

    irisan_tr_te = set_tr_blok & set_te_blok
    if irisan_tr_te:
        raise AssertionError(f"Kebocoran blok antara latih dan uji: {sorted(irisan_tr_te)}")

    irisan_va_te = set_va_blok & set_te_blok
    if irisan_va_te:
        raise AssertionError(f"Kebocoran blok antara validasi dan uji: {sorted(irisan_va_te)}")

    if kondisi_train is not None and kondisi_val is not None:
        set_tr_k = set(np.asarray(kondisi_train).tolist())
        set_va_k = set(np.asarray(kondisi_val).tolist())
        irisan_tr_va = set_tr_k & set_va_k
        if irisan_tr_va:
            raise AssertionError(f"Kebocoran kondisi antara latih dan validasi: {sorted(irisan_tr_va)}")
        if kondisi_test is not None:
            set_te_k = set(np.asarray(kondisi_test).tolist())
            irisan_k_te = (set_tr_k | set_va_k) & set_te_k
            if irisan_k_te:
                raise AssertionError(f"Kebocoran kondisi dengan data uji: {sorted(irisan_k_te)}")


def _resolusi_folder_sesi(folder_sesi: Path | str | None) -> Path:
    if folder_sesi is not None:
        return Path(folder_sesi)
    if manifest.MANIFEST.exists():
        m = manifest.baca()
        for rel in m.get("berkas", []):
            p = manifest.AKAR / rel
            if p.name == "desain.csv":
                return p.parent
    return FOLDER_SESI_BAWAAN


def _potong_jendela(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Baca desain.csv, imu.csv, daya.csv lalu potong menjadi jendela 1 detik."""
    for berkas in ("desain.csv", "imu.csv", "daya.csv"):
        if not (folder / berkas).exists():
            raise FileNotFoundError(
                f"Berkas sesi tidak ditemukan: {folder / berkas}. "
                "Jalankan `python -m src.protokol.buat_simulasi_dummy` untuk membuat data simulasi."
            )

    df_desain = pd.read_csv(folder / "desain.csv")
    df_desain["penghalang"] = df_desain["penghalang"].astype(str).str.strip()
    tidak_dikenal = set(df_desain["penghalang"]) - set(PETA_LABEL)
    if tidak_dikenal:
        raise ValueError(f"Label penghalang tidak dikenal di desain.csv: {sorted(tidak_dikenal)}")

    df_imu = pd.read_csv(folder / "imu.csv")
    df_imu = df_imu[df_imu["imu"].astype(str).str.strip() == "kipas"].sort_values("t")
    if df_imu.empty:
        raise ValueError("Tidak ada baris `imu == 'kipas'` pada imu.csv")

    df_daya = pd.read_csv(folder / "daya.csv")
    df_daya = df_daya[df_daya["kanal"].astype(str).str.strip() == "kipas"].sort_values("t")
    if df_daya.empty:
        raise ValueError("Tidak ada baris `kanal == 'kipas'` pada daya.csv")

    t_imu = df_imu["t"].to_numpy(dtype=np.float64)
    sig_imu = df_imu[list(KOLOM_IMU)].to_numpy(dtype=np.float64)

    t_daya = df_daya["t"].to_numpy(dtype=np.float64)
    sig_arus = df_daya["arus"].to_numpy(dtype=np.float64)

    daftar_x: list[np.ndarray] = []
    daftar_y: list[int] = []
    daftar_blok: list[int] = []
    daftar_kondisi: list[int] = []

    offset_grid = np.arange(FS_IMU, dtype=np.float64) / FS_IMU

    for baris in df_desain.itertuples(index=False):
        t_mulai = float(baris.t_mulai)
        t_selesai = float(baris.t_selesai)
        t_efektif_awal = t_mulai + BUANG_AWAL_DETIK
        durasi_efektif = t_selesai - t_efektif_awal
        n_jendela = int(np.floor((durasi_efektif + 1e-9) / JENDELA_DETIK))
        if n_jendela <= 0:
            continue

        # Batasi irisan pada kondisi ini saja (setelah membuang 1 detik awal)
        i0_imu = int(np.searchsorted(t_imu, t_efektif_awal - 1e-6, side="left"))
        i1_imu = int(np.searchsorted(t_imu, t_selesai + 1e-6, side="right"))
        t_imu_k = t_imu[i0_imu:i1_imu]
        s_imu_k = sig_imu[i0_imu:i1_imu]

        i0_daya = int(np.searchsorted(t_daya, t_efektif_awal - 1e-6, side="left"))
        i1_daya = int(np.searchsorted(t_daya, t_selesai + 1e-6, side="right"))
        t_daya_k = t_daya[i0_daya:i1_daya]
        s_arus_k = sig_arus[i0_daya:i1_daya]

        label = PETA_LABEL[str(baris.penghalang)]
        blok_id = int(baris.blok)
        kondisi_id = int(baris.kondisi)

        for w in range(n_jendela):
            w_awal = t_efektif_awal + w * JENDELA_DETIK
            w_akhir = w_awal + JENDELA_DETIK

            # Cek cakupan sampel minimum di dalam jendela 1 detik
            c_imu_0 = int(np.searchsorted(t_imu_k, w_awal - 1e-6, side="left"))
            c_imu_1 = int(np.searchsorted(t_imu_k, w_akhir - 1e-6, side="left"))
            c_daya_0 = int(np.searchsorted(t_daya_k, w_awal - 1e-6, side="left"))
            c_daya_1 = int(np.searchsorted(t_daya_k, w_akhir - 1e-6, side="left"))
            if (c_imu_1 - c_imu_0) < FS_IMU // 2 or (c_daya_1 - c_daya_0) < FS_DAYA // 2:
                continue

            t_grid = w_awal + offset_grid
            jendela = np.empty((FS_IMU, len(NAMA_FITUR)), dtype=np.float32)
            for idx_kol in range(len(KOLOM_IMU)):
                jendela[:, idx_kol] = np.interp(t_grid, t_imu_k, s_imu_k[:, idx_kol])
            jendela[:, len(KOLOM_IMU)] = np.interp(t_grid, t_daya_k, s_arus_k)

            daftar_x.append(jendela)
            daftar_y.append(label)
            daftar_blok.append(blok_id)
            daftar_kondisi.append(kondisi_id)

    if not daftar_x:
        raise ValueError(f"Tidak ada jendela 1 detik yang valid dihasilkan dari {folder}")

    return (
        np.stack(daftar_x, axis=0).astype(np.float32),
        np.asarray(daftar_y, dtype=np.int64),
        np.asarray(daftar_blok, dtype=np.int64),
        np.asarray(daftar_kondisi, dtype=np.int64),
    )


def _bagi_blok_uji(
    win_y: np.ndarray,
    win_blok: np.ndarray,
) -> tuple[set[int], set[int]]:
    """Pisahkan blok menjadi blok latih/validasi dan blok uji secara berimbang per kelas."""
    blok_latih_val: set[int] = set()
    blok_uji: set[int] = set()

    for kelas in sorted(np.unique(win_y).tolist()):
        blok_kelas = sorted(np.unique(win_blok[win_y == kelas]).tolist())
        if len(blok_kelas) < 2:
            raise ValueError(
                f"Kelas {kelas} hanya memiliki {len(blok_kelas)} blok; butuh minimal 2 blok untuk split per blok."
            )
        n_uji = max(1, len(blok_kelas) // 2)
        blok_latih_val.update(blok_kelas[:-n_uji])
        blok_uji.update(blok_kelas[-n_uji:])

    if blok_latih_val & blok_uji:
        raise AssertionError("Satu blok terdeteksi memiliki lebih dari satu kelas atau tumpang tindih.")
    return blok_latih_val, blok_uji


def _normalisasi_latih_saja(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hitung mean dan std per kanal hanya dari X_train, lalu terapkan ke semua split."""
    X_tr64 = X_tr.astype(np.float64)
    mean = X_tr64.mean(axis=(0, 1), keepdims=True)
    std = X_tr64.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    tr_norm = ((X_tr64 - mean) / std).astype(np.float32)
    va_norm = ((X_va.astype(np.float64) - mean) / std).astype(np.float32)
    te_norm = ((X_te.astype(np.float64) - mean) / std).astype(np.float32)
    return tr_norm, va_norm, te_norm, mean.astype(np.float32), std.astype(np.float32)


def muat_dengan_meta(
    folder_sesi: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Memuat data terpotong, tersplit per blok, ternormalisasi latih-saja, beserta metadata audit."""
    folder = _resolusi_folder_sesi(folder_sesi)
    X_all, y_all, win_blok, win_kondisi = _potong_jendela(folder)

    blok_lv, blok_te = _bagi_blok_uji(y_all, win_blok)

    # Kumpulkan kondisi unik di blok latih/validasi, urutkan deterministik per (kelas, kondisi)
    mask_lv = np.isin(win_blok, sorted(blok_lv))
    pasang_unik = sorted(
        {(int(y_all[i]), int(win_kondisi[i])) for i in np.flatnonzero(mask_lv)}
    )
    kondisi_lv = np.asarray([k for _, k in pasang_unik], dtype=np.int64)

    idx_tr_k, idx_va_k = split.indeks(len(kondisi_lv), porsi_val=PORSI_VAL, seed=split.SEED_SPLIT)
    kondisi_tr = set(kondisi_lv[idx_tr_k].tolist())
    kondisi_va = set(kondisi_lv[idx_va_k].tolist())

    mask_tr = np.isin(win_kondisi, sorted(kondisi_tr))
    mask_va = np.isin(win_kondisi, sorted(kondisi_va))
    mask_te = np.isin(win_blok, sorted(blok_te))

    periksa_tanpa_kebocoran(
        win_blok[mask_tr],
        win_blok[mask_va],
        win_blok[mask_te],
        win_kondisi[mask_tr],
        win_kondisi[mask_va],
        win_kondisi[mask_te],
    )

    X_tr, X_va, X_te, mean, std = _normalisasi_latih_saja(X_all[mask_tr], X_all[mask_va], X_all[mask_te])
    y_tr, y_va, y_te = y_all[mask_tr], y_all[mask_va], y_all[mask_te]

    meta: dict[str, Any] = {
        "folder_sesi": str(folder),
        "fitur": NAMA_FITUR,
        "kelas": KELAS_PENGHALANG,
        "blok_train": sorted(np.unique(win_blok[mask_tr]).tolist()),
        "blok_val": sorted(np.unique(win_blok[mask_va]).tolist()),
        "blok_test": sorted(np.unique(win_blok[mask_te]).tolist()),
        "kondisi_train": sorted(kondisi_tr),
        "kondisi_val": sorted(kondisi_va),
        "kondisi_test": sorted(np.unique(win_kondisi[mask_te]).tolist()),
        "norm_mean": mean,
        "norm_std": std,
    }
    return X_tr, y_tr, X_va, y_va, X_te, y_te, meta


def muat(
    folder_sesi: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mengembalikan (X_train, y_train, X_val, y_val, X_test, y_test)."""
    X_tr, y_tr, X_va, y_va, X_te, y_te, _ = muat_dengan_meta(folder_sesi=folder_sesi)
    return X_tr, y_tr, X_va, y_va, X_te, y_te
