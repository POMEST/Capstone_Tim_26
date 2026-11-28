"""Generator data simulasi dummy untuk Tim 26 (Penghalang Aliran Udara Kipas).

Meniru struktur keluaran sesi ``meja.py --simulasi jalankan ../rencana/t26-penghalang-kipas.yaml``
sebelum data meja asli tersedia (~11 Nov):
- 8 blok (4 taraf penghalang x ulangan_blok 2), urutan diacak dengan seed 1026.
- 12 kondisi per blok (3 taraf kipas [120, 190, 255] x ulangan 4), total 96 kondisi.
- Durasi 20 detik per kondisi + jeda 2 detik antar-kondisi.
- ``desain.csv`` : metadata kondisi, blok, faktor, dan selang waktu [t_mulai, t_selesai].
- ``imu.csv``    : getaran IMU MPU6050 (``imu = kipas`` pada 400 Hz, plus ``imu = meja`` 50 Hz).
- ``daya.csv``   : arus & tegangan (``kanal = kipas`` pada 200 Hz, plus ``kanal = utama`` 50 Hz).
- ``manifest.json`` : ringkasan sesi simulasi.

Jalankan:
    python -m src.protokol.buat_simulasi_dummy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

AKAR = Path(__file__).resolve().parents[2]
FOLDER_KELUAR = AKAR / "data" / "sesi_simulasi"

SEED_SIMULASI = 1026
ULANGAN = 4
ULANGAN_BLOK = 2
DURASI_KONDISI_S = 20.0
JEDA_S = 2.0
FS_IMU_KIPAS = 400
FS_IMU_MEJA = 50
FS_DAYA_KIPAS = 200
FS_DAYA_UTAMA = 50

TARAF_KIPAS = [120, 190, 255]
TARAF_PENGHALANG = [
    "tanpa penghalang",
    "kertas menutup 25%",
    "kertas menutup 50%",
    "kertas menutup 75%",
]

# Parameter fisika sintetis per tingkat tutupan (0%, 25%, 50%, 75%)
# Tutupan lebih besar meningkatkan turbulensi/getaran harmonik & mengubah beban arus kipas.
PROFIL_PENGHALANG = {
    "tanpa penghalang": {"amp_vib": 0.45, "turbulensi": 0.12, "geser_arus_ma": 0.0, "riak_arus": 6.0},
    "kertas menutup 25%": {"amp_vib": 0.65, "turbulensi": 0.25, "geser_arus_ma": -14.0, "riak_arus": 10.0},
    "kertas menutup 50%": {"amp_vib": 0.92, "turbulensi": 0.44, "geser_arus_ma": -30.0, "riak_arus": 16.0},
    "kertas menutup 75%": {"amp_vib": 1.28, "turbulensi": 0.72, "geser_arus_ma": -48.0, "riak_arus": 24.0},
}


def buat_sesi(folder: Path = FOLDER_KELUAR, seed: int = SEED_SIMULASI) -> Path:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)

    # 8 blok: 4 taraf penghalang x 2 ulangan_blok, urutan blok diacak dengan seed 1026
    daftar_blok = []
    for u_blok in range(1, ULANGAN_BLOK + 1):
        for taraf in TARAF_PENGHALANG:
            daftar_blok.append((u_blok, taraf))
    urut_blok = rng.permutation(len(daftar_blok))
    daftar_blok = [daftar_blok[i] for i in urut_blok]

    baris_desain = []
    potongan_imu = []
    potongan_daya = []

    t_kursor = 0.0
    kondisi_id = 0

    for idx_blok, (u_blok, penghalang) in enumerate(daftar_blok, start=1):
        # Efek acak per blok (pemasangan ulang kertas / penjepit)
        bias_blok_vib = float(rng.normal(0.0, 0.04))
        bias_blok_arus = float(rng.normal(0.0, 2.5))
        profil = PROFIL_PENGHALANG[penghalang]

        # 12 kondisi per blok: 3 taraf kipas x 4 ulangan, urutan diacak
        kombinasi = [(k, u) for u in range(1, ULANGAN + 1) for k in TARAF_KIPAS]
        urut_k = rng.permutation(len(kombinasi))
        kombinasi = [kombinasi[i] for i in urut_k]

        for kipas, ulangan in kombinasi:
            kondisi_id += 1
            t_mulai = t_kursor
            t_selesai = t_mulai + DURASI_KONDISI_S

            baris_desain.append(
                {
                    "kondisi": kondisi_id,
                    "blok": idx_blok,
                    "ulangan_blok": u_blok,
                    "penghalang": penghalang,
                    "kipas": kipas,
                    "ulangan": ulangan,
                    "t_mulai": round(t_mulai, 4),
                    "t_selesai": round(t_selesai, 4),
                }
            )

            # Frekuensi dasar putaran kipas & arus dasar berdasarkan PWM kipas (120, 190, 255)
            f_kipas = 20.0 + (kipas / 255.0) * 38.0  # ~38 Hz .. 58 Hz
            arus_dasar = 110.0 + (kipas / 255.0) * 190.0  # ~200 .. 300 mA

            # --- IMU kipas (400 Hz) ---
            n_imu = int(DURASI_KONDISI_S * FS_IMU_KIPAS)
            dt_imu = 1.0 / FS_IMU_KIPAS
            t_rel_imu = np.arange(n_imu, dtype=np.float64) * dt_imu
            t_abs_imu = t_mulai + t_rel_imu

            # Peralihan pada 1 detik pertama (spin-up kipas)
            transien_imu = np.exp(-t_rel_imu / 0.35)
            amplop_speed = 1.0 - 0.65 * transien_imu
            amp = (profil["amp_vib"] + bias_blok_vib) * (0.6 + 0.5 * kipas / 255.0)
            turb = profil["turbulensi"]

            fase = rng.uniform(0, 2 * np.pi)
            ax = (
                amp * amplop_speed * np.sin(2 * np.pi * f_kipas * t_rel_imu + fase)
                + 0.45 * turb * np.sin(2 * np.pi * 2.0 * f_kipas * t_rel_imu + 0.4)
                + 1.1 * transien_imu * np.sin(2 * np.pi * 12.0 * t_rel_imu)
                + rng.normal(0.0, 0.08 + 0.09 * turb, size=n_imu)
            )
            ay = (
                0.85 * amp * amplop_speed * np.cos(2 * np.pi * f_kipas * t_rel_imu + fase)
                + 0.35 * turb * np.sin(2 * np.pi * 0.5 * f_kipas * t_rel_imu + 1.1)
                + 0.9 * transien_imu
                + rng.normal(0.0, 0.08 + 0.09 * turb, size=n_imu)
            )
            az = (
                9.81
                + 0.45 * amp * amplop_speed * np.sin(2 * np.pi * 3.0 * f_kipas * t_rel_imu + 0.7)
                + 0.55 * turb * np.cos(2 * np.pi * 1.5 * f_kipas * t_rel_imu)
                + rng.normal(0.0, 0.09 + 0.11 * turb, size=n_imu)
            )
            gx = 12.0 * amp * np.sin(2 * np.pi * f_kipas * t_rel_imu + 0.2) + rng.normal(0.0, 1.8 + 2.5 * turb, size=n_imu)
            gy = 10.0 * amp * np.cos(2 * np.pi * f_kipas * t_rel_imu + 0.9) + rng.normal(0.0, 1.8 + 2.5 * turb, size=n_imu)
            gz = 8.0 * turb * np.sin(2 * np.pi * 2.0 * f_kipas * t_rel_imu) + rng.normal(0.0, 1.5 + 2.0 * turb, size=n_imu)

            potongan_imu.append(
                pd.DataFrame(
                    {
                        "t": np.round(t_abs_imu, 6),
                        "imu": "kipas",
                        "ax": np.round(ax, 5),
                        "ay": np.round(ay, 5),
                        "az": np.round(az, 5),
                        "gx": np.round(gx, 5),
                        "gy": np.round(gy, 5),
                        "gz": np.round(gz, 5),
                    }
                )
            )

            # Baris distraktor imu = meja (50 Hz) untuk menguji filter baris imu == "kipas"
            n_meja = int(DURASI_KONDISI_S * FS_IMU_MEJA)
            t_abs_meja = t_mulai + np.arange(n_meja, dtype=np.float64) / FS_IMU_MEJA
            potongan_imu.append(
                pd.DataFrame(
                    {
                        "t": np.round(t_abs_meja, 6),
                        "imu": "meja",
                        "ax": np.round(rng.normal(0.0, 0.02, size=n_meja), 5),
                        "ay": np.round(rng.normal(0.0, 0.02, size=n_meja), 5),
                        "az": np.round(9.81 + rng.normal(0.0, 0.02, size=n_meja), 5),
                        "gx": np.round(rng.normal(0.0, 0.2, size=n_meja), 5),
                        "gy": np.round(rng.normal(0.0, 0.2, size=n_meja), 5),
                        "gz": np.round(rng.normal(0.0, 0.2, size=n_meja), 5),
                    }
                )
            )

            # --- Arus daya.csv kanal = kipas (200 Hz) ---
            n_daya = int(DURASI_KONDISI_S * FS_DAYA_KIPAS)
            dt_daya = 1.0 / FS_DAYA_KIPAS
            t_rel_daya = np.arange(n_daya, dtype=np.float64) * dt_daya
            t_abs_daya = t_mulai + t_rel_daya

            # Lonjakan arus inrush pada 1 detik pertama
            transien_arus = 85.0 * np.exp(-t_rel_daya / 0.28)
            arus_kipas = (
                arus_dasar
                + profil["geser_arus_ma"]
                + bias_blok_arus
                + transien_arus
                + profil["riak_arus"] * np.sin(2 * np.pi * f_kipas * t_rel_daya + 0.3)
                + rng.normal(0.0, 2.5 + 0.15 * profil["riak_arus"], size=n_daya)
            )
            tegangan_kipas = 5.0 - 0.0004 * arus_kipas + rng.normal(0.0, 0.01, size=n_daya)

            potongan_daya.append(
                pd.DataFrame(
                    {
                        "t": np.round(t_abs_daya, 6),
                        "kanal": "kipas",
                        "arus": np.round(arus_kipas, 4),
                        "tegangan": np.round(tegangan_kipas, 4),
                    }
                )
            )

            # Baris distraktor kanal = utama (50 Hz) untuk menguji filter kanal == "kipas"
            n_utama = int(DURASI_KONDISI_S * FS_DAYA_UTAMA)
            t_abs_utama = t_mulai + np.arange(n_utama, dtype=np.float64) / FS_DAYA_UTAMA
            potongan_daya.append(
                pd.DataFrame(
                    {
                        "t": np.round(t_abs_utama, 6),
                        "kanal": "utama",
                        "arus": np.round(450.0 + rng.normal(0.0, 8.0, size=n_utama), 4),
                        "tegangan": np.round(5.02 + rng.normal(0.0, 0.01, size=n_utama), 4),
                    }
                )
            )

            t_kursor = t_selesai + JEDA_S

    df_desain = pd.DataFrame(baris_desain)
    df_imu = pd.concat(potongan_imu, ignore_index=True).sort_values(["t", "imu"], kind="mergesort")
    df_daya = pd.concat(potongan_daya, ignore_index=True).sort_values(["t", "kanal"], kind="mergesort")

    df_desain.to_csv(folder / "desain.csv", index=False)
    df_imu.to_csv(folder / "imu.csv", index=False)
    df_daya.to_csv(folder / "daya.csv", index=False)

    info = {
        "tim": 26,
        "judul": "Deteksi penghalang aliran udara kipas",
        "simulasi": True,
        "seed": seed,
        "jumlah_blok": len(daftar_blok),
        "jumlah_kondisi": len(df_desain),
        "durasi_kondisi_detik": DURASI_KONDISI_S,
        "jeda_detik": JEDA_S,
        "frekuensi_imu_kipas_hz": FS_IMU_KIPAS,
        "frekuensi_daya_kipas_hz": FS_DAYA_KIPAS,
        "kelas": TARAF_PENGHALANG,
    }
    (folder / "manifest.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return folder


def main() -> None:
    folder = buat_sesi()
    print(f"Data simulasi dummy ditulis ke {folder}")


if __name__ == "__main__":
    main()
