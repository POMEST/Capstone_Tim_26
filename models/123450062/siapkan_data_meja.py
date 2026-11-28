"""Konversi sesi meja asli Tim 26 ke format yang dibaca ``src.protokol.data`` (tanpa mengubah data.py).

Sesi meja (``data/sesi-YYYYMMDD-HHMMSS/``) berbeda dari format simulasi protokol:

| Protokol (data.py)                | Sesi meja asli                                   |
|-----------------------------------|--------------------------------------------------|
| ``desain.csv`` (t_mulai/t_selesai)| ``label.csv`` (``t_mulai_host``/``t_selesai_host``) |
| ``imu.csv``: ``t``, imu=``kipas`` | ``t_host``/``t_esp_ms``, imu=``k``                |
| ``daya.csv``: ``t``, ``arus``     | ``t_host``/``t_esp_ms``, ``arus_mA``              |

Konversi yang dilakukan (tidak mengubah isi sinyal, hanya skema & sumbu waktu):
- Waktu memakai jam ESP32 (``t_esp_ms``, seragam & monoton) digeser ke jam host dengan satu offset
  median ``t_host - t_esp``; ``t_host`` per baris tidak dipakai karena berjitter (paket Wi-Fi tiba
  bergerombol). Lalu semua waktu dibuat relatif terhadap awal kondisi pertama.
- Baris bernilai tidak numerik (mis. paket rusak) dibuang.
- ``label.csv`` -> ``desain.csv`` dengan kolom kondisi, blok, ulangan_blok, penghalang, kipas,
  ulangan, t_mulai, t_selesai (detik relatif).

Pemakaian:
    python models/123450062/siapkan_data_meja.py [--sesi data/sesi-20261117-160506]
Keluaran: ``data/sesi_meja_t26/`` (tidak di-commit; ``data/*`` ada di .gitignore).
Lalu arahkan ``data/MANIFEST.json`` ke berkas di folder itu (lihat catatan.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AKAR = Path(__file__).resolve().parents[2]
KELUAR_BAWAAN = AKAR / "data" / "sesi_meja_t26"
KOLOM_IMU = ["ax", "ay", "az", "gx", "gy", "gz"]


def _angka(df: pd.DataFrame, kolom: list[str]) -> pd.DataFrame:
    for k in kolom:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=kolom).reset_index(drop=True)
    if len(df) != n0:
        print(f"  dibuang {n0 - len(df)} baris tidak numerik")
    return df


def _waktu_esp_ke_host(df: pd.DataFrame) -> np.ndarray:
    t_esp = df["t_esp_ms"].to_numpy(np.float64) / 1000.0
    offset = float(np.median(df["t_host"].to_numpy(np.float64) - t_esp))
    return t_esp + offset


def konversi(sesi: Path, keluar: Path) -> dict:
    label = pd.read_csv(sesi / "label.csv")
    t_ref = float(label["t_mulai_host"].min())

    # Pada label.csv kolom `blok` hanya 1..4 (taraf), sedangkan satuan pemisahan protokol adalah
    # 8 blok fisik = pasangan (blok, ulangan_blok). Beri nomor 1..8 menurut urutan waktu.
    label = label.sort_values("no").reset_index(drop=True)
    pasangan = list(zip(label["blok"], label["ulangan_blok"]))
    urutan: dict[tuple[int, int], int] = {}
    for p in pasangan:
        urutan.setdefault(p, len(urutan) + 1)
    blok_fisik = np.array([urutan[p] for p in pasangan], dtype=int)

    desain = pd.DataFrame(
        {
            "kondisi": label["no"].astype(int),
            "blok": blok_fisik,
            "ulangan_blok": label["ulangan_blok"].astype(int),
            "penghalang": label["penghalang"].astype(str).str.strip(),
            "kipas": label["kipas"].astype(int),
            "ulangan": label["ulangan"].astype(int),
            "t_mulai": (label["t_mulai_host"] - t_ref).round(4),
            "t_selesai": (label["t_selesai_host"] - t_ref).round(4),
        }
    ).sort_values("kondisi")

    print("membaca imu.csv ...")
    imu = pd.read_csv(sesi / "imu.csv", low_memory=False)
    imu = imu[imu["imu"].astype(str).str.strip() == "k"].copy()
    imu = _angka(imu, ["t_host", "t_esp_ms", *KOLOM_IMU])
    imu_out = imu[KOLOM_IMU].copy()
    imu_out.insert(0, "imu", "kipas")
    imu_out.insert(0, "t", np.round(_waktu_esp_ke_host(imu) - t_ref, 4))

    print("membaca daya.csv ...")
    daya = pd.read_csv(sesi / "daya.csv", low_memory=False)
    daya = daya[daya["kanal"].astype(str).str.strip() == "kipas"].copy()
    daya = _angka(daya, ["t_host", "t_esp_ms", "bus_mV", "arus_mA"])
    daya_out = pd.DataFrame(
        {
            "t": np.round(_waktu_esp_ke_host(daya) - t_ref, 4),
            "kanal": "kipas",
            "arus": daya["arus_mA"].to_numpy(),
            "tegangan": (daya["bus_mV"].to_numpy() / 1000.0).round(4),
        }
    )

    keluar.mkdir(parents=True, exist_ok=True)
    desain.to_csv(keluar / "desain.csv", index=False)
    imu_out.sort_values("t", kind="mergesort").to_csv(keluar / "imu.csv", index=False)
    daya_out.sort_values("t", kind="mergesort").to_csv(keluar / "daya.csv", index=False)

    info = {
        "tim": 26,
        "simulasi": False,
        "sumber_sesi": sesi.name,
        "dikonversi_oleh": "models/123450062/siapkan_data_meja.py",
        "jumlah_kondisi": int(len(desain)),
        "jumlah_blok": int(desain["blok"].nunique()),
        "baris_imu_kipas": int(len(imu_out)),
        "baris_daya_kipas": int(len(daya_out)),
        "laju_imu_efektif_hz": round(len(imu_out) / float(imu_out["t"].max() - imu_out["t"].min()), 1),
        "laju_daya_efektif_hz": round(len(daya_out) / float(daya_out["t"].max() - daya_out["t"].min()), 1),
        "catatan": "Waktu = jam ESP32 + offset median ke jam host, relatif terhadap awal kondisi 1.",
    }
    (keluar / "manifest.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesi", default=None, help="folder sesi meja; bawaan: sesi-* terbaru di data/")
    ap.add_argument("--keluar", default=str(KELUAR_BAWAAN))
    a = ap.parse_args()

    if a.sesi:
        sesi = Path(a.sesi)
        sesi = sesi if sesi.is_absolute() else AKAR / sesi
    else:
        kandidat = sorted((AKAR / "data").glob("sesi-*"))
        if not kandidat:
            raise SystemExit("Tidak ada data/sesi-* ; salin sesi meja ke data/ terlebih dahulu.")
        sesi = kandidat[-1]
    print(f"sesi meja: {sesi}")
    info = konversi(sesi, Path(a.keluar))
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"-> {a.keluar}")


if __name__ == "__main__":
    main()
