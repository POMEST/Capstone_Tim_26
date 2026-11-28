"""Pengujian end-to-end models/123450062/run.py pada data simulasi Tim 26.

Skrip ini TIDAK mengubah ``src/protokol/data.py`` maupun ``data/MANIFEST.json``
milik Penjaga Data & Evaluasi. Data simulasi dibangkitkan di folder sementara
mengikuti spesifikasi ``rencana/t26-penghalang-kipas.yaml`` dan ``tim-26.md``:
- 96 kondisi (8 blok x 12 kondisi): 4 kelas penghalang (0%, 25%, 50%, 75%)
  x 2 ulangan blok, masing-masing 3 taraf PWM kipas (120, 190, 255) x 4 ulangan.
- Masukan: 6 kanal getaran IMU (400 Hz) + 1 kanal arus kipas (200 Hz -> 400 Hz),
  jendela 1 detik (400 sampel), 1 detik awal tiap kondisi dibuang (19 jendela/kondisi).
- Split per blok: tidak ada kebocoran antar-blok antara data latih dan validasi/uji.

Pemakaian:
    python models/123450062/uji_simulasi.py
    python models/123450062/uji_simulasi.py --seed 1 --final
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
DIREKTORI_MODEL = Path(__file__).resolve().parent
sys.path.insert(0, str(AKAR))
sys.path.insert(0, str(DIREKTORI_MODEL))

from model import KipasCNN1D, ke_kanal_pertama  # noqa: E402

TARAF_PWM = (120, 190, 255)
JUMLAH_KELAS = 4          # 0%, 25%, 50%, 75%
FS_IMU = 400              # Hz (jendela 1 s = 400 titik)
FS_ARUS = 200             # Hz (diinterpolasi ke 400 titik)
JENDELA_PER_KONDISI = 19  # 20 s rekam - 1 s transien awal


def buat_data_simulasi(
    seed: int = 1026,
    jendela_per_kondisi: int = JENDELA_PER_KONDISI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bangkitkan data simulasi realistis getaran IMU (6 kanal) + arus kipas (1 kanal)."""
    rng = np.random.default_rng(seed)
    t_imu = np.linspace(0.0, 1.0, FS_IMU, endpoint=False, dtype=np.float32)
    t_arus = np.linspace(0.0, 1.0, FS_ARUS, endpoint=False, dtype=np.float32)

    X_tr, y_tr = [], []
    X_va, y_va = [], []
    X_te, y_te = [], []

    # 2 ulangan blok x 4 kelas = 8 blok
    for ulangan_blok in range(2):
        # Pergeseran kecil antar-blok (simulasi pemasangan ulang kertas/kisi)
        geser_blok = rng.normal(0.0, 0.03, size=7).astype(np.float32)
        for kelas in range(JUMLAH_KELAS):
            fraksi_tutup = kelas * 0.25  # 0.00, 0.25, 0.50, 0.75
            for pwm in TARAF_PWM:
                f_kipas = 22.0 + (pwm - 120.0) * 0.18  # ~22 Hz .. 46.3 Hz
                amp_dasar = 0.4 + (pwm / 255.0) * 0.6
                arus_dasar = 0.18 + (pwm / 255.0) * 0.35

                for ulangan in range(4):
                    for _ in range(jendela_per_kondisi):
                        fasa = rng.uniform(0.0, 2.0 * np.pi)
                        # 6 kanal IMU (ax, ay, az, gx, gy, gz):
                        # penghalang menambah turbulensi pita-lebar & harmonisa 2x/3x
                        imu = np.zeros((6, FS_IMU), dtype=np.float32)
                        for ch in range(6):
                            skala_ch = 1.0 if ch < 3 else 0.65
                            f_harm = f_kipas * (1.0 + 0.12 * fraksi_tutup)
                            sinyal = (
                                skala_ch * amp_dasar * (1.0 + 0.45 * fraksi_tutup)
                                * np.sin(2.0 * np.pi * f_harm * t_imu + fasa + ch * 0.7)
                                + skala_ch * (0.15 + 0.55 * fraksi_tutup)
                                * np.sin(2.0 * np.pi * (2.0 * f_harm) * t_imu + 0.5 * fasa)
                                + skala_ch * (0.05 + 0.35 * (fraksi_tutup ** 1.5))
                                * np.sin(2.0 * np.pi * (3.0 * f_harm) * t_imu + ch)
                            )
                            derau_turbulen = rng.normal(
                                0.0, 0.12 + 0.28 * fraksi_tutup, size=FS_IMU
                            )
                            imu[ch] = (sinyal + derau_turbulen + geser_blok[ch]).astype(np.float32)

                        # 1 kanal arus kipas (200 Hz -> interpolasi ke 400 Hz):
                        # beban aerodinamis berubah sesuai taraf penghalang pada PWM yang sama
                        arus_200 = (
                            arus_dasar * (1.0 + 0.22 * fraksi_tutup + 0.08 * (fraksi_tutup ** 2))
                            + (0.015 + 0.03 * fraksi_tutup) * np.sin(2.0 * np.pi * f_kipas * t_arus + fasa)
                            + rng.normal(0.0, 0.018, size=FS_ARUS)
                            + geser_blok[6]
                        )
                        arus_400 = np.interp(t_imu, t_arus, arus_200).astype(np.float32)[np.newaxis, :]
                        jendela = np.concatenate([imu, arus_400], axis=0)  # [7, 400]

                        # Split per blok: blok ulangan 0 untuk latih; blok ulangan 1 untuk val & uji
                        if ulangan_blok == 0:
                            X_tr.append(jendela)
                            y_tr.append(kelas)
                        elif ulangan < 2:
                            X_va.append(jendela)
                            y_va.append(kelas)
                        else:
                            X_te.append(jendela)
                            y_te.append(kelas)

    return (
        np.stack(X_tr, axis=0),
        np.asarray(y_tr, dtype=np.int64),
        np.stack(X_va, axis=0),
        np.asarray(y_va, dtype=np.int64),
        np.stack(X_te, axis=0),
        np.asarray(y_te, dtype=np.int64),
    )


def uji_ukuran_masukan_keluaran() -> None:
    """Verifikasi dimensi masukan & keluaran model 1D CNN untuk klasifikasi 4 kelas."""
    import torch

    kasus = [
        ("channels-first 400Hz [N,7,400]", np.zeros((8, 7, 400), dtype=np.float32), (8, 7, 400)),
        ("channels-last 400Hz [N,400,7]", np.zeros((8, 400, 7), dtype=np.float32), (8, 7, 400)),
        ("channels-first 200Hz [N,4,200]", np.zeros((5, 4, 200), dtype=np.float32), (5, 4, 200)),
        ("single-channel 2D [N,400]", np.zeros((6, 400), dtype=np.float32), (6, 1, 400)),
    ]
    for nama, x_raw, bentuk_harap in kasus:
        x_nct = ke_kanal_pertama(x_raw)
        assert x_nct.shape == bentuk_harap, f"{nama}: {x_nct.shape} != {bentuk_harap}"
        model = KipasCNN1D(in_channels=x_nct.shape[1], num_classes=JUMLAH_KELAS)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_nct))
        assert logits.shape == (x_nct.shape[0], JUMLAH_KELAS), (
            f"{nama}: keluaran {tuple(logits.shape)} != {(x_nct.shape[0], JUMLAH_KELAS)}"
        )
        pred = logits.argmax(dim=1).numpy()
        assert pred.shape == (x_nct.shape[0],)
        assert pred.min() >= 0 and pred.max() < JUMLAH_KELAS
        print(f"  [OK] {nama} -> masuk {x_nct.shape} -> logits {tuple(logits.shape)} -> pred {pred.shape}")


SITECUSTOMIZE_TEMPLATE = '''"""Shim sementara hanya untuk uji_simulasi.py (tidak mengubah berkas repo)."""
import hashlib
import os
import sys
from pathlib import Path
import numpy as np

_AKAR = Path(os.environ["SIM_AKAR"])
if str(_AKAR) not in sys.path:
    sys.path.insert(0, str(_AKAR))

_NPZ = Path(os.environ["SIM_DATA_NPZ"])
_CACHE = None


def _muat_simulasi():
    global _CACHE
    if _CACHE is None:
        d = np.load(_NPZ)
        _CACHE = (d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["Xte"], d["yte"])
    return _CACHE


def _hash_simulasi(tambahan: bytes = b"") -> str:
    h = hashlib.sha256()
    h.update(b"simulasi-tim26-t26-penghalang-kipas")
    h.update(_NPZ.read_bytes())
    h.update(tambahan)
    return h.hexdigest()[:32]


from src.protokol import data as _proto_data
from src.common import manifest as _manifest

_proto_data.TUGAS = "klasifikasi"
_proto_data.muat = _muat_simulasi
_manifest.data_hash = _hash_simulasi
'''


def main() -> None:
    ap = argparse.ArgumentParser(description="Uji end-to-end models/123450062/run.py pada data simulasi.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--perangkat", default="cpu laptop")
    a, ekstra = ap.parse_known_args()

    print("1) Verifikasi ukuran masukan & keluaran KipasCNN1D (4 kelas):", flush=True)
    uji_ukuran_masukan_keluaran()

    print("\n2) Membangkitkan data simulasi Tim 26 (getaran 6 kanal + arus 1 kanal, jendela 1 s)...", flush=True)
    Xtr, ytr, Xva, yva, Xte, yte = buat_data_simulasi()
    print(
        f"   X_train={Xtr.shape}, X_val={Xva.shape}, X_test={Xte.shape} | "
        f"kelas latih={np.unique(ytr).tolist()}, val={np.unique(yva).tolist()}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="sim_t26_") as tmp:
        tmp_dir = Path(tmp)
        npz_path = tmp_dir / "simulasi_tim26.npz"
        np.savez(npz_path, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
        (tmp_dir / "sitecustomize.py").write_text(SITECUSTOMIZE_TEMPLATE, encoding="utf-8")

        env = os.environ.copy()
        env["SIM_AKAR"] = str(AKAR)
        env["SIM_DATA_NPZ"] = str(npz_path)
        env["PYTHONPATH"] = str(tmp_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        perintah = [
            sys.executable,
            str(DIREKTORI_MODEL / "run.py"),
            "--seed",
            str(a.seed),
            "--epochs",
            str(a.epochs),
            "--perangkat",
            a.perangkat,
        ]
        if a.final:
            perintah.append("--final")
        perintah.extend(ekstra)

        print(f"\n3) Menjalankan perintah baku: {' '.join(perintah)}", flush=True)
        subprocess.run(perintah, check=True, cwd=AKAR, env=env)

    metrics_path = AKAR / "results" / "123450062" / "metrics.json"
    hasil = json.loads(metrics_path.read_text(encoding="utf-8"))
    for kunci in ("metrik", "seed", "data_hash", "perangkat", "model", "n_val", "durasi_detik"):
        assert kunci in hasil, f"Kunci wajib {kunci!r} tidak ada di metrics.json"
    assert hasil["model"] == "cnn"
    assert hasil["seed"] == a.seed
    assert hasil["n_val"] == len(yva)
    assert "val_acc" in hasil["metrik"] and "val_macro_f1" in hasil["metrik"]
    if a.final:
        assert hasil["n_test"] == len(yte)
        assert "test_acc" in hasil["metrik"] and "test_macro_f1" in hasil["metrik"]

    print("\n4) Verifikasi results/123450062/metrics.json [OK]:", flush=True)
    print(json.dumps(hasil, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
