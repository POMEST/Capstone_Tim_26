"""Pembanding non-deep learning Tim 26: Fitur statistik + FFT + Random Forest.

Mengekstrak fitur domain waktu dan frekuensi per kanal dari jendela 1 detik ``[N, 400, 7]``:
- RMS (root mean square)
- Puncak (nilai absolut maksimum)
- Kurtosis (``scipy.stats.kurtosis``)
- Energi pita FFT pada 4 pita frekuensi (0-25 Hz, 25-60 Hz, 60-120 Hz, 120-200 Hz)

Lalu melatih ``RandomForestClassifier`` dan menulis hasil evaluasi baku lewat ``src.common.evaluate``.

Jalankan:
    python -m src.protokol.pembanding_rf --seed 2026
    python -m src.protokol.pembanding_rf --seed 2026 --final
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

FS_HZ = 400.0
PITA_FREKUENSI_HZ = (
    (0.0, 25.0),
    (25.0, 60.0),
    (60.0, 120.0),
    (120.0, 200.0 + 1e-6),
)


def ekstrak_fitur_statistik_fft(X: np.ndarray, fs: float = FS_HZ) -> np.ndarray:
    """Ekstrak [RMS, puncak, kurtosis, energi pita FFT (4 pita)] per kanal -> [N, C * 7]."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Masukan harus [N, langkah, kanal], dapat {arr.shape}")

    n, t, c = arr.shape
    rms = np.sqrt(np.mean(arr ** 2, axis=1))
    puncak = np.max(np.abs(arr), axis=1)
    kurt = kurtosis(arr, axis=1, fisher=True, bias=False)
    kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)

    # Energi pita FFT (Parseval / spektrum daya satu sisi per kanal)
    spek = np.fft.rfft(arr, axis=1)
    daya = (np.abs(spek) ** 2) / max(1, t)
    frek = np.fft.rfftfreq(t, d=1.0 / fs)

    daftar_pita = []
    for f_lo, f_hi in PITA_FREKUENSI_HZ:
        mask = (frek >= f_lo) & (frek < f_hi)
        energi = daya[:, mask, :].sum(axis=1) if np.any(mask) else np.zeros((n, c), dtype=np.float64)
        daftar_pita.append(energi)

    fitur = np.concatenate([rms, puncak, kurt, *daftar_pita], axis=1)
    return fitur.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    Ftr = ekstrak_fitur_statistik_fft(Xtr)
    Fva = ekstrak_fitur_statistik_fft(Xva)

    clf = RandomForestClassifier(
        n_estimators=a.n_estimators,
        random_state=a.seed,
        n_jobs=-1,
    )
    clf.fit(Ftr, np.asarray(ytr, dtype=np.int64))

    t_inf0 = time.perf_counter()
    pred_val = clf.predict(Fva).astype(np.int64)
    latensi_ms = ((time.perf_counter() - t_inf0) * 1000.0) / max(1, len(Xva))
    durasi = time.time() - t0
    print(f"RandomForest latensi={latensi_ms:.3f} ms/jendela, fitur={Ftr.shape[1]}", flush=True)

    out = AKAR / "results" / "pembanding_rf"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        "pembanding_rf",
        "--model",
        "rf",
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if a.final and Xte is not None:
        Fte = ekstrak_fitur_statistik_fft(Xte)
        pred_test = clf.predict(Fte).astype(np.int64)
        np.save(out / "pred_test.npy", pred_test)
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
