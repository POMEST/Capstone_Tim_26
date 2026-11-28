"""Skrip pelatihan dan evaluasi jalur CNN (1D CNN) — NIM 123450062.

Kasus Tim 26: Deteksi penghalang aliran udara kipas (klasifikasi 4 kelas:
0%, 25%, 50%, 75% tutupan kertas) dari sinyal getaran IMU + arus kipas
berjendela 1 detik.

Pemakaian baku:
    python models/123450062/run.py --seed 1
    python models/123450062/run.py --seed 1 --final
"""
from __future__ import annotations

import argparse
import copy
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
DIREKTORI_MODEL = Path(__file__).resolve().parent
sys.path.insert(0, str(AKAR))
sys.path.insert(0, str(DIREKTORI_MODEL))

from model import KipasCNN1D, hitung_parameter, ke_kanal_pertama  # noqa: E402
from src.common.evaluate import macro_f1  # noqa: E402
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = DIREKTORI_MODEL.name  # "123450062"
JALUR = "cnn"               # cnn | tl | rnn | trf | ae
JUMLAH_KELAS_BAKU = 4       # 0%, 25%, 50%, 75% penghalang


def tetapkan_seed(seed: int) -> None:
    """Tetapkan seed agar pelatihan dapat direproduksi."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pilih_kanal_sensor(X: np.ndarray, sensor: str) -> np.ndarray:
    """Pilih subset kanal untuk ablation domain: semua | getaran | arus.

    Asumsi konvensi Tim 26 (7 kanal): kanal awal = getaran IMU (6 sumbu),
    kanal terakhir = arus kipas (1 kanal).
    """
    if sensor == "semua":
        return X
    if X.shape[1] < 2:
        raise ValueError(f"--sensor {sensor!r} membutuhkan minimal 2 kanal, diterima {X.shape[1]}")
    if sensor == "getaran":
        return X[:, :-1, :].copy()
    if sensor == "arus":
        return X[:, -1:, :].copy()
    raise ValueError(f"Mode sensor tidak dikenal: {sensor!r}")


def standar_per_kanal(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    Xte: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Standarisasi z-score per kanal menggunakan statistik data latih saja."""
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xtr_s = ((Xtr - mu) / sd).astype(np.float32)
    Xva_s = ((Xva - mu) / sd).astype(np.float32)
    Xte_s = ((Xte - mu) / sd).astype(np.float32) if Xte is not None else None
    return Xtr_s, Xva_s, Xte_s


def hitung_bobot_kelas(ytr: np.ndarray, num_classes: int) -> np.ndarray:
    """Hitung bobot kelas frekuensi-terbalik dari y_train (rata-rata = 1.0)."""
    jumlah = np.bincount(ytr, minlength=num_classes).astype(np.float32)
    ada = jumlah > 0
    bobot = np.ones(num_classes, dtype=np.float32)
    if ada.any():
        bobot[ada] = len(ytr) / (ada.sum() * jumlah[ada])
        bobot[ada] /= bobot[ada].mean()
    return bobot


def _augmentasi_batch(xb, noise_std: float, max_shift: int):
    """Augmentasi ringan khusus data latih: geser waktu siklik + derau Gaussian."""
    import torch

    if max_shift > 0 and xb.shape[-1] > 2 * max_shift:
        geser = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if geser != 0:
            xb = torch.roll(xb, shifts=geser, dims=-1)
    if noise_std > 0.0:
        xb = xb + torch.randn_like(xb) * noise_std
    return xb


def _prediksi_batch(model, X: np.ndarray, batch_size: int, device) -> tuple[np.ndarray, float]:
    """Inferensi per batch; kembalikan (logits, latensi_ms_per_jendela)."""
    import torch

    model.eval()
    hasil = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            logits = model(xb)
            hasil.append(logits.cpu().numpy())
    dt = time.perf_counter() - t0
    latensi_ms = (dt / max(1, len(X))) * 1000.0
    return np.concatenate(hasil, axis=0), latensi_ms


def latih_dan_prediksi(
    seed: int,
    final: bool,
    *,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    channels: tuple[int, int, int] = (32, 64, 128),
    kernels: tuple[int, int, int] = (11, 7, 5),
    optimizer_name: str = "adamw",
    use_bn: bool = True,
    pool_mode: str = "dual",
    augment: bool = True,
    use_class_weight: bool = True,
    sensor: str = "semua",
    layout: str = "auto",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Latih 1D CNN pada data protokol tim dan kembalikan (pred_val, pred_test|None)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tetapkan_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_raw, ytr_raw, Xva_raw, yva_raw, Xte_raw, _ = data.muat()
    tugas = getattr(data, "TUGAS", "klasifikasi")

    Xtr = pilih_kanal_sensor(ke_kanal_pertama(Xtr_raw, layout=layout), sensor)
    Xva = pilih_kanal_sensor(ke_kanal_pertama(Xva_raw, layout=layout), sensor)
    Xte = (
        pilih_kanal_sensor(ke_kanal_pertama(Xte_raw, layout=layout), sensor)
        if (final and Xte_raw is not None)
        else None
    )

    if Xva.shape[1] != Xtr.shape[1]:
        raise ValueError(f"Jumlah kanal X_val ({Xva.shape[1]}) != X_train ({Xtr.shape[1]})")
    if Xte is not None and Xte.shape[1] != Xtr.shape[1]:
        raise ValueError(f"Jumlah kanal X_test ({Xte.shape[1]}) != X_train ({Xtr.shape[1]})")

    Xtr, Xva, Xte = standar_per_kanal(Xtr, Xva, Xte)

    in_channels = Xtr.shape[1]
    if tugas == "regresi":
        ytr = np.asarray(ytr_raw, dtype=np.float32).reshape(-1, 1)
        yva = np.asarray(yva_raw, dtype=np.float32).ravel()
        out_dim = 1
        kriteria = torch.nn.MSELoss()
    else:
        ytr = np.asarray(ytr_raw, dtype=np.int64).ravel()
        yva = np.asarray(yva_raw, dtype=np.int64).ravel()
        out_dim = max(JUMLAH_KELAS_BAKU, int(ytr.max()) + 1) if len(ytr) > 0 else JUMLAH_KELAS_BAKU
        if use_class_weight and len(ytr) > 0:
            bobot = torch.from_numpy(hitung_bobot_kelas(ytr, out_dim)).to(device)
            kriteria = torch.nn.CrossEntropyLoss(weight=bobot)
        else:
            kriteria = torch.nn.CrossEntropyLoss()

    model = KipasCNN1D(
        in_channels=in_channels,
        num_classes=out_dim,
        channels=channels,
        kernels=kernels,
        dropout=dropout,
        use_bn=use_bn,
        pool_mode=pool_mode,
    ).to(device)

    print(
        f"[{NIM}/{JALUR}] tugas={tugas} | sensor={sensor} | X_train={tuple(Xtr.shape)} "
        f"| X_val={tuple(Xva.shape)} | kernels={kernels} | out_dim={out_dim} "
        f"| params={hitung_parameter(model):,} | device={device.type}",
        flush=True,
    )

    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer tidak dikenal: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    gen = torch.Generator()
    gen.manual_seed(seed)
    ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    drop_last = use_bn and len(ds_tr) > batch_size and (len(ds_tr) % batch_size == 1)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=gen, drop_last=drop_last)

    noise_std = 0.02 if augment else 0.0
    max_shift = max(1, Xtr.shape[-1] // 40) if augment else 0

    skor_terbaik = -float("inf")
    epoch_terbaik = 1
    bobot_terbaik = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_sampel = 0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            if augment:
                xb = _augmentasi_batch(xb, noise_std=noise_std, max_shift=max_shift)
            opt.zero_grad(set_to_none=True)
            keluaran = model(xb)
            loss = kriteria(keluaran, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            total_loss += float(loss.item()) * len(xb)
            n_sampel += len(xb)
        scheduler.step()

        logits_ep, _ = _prediksi_batch(model, Xva, batch_size=batch_size, device=device)
        if tugas == "regresi":
            rmse_va = float(np.sqrt(((yva - logits_ep.reshape(-1)) ** 2).mean()))
            skor_ep = -rmse_va
            label_skor = f"val_rmse={rmse_va:.4f}"
        else:
            p_ep = logits_ep.argmax(axis=1)
            f1_va = macro_f1(yva, p_ep)
            skor_ep = f1_va
            label_skor = f"val_macro_f1={f1_va:.4f}"

        if skor_ep > skor_terbaik:
            skor_terbaik = skor_ep
            epoch_terbaik = ep
            bobot_terbaik = copy.deepcopy(model.state_dict())

        if ep == 1 or ep == epochs or ep % max(1, epochs // 5) == 0:
            rata_loss = total_loss / max(1, n_sampel)
            print(f"  epoch {ep:02d}/{epochs:02d} | train_loss={rata_loss:.4f} | {label_skor}", flush=True)

    model.load_state_dict(bobot_terbaik)
    logits_val, latensi_ms = _prediksi_batch(model, Xva, batch_size=batch_size, device=device)
    print(
        f"[{NIM}/{JALUR}] epoch_terbaik={epoch_terbaik}/{epochs} "
        f"| latensi={latensi_ms:.3f} ms/jendela",
        flush=True,
    )

    if tugas == "regresi":
        pred_val = logits_val.reshape(-1).astype(np.float32)
    else:
        pred_val = logits_val.argmax(axis=1).astype(np.int64)

    pred_test = None
    if final and Xte is not None:
        logits_test, _ = _prediksi_batch(model, Xte, batch_size=batch_size, device=device)
        if tugas == "regresi":
            pred_test = logits_test.reshape(-1).astype(np.float32)
        else:
            pred_test = logits_test.argmax(axis=1).astype(np.int64)

    return pred_val, pred_test


def _parse_channels(teks: str) -> tuple[int, int, int]:
    bagian = tuple(int(x.strip()) for x in teks.split(","))
    if len(bagian) != 3:
        raise argparse.ArgumentTypeError("--channels harus 3 angka dipisah koma, mis. 32,64,128")
    return bagian


def _parse_kernels(teks: str) -> tuple[int, int, int]:
    bagian = tuple(int(x.strip()) for x in teks.split(","))
    if len(bagian) == 1:
        k = bagian[0]
        bagian = (k, k, k)
    if len(bagian) != 3 or any(k < 1 or k % 2 == 0 for k in bagian):
        raise argparse.ArgumentTypeError("--kernels harus 1 atau 3 bilangan ganjil, mis. 7 atau 11,7,5")
    return bagian


def main() -> None:
    ap = argparse.ArgumentParser(description="Pelatihan dan evaluasi model 1D CNN (NIM 123450062).")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--channels", type=_parse_channels, default=(32, 64, 128))
    ap.add_argument("--kernels", type=_parse_kernels, default=(11, 7, 5),
                    help="1 angka (mis. 3, 7, 15) atau 3 angka ganjil dipisah koma")
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--no-bn", action="store_true", help="matikan BatchNorm1d (untuk ablation)")
    ap.add_argument("--pool", choices=["dual", "avg", "max"], default="dual")
    ap.add_argument("--no-augment", action="store_true", help="matikan augmentasi sinyal (untuk ablation)")
    ap.add_argument("--no-class-weight", action="store_true", help="matikan bobot kelas pada loss")
    ap.add_argument("--sensor", choices=["semua", "getaran", "arus"], default="semua",
                    help="subset kanal untuk ablation domain Tim 26")
    ap.add_argument("--layout", choices=["auto", "nct", "ntc"], default="auto")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(
        a.seed,
        a.final,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        dropout=a.dropout,
        channels=a.channels,
        kernels=a.kernels,
        optimizer_name=a.optimizer,
        use_bn=not a.no_bn,
        pool_mode=a.pool,
        augment=not a.no_augment,
        use_class_weight=not a.no_class_weight,
        sensor=a.sensor,
        layout=a.layout,
    )
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
