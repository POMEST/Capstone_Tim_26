"""Skrip pelatihan dan evaluasi jalur Transfer Learning (TL) — NIM 123450122.

Mengadaptasi backbone pralatih ResNet-18 (bobot ImageNet-1K) ke data getaran + arus kipas
berukuran jendela 1 detik (``[N, 400, 7]``):
1. Setiap kanal sinyal diubah menjadi spektrogram log-STFT 2D (``[N, 7, 64, 64]``).
2. Adapter ``Conv2d(7, 3, kernel_size=1) + BatchNorm2d(3)`` memetakan 7 kanal sensor
   ke 3 kanal masukan ResNet-18.
3. Lapisan awal ResNet-18 (``conv1``, ``bn1``, ``layer1``, ``layer2``) dibekukan;
   ``layer3``, ``layer4``, adapter, dan head klasifikasi 4 kelas di-*fine-tune*.

Jalankan:
    python models/123450122/run.py --seed 1
    python models/123450122/run.py --seed 1 --final
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

# Impor torch sebelum numpy/pandas untuk menghindari konflik inisialisasi DLL (WinError 1114) di Windows.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.common.evaluate import macro_f1  # noqa: E402
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"

UKURAN_SPEKTROGRAM = (64, 64)
N_FFT = 64
HOP_LENGTH = 8
EPOCHS = 12
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.3


def sinyal_ke_spektrogram(X: np.ndarray) -> torch.Tensor:
    """Ubah jendela sinyal ``[N, 400, 7]`` menjadi tensor spektrogram log-STFT ``[N, 7, 64, 64]``."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Masukan harus berdimensi 3 [N, langkah, kanal], dapat {arr.shape}")

    n, _, c = arr.shape
    xt = torch.from_numpy(arr).permute(0, 2, 1).reshape(n * c, -1)  # [N*C, T]
    jendela_hann = torch.hann_window(N_FFT, dtype=xt.dtype)
    stft = torch.stft(
        xt,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=jendela_hann,
        center=True,
        return_complex=True,
    )
    mag_log = torch.log1p(stft.abs()).reshape(n, c, stft.shape[-2], stft.shape[-1])
    return F.interpolate(mag_log, size=UKURAN_SPEKTROGRAM, mode="bilinear", align_corners=False)


class ResNet18AdapterTL(nn.Module):
    """Adapter 7->3 kanal di atas spektrogram 2D + ResNet-18 pralatih."""

    def __init__(self, jumlah_kanal: int, jumlah_kelas: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(jumlah_kanal, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Bekukan lapisan awal agar fitur dasar ImageNet tetap terjaga
        for nama, param in self.backbone.named_parameters():
            if not (nama.startswith("layer3.") or nama.startswith("layer4.") or nama.startswith("fc.")):
                param.requires_grad = False

        in_fitur = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_fitur, jumlah_kelas),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self.adapter(x)
        return self.backbone(x3)


def _prediksi_batch(model: nn.Module, X_spec: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    daftar: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_spec), BATCH_SIZE):
            logits = model(X_spec[i : i + BATCH_SIZE].to(device))
            daftar.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(daftar, axis=0).astype(np.int64)


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Latih model ResNet-18 TL pada data simulasi/meja dan kembalikan (pred_val, pred_test|None)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    jumlah_kanal = Xtr.shape[2]
    jumlah_kelas = len(getattr(data, "KELAS_PENGHALANG", range(int(ytr.max()) + 1)))

    S_tr = sinyal_ke_spektrogram(Xtr)
    S_va = sinyal_ke_spektrogram(Xva)
    y_tr_t = torch.from_numpy(np.asarray(ytr, dtype=np.int64))
    y_va_np = np.asarray(yva, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18AdapterTL(jumlah_kanal=jumlah_kanal, jumlah_kelas=jumlah_kelas).to(device)

    param_latih = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_latih, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(S_tr, y_tr_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=gen,
    )

    f1_terbaik = -1.0
    state_terbaik = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by, label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(bx)
        scheduler.step()

        pred_va_ep = _prediksi_batch(model, S_va, device)
        f1_va = macro_f1(y_va_np, pred_va_ep)
        if f1_va >= f1_terbaik:
            f1_terbaik = f1_va
            state_terbaik = copy.deepcopy(model.state_dict())
        print(
            f"[epoch {epoch:02d}/{EPOCHS}] loss={total_loss / len(S_tr):.4f} "
            f"val_macro_f1={f1_va:.4f}",
            flush=True,
        )

    model.load_state_dict(state_terbaik)
    pred_val = _prediksi_batch(model, S_va, device)
    assert pred_val.shape == (len(Xva),), f"Bentuk pred_val salah: {pred_val.shape}"

    pred_test: np.ndarray | None = None
    if final and Xte is not None:
        S_te = sinyal_ke_spektrogram(Xte)
        pred_test = _prediksi_batch(model, S_te, device)
        assert pred_test.shape == (len(Xte),), f"Bentuk pred_test salah: {pred_test.shape}"

    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val, dtype=np.int64))

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
        np.save(out / "pred_test.npy", np.asarray(pred_test, dtype=np.int64))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
