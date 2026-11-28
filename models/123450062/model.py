"""Arsitektur 1D CNN untuk klasifikasi penghalang aliran udara kipas (Tim 26).

Masukan: jendela sinyal multi-kanal 1 detik (getaran IMU + arus kipas), bentuk [N, C, T].
Keluaran: logit klasifikasi 4 kelas (0%, 25%, 50%, 75% tutupan penghalang), bentuk [N, K].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def ke_kanal_pertama(X: np.ndarray, layout: str = "auto") -> np.ndarray:
    """Normalisasi bentuk larik masukan menjadi float32 berdimensi [N, C, T].

    Mendukung format umum dari ``src.protokol.data.muat()``:
    - [N, C, T]: sinyal multi-kanal channels-first (mis. [N, 7, 400])
    - [N, T, C]: sinyal multi-kanal channels-last (mis. [N, 400, 7])
    - [N, T]:    sinyal 1 kanal atau fitur datar (diubah ke [N, 1, T])
    - [N, C, H, W]: bila ada dimensi tunggal, diringkas ke [N, C_eff, T]
    """
    arr = np.asarray(X, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 2:
        return arr[:, np.newaxis, :]

    if arr.ndim == 3:
        if layout == "ntc":
            return np.transpose(arr, (0, 2, 1)).copy()
        if layout == "nct":
            return arr
        # mode "auto": pada sinyal jendela 1 s, panjang waktu T > jumlah kanal C
        if arr.shape[1] > arr.shape[2]:
            return np.transpose(arr, (0, 2, 1)).copy()
        return arr

    if arr.ndim == 4:
        n = arr.shape[0]
        return arr.reshape(n, -1, arr.shape[-1])

    raise ValueError(f"Bentuk masukan tidak didukung untuk 1D CNN: {arr.shape}")


class BlokKonv1D(nn.Module):
    """Satu blok konvolusi 1D: Conv1d -> (BatchNorm1d) -> ReLU -> (MaxPool1d) -> (Dropout)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        use_bn: bool = True,
        pool: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lapisan: list[nn.Module] = [
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=not use_bn,
            )
        ]
        if use_bn:
            lapisan.append(nn.BatchNorm1d(out_ch))
        lapisan.append(nn.ReLU(inplace=True))
        if pool:
            lapisan.append(nn.MaxPool1d(kernel_size=2, stride=2))
        if dropout > 0.0:
            lapisan.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*lapisan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KipasCNN1D(nn.Module):
    """1D CNN untuk jendela sinyal 1 detik getaran + arus kipas.

    Menggunakan 3 blok konvolusi bertingkat (kernel 11 -> 7 -> 5) diikuti
    dual global pooling (AdaptiveAvgPool1d + AdaptiveMaxPool1d) sehingga:
    - Rata-rata global menangkap energi getaran & level arus DC rata-rata.
    - Maksimum global menangkap puncak getaran & turbulensi aliran udara.
    - Bebas ukuran panjang jendela T (berjalan pada 200 Hz, 400 Hz, dll.).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 4,
        channels: tuple[int, int, int] = (32, 64, 128),
        kernels: tuple[int, int, int] = (11, 7, 5),
        dropout: float = 0.3,
        use_bn: bool = True,
        pool_mode: str = "dual",
    ) -> None:
        super().__init__()
        if len(channels) != 3 or len(kernels) != 3:
            raise ValueError("channels dan kernels harus berisi tepat 3 nilai.")
        if pool_mode not in {"dual", "avg", "max"}:
            raise ValueError(f"pool_mode harus 'dual', 'avg', atau 'max', bukan {pool_mode!r}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pool_mode = pool_mode

        c1, c2, c3 = channels
        k1, k2, k3 = kernels
        self.fitur = nn.Sequential(
            BlokKonv1D(in_channels, c1, kernel_size=k1, use_bn=use_bn, pool=True, dropout=0.0),
            BlokKonv1D(c1, c2, kernel_size=k2, use_bn=use_bn, pool=True, dropout=dropout * 0.5),
            BlokKonv1D(c2, c3, kernel_size=k3, use_bn=use_bn, pool=False, dropout=0.0),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        dim_kepala = c3 * 2 if pool_mode == "dual" else c3
        self.kepala = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(dim_kepala, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Masukan KipasCNN1D harus 3D [N, C, T], diterima {tuple(x.shape)}")
        h = self.fitur(x)
        if self.pool_mode == "dual":
            z = torch.cat([self.avg_pool(h), self.max_pool(h)], dim=1).squeeze(-1)
        elif self.pool_mode == "avg":
            z = self.avg_pool(h).squeeze(-1)
        else:
            z = self.max_pool(h).squeeze(-1)
        return self.kepala(z)


def hitung_parameter(model: nn.Module) -> int:
    """Jumlah parameter yang dapat dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
