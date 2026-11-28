# Catatan model 123450122 (Jalur TL — Transfer Learning)

## Arsitektur & adaptasi masukan
- **Masukan protokol**: `[N, 400, 7]` (jendela 1 detik pada 400 Hz; 6 kanal IMU `ax, ay, az, gx, gy, gz` + 1 kanal arus `kipas`).
- **Transformasi ke 2D**: log-STFT (`n_fft=64`, `hop_length=8`, jendela Hann) di-resize bilinear ke `[N, 7, 64, 64]`.
- **Adapter + Backbone**: `Conv2d(7, 3, kernel_size=1, bias=False) + BatchNorm2d(3) + ReLU` memetakan 7 kanal sensor menjadi 3 kanal masukan bagi `ResNet-18` pralatih ImageNet-1K (`ResNet18_Weights.DEFAULT`), dengan head `Dropout(0.3) -> Linear(512, 4)`.
- **Strategi fine-tuning**: `conv1`, `bn1`, `layer1`, dan `layer2` dibekukan; `adapter`, `layer3`, `layer4`, dan `fc` dilatih.

## Alasan optimizer
- `AdamW` (`lr=3e-4`, `weight_decay=1e-2`) dengan `CosineAnnealingLR` dipilih karena adaptasi bobot pralatih dengan jumlah sampel kecil (`722` jendela latih) membutuhkan pembaruan langkah adaptif dan *decoupled weight decay* agar stabil tanpa merusak representasi konvolusi pralatih.

## Regularisasi
- Pembekuan lapisan awal ResNet-18 (`layer1–layer2`), `weight_decay=1e-2` pada `AdamW`, `Dropout(0.3)` sebelum lapisan linear akhir, dan `label_smoothing=0.05` pada `cross_entropy`.

## Hasil Sprint 1 (data simulasi dummy, `data_hash`: `32f8c4f2f268315a3cd57fdf11691978`)
| Model / Konfigurasi | Seed | `val_acc` | `val_macro_f1` | Durasi (s) | Berkas hasil |
|---|---|---|---|---|---|
| Baseline linear (`src.common.baseline`) | 2026 | 0.221053 | 0.216055 | 1.9 | `results/baseline/metrics.json` |
| Pembanding non-DL (`src.protokol.pembanding_rf`) | 2026 | 1.000000 | 1.000000 | 5.4 | `results/pembanding_rf/metrics.json` |
| TL ResNet-18 Adapter (`models/123450122/run.py`) | 1 | 0.989474 | 0.988044 | 147.5 | `results/123450122/metrics.json` |

## Pencarian hyperparameter & Ablation (>= 2) — rencana Sprint 2 (data primer meja)
- Pencarian `lr` (`1e-4`, `3e-4`, `1e-3`), `dropout` (`0.2`, `0.3`, `0.5`), dan ukuran spektrogram (`64x64` vs `96x96`).
- Ablation 1: pembekuan penuh backbone (*linear probe* / head + adapter saja) vs *fine-tune* `layer3–layer4`.
- Ablation 2: STFT 7-kanal + adapter `7->3` vs 3-kanal ringkas (`||acc||`, `||gyro||`, `arus`).
- Evaluasi 3 seed (`--seed 1`, `--seed 2`, `--seed 3`), pengukuran ukuran ONNX (KB) dan latensi inferensi (ms).
