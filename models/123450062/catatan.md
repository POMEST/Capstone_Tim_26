# Catatan model 123450062 (Jalur CNN — 1D CNN)

**Kasus Tim 26**: Deteksi penghalang aliran udara kipas (klasifikasi 4 kelas: 0%, 25%, 50%, 75%).  
**Masukan**: Sinyal multi-kanal getaran IMU (6 sumbu, 400 Hz) + arus kipas (1 kanal, 200 Hz → diselaraskan), jendela 1 detik (`[N, C, T]` atau `[N, T, C]`, dinormalisasi otomatis ke `[N, C, T]`).  
**Arsitektur**: `KipasCNN1D` (`models/123450062/model.py`) — 3 blok `Conv1d` (kanal `32 -> 64 -> 128`, kernel `11 -> 7 -> 5`, `BatchNorm1d`, `ReLU`, `MaxPool1d(2)` pada blok 1–2) + *dual global pooling* (`AdaptiveAvgPool1d + AdaptiveMaxPool1d`) + kepala klasifikasi linear `256 -> 4` (59.236 parameter, ringan di CPU laptop). Bobot yang dipakai untuk evaluasi dipilih dari **epoch terbaik** berdasarkan `val_macro_f1`, disertai pengukuran **latensi inferensi per jendela (ms)**.

## Alasan optimizer

Menggunakan **AdamW** (`lr=1e-3`, `weight_decay=1e-4`) dengan penjadwal **CosineAnnealingLR** (25 epoch), **bobot kelas** (*inverse-frequency class weights*) pada `CrossEntropyLoss`, dan *gradient clipping* (`max_norm=2.0`):
- Sinyal getaran (MPU6050) dan arus kipas memiliki karakter spektrum dan dinamika gradien yang berbeda antar-kanal; pembaruan adaptif per-parameter pada AdamW membuat konvergensi awal lebih stabil dibanding SGD polos.
- *Decoupled weight decay* pada AdamW menjaga regularisasi L2 tetap konsisten tanpa tercampur estimasi momen adaptif.
- Bobot kelas menjaga kontribusi setiap taraf penghalang (0%, 25%, 50%, 75%) tetap seimbang terhadap metrik utama **F1-makro** meskipun jumlah jendela antar-blok sedikit berbeda.
- CosineAnnealingLR menurunkan *learning rate* secara halus sehingga bobot terbaik cepat tercapai dalam anggaran CPU ≤ 20 menit.

## Regularisasi

1. **Standarisasi z-score per kanal** menggunakan rata-rata dan simpangan baku **data latih saja** (mencegah kebocoran informasi dari validasi/uji).
2. **BatchNorm1d** di setiap blok konvolusi untuk menstabilkan distribusi aktivasi antar-batch dan antar-blok eksperimen.
3. **Dropout** (`0.15` pada blok 2 dan `0.30` sebelum lapisan linear klasifikasi) serta **weight decay** (`1e-4`).
4. **Bobot kelas** pada `CrossEntropyLoss` dan **augmentasi sinyal khusus data latih** (geser waktu siklik ringan `±2.5%` panjang jendela + derau Gaussian `std=0.02`).

## Pencarian hyperparameter (tabel)

*(Diisi ulang dengan angka data meja setelah `src/protokol/data.py` dibekukan oleh Penjaga Data.)*

| ID | Data | Channels | Kernels | LR | Dropout | Epochs | Epoch terbaik | val_acc | val_macro_f1 | Latensi (ms/jendela) | Durasi (s) | Catatan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sim-01 | Simulasi Tim 26 (7 ch, 400 Hz) | 32,64,128 | 11,7,5 | 1e-3 | 0.30 | 25 | 4/25 | 1.0000 | 1.0000 | 0.281 | 67.1 | Uji kerangka Sprint 1 (`--seed 1`) |

## Ablation (>= 2)

`run.py` mendukung ablation domain Tim 26 maupun ablation arsitektur langsung dari CLI:
1. **Kontribusi sensor (domain)**: `--sensor getaran` (getaran saja) vs `--sensor arus` (arus saja) vs `--sensor semua`.
2. **Ukuran kernel temporal**: `--kernels 3` (≈7,5 ms pada 400 Hz) vs `--kernels 7` (≈17,5 ms, ~1 putaran bilah) vs `--kernels 15` (≈37,5 ms).
3. **Arsitektur & regularisasi**: `--no-bn` (tanpa BatchNorm1d), `--pool avg` (average pooling saja), `--no-augment` (tanpa augmentasi), `--no-class-weight` (tanpa bobot kelas).

| Eksperimen Ablation | Perintah | val_acc | val_macro_f1 | durasi (s) | Catatan |
|---|---|---|---|---|---|
| Konfigurasi penuh (baseline CNN) | `python models/123450062/run.py --seed 1` | — | — | — | Menunggu data meja |
| Ablation 1: getaran saja | `python models/123450062/run.py --seed 1 --sensor getaran` | — | — | — | Menunggu data meja |
| Ablation 2: arus saja | `python models/123450062/run.py --seed 1 --sensor arus` | — | — | — | Menunggu data meja |
| Ablation 3: kernel kecil (3) | `python models/123450062/run.py --seed 1 --kernels 3` | — | — | — | Menunggu data meja |
| Ablation 4: kernel lebar (15) | `python models/123450062/run.py --seed 1 --kernels 15` | — | — | — | Menunggu data meja |
| Ablation 5: tanpa BatchNorm | `python models/123450062/run.py --seed 1 --no-bn` | — | — | — | Menunggu data meja |

## Hasil 3 seed konfigurasi akhir

*(Diisi setelah pelatihan 3 seed `--seed 1`, `--seed 2`, `--seed 3` pada data meja.)*

| Seed | val_acc | val_macro_f1 | durasi (s) |
|---|---|---|---|
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |
| **Rerata ± SD** | — | — | — |
