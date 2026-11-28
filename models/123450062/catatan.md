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
| sim-01 | Simulasi protokol (`sesi_simulasi`, 722/190) | 32,64,128 | 11,7,5 | 1e-3 | 0.30 | 25 | 4/25 | 1.0000 | 1.0000 | 0.229 | ~16 | Uji kerangka; simulasi terlalu mudah (terpisah sempurna) |
| **meja-01** | **Data meja 17 Nov (473/135 jendela)** | 32,64,128 | 11,7,5 | 1e-3 | 0.30 | 25 | 12/25 | **0.7185** | **0.6272** | 0.217 | 16.1 | **Sprint 1, `--seed 1`, konfigurasi awal belum disetel** |

> Catatan Sprint 1: satu seed pada data meja sudah memenuhi Definition of Done Sprint 1. Pencarian hyperparameter, 3 seed, dan ablation lengkap adalah pekerjaan Sprint 2 (28 Nov–12 Des).

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

## Data meja: cara menjalankan dan temuan Sprint 1

Sesi meja asli (`data/sesi-20261117-160506`) tidak berskema sama dengan `src/protokol/data.py`
(tidak ada `desain.csv`; `imu` bernilai `k`; kolom waktu `t_host`/`t_esp_ms`; arus bernama `arus_mA`).
Tanpa mengubah `data.py`, `models/123450062/siapkan_data_meja.py` mengonversinya ke skema protokol
di `data/sesi_meja_t26/` (tidak di-commit), dan `data/MANIFEST.json` diarahkan ke folder itu:

```bash
python models/123450062/siapkan_data_meja.py     # sekali, setelah sesi meja ada di data/
python models/123450062/run.py --seed 1          # memanggil src.protokol.data.muat()
```

Temuan yang memengaruhi hasil (fakta dari data):
- Kolom `blok` pada `label.csv` hanya 1–4; satuan blok fisik adalah pasangan (`blok`, `ulangan_blok`) = 8 blok. Konverter menomori ulang 1–8 menurut waktu; tanpa itu split per blok gagal (kelas 50 % hanya punya satu blok).
- Laju efektif IMU kipas ≈ 303 Hz (nominal 400) dan arus ≈ 152 Hz (nominal 200) karena sambungan Wi-Fi ESP32 sering putus (lihat `log.txt`). Hanya 1.382 dari 1.824 jendela (76 %) yang lolos syarat cakupan sampel pada `data.py`; sisanya dibuang.
- Jam: `t_host` per baris berjitter (paket tiba bergerombol), jadi konverter memakai `t_esp_ms` + satu offset median ke jam host (selisih awal–akhir sesi < 0,03 s).
- Split protokol tidak seimbang kelas: latih per kelas `167/134/121/51`, validasi `49/28/49/9` (kelas 75 % paling sedikit). Validasi hanya 135 jendela dari 10 kondisi, sehingga F1-makro berisik; jangan dibaca sebagai angka akhir.
- Arus di sesi meja ada yang mendekati 0 mA (kuartil bawah ≈ −0,1 mA, median 99,5 mA). Dugaan: `daya.csv` merekam terus, termasuk jeda 2 detik antar-kondisi saat kipas mati; belum diperiksa apakah ada sebagian kondisi aktif yang juga ≈ 0 (tugas Sprint 2).

Hasil Sprint 1: `val_macro_f1 = 0.6272`, `val_acc = 0.7185` (seed 1, epoch terbaik 12/25, 16 s di CPU,
59.236 parameter, latensi 0,217 ms/jendela). Dugaan (belum diuji): jumlah jendela latih kecil
(473) dan hanya 10 kondisi validasi membuat model sulit membedakan 25 % vs 50 %;
akan diuji lewat ablation `--sensor`, `--kernels`, dan augmentasi di Sprint 2.

## Hasil 3 seed konfigurasi akhir

*(Diisi setelah pelatihan 3 seed `--seed 1`, `--seed 2`, `--seed 3` pada data meja.)*

| Seed | val_acc | val_macro_f1 | durasi (s) |
|---|---|---|---|
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |
| **Rerata ± SD** | — | — | — |
