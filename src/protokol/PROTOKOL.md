# Protokol evaluasi tim — Tim 26

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Deteksi penghalang aliran udara kipas (Tim 26, pembimbing: Christyan Tamaro Nadeak) |
| Sumber dataset | Data simulasi pra-pengambilan data meja 11 Nov (`python meja.py --simulasi jalankan ../rencana/t26-penghalang-kipas.yaml` / generator dummy `src/protokol/buat_simulasi_dummy.py` di `data/sesi_simulasi/`, mengacu pada `rencana/t26-penghalang-kipas.yaml`). |
| Jenis tugas (`TUGAS`) | `klasifikasi` (4 kelas) |
| Target / label | Tingkat tutupan penghalang dari kolom `penghalang` pada `desain.csv`: `0` = `tanpa penghalang`, `1` = `kertas menutup 25%`, `2` = `kertas menutup 50%`, `3` = `kertas menutup 75%` |
| Masukan model | Getaran `imu.csv` baris `imu = kipas` (400 Hz: `ax, ay, az, gx, gy, gz`) dan arus `daya.csv` baris `kanal = kipas` (200 Hz: `arus`), dipotong menjadi jendela 1 detik pada grid 400 Hz (`[N, 400, 7]`) |
| Ukuran data latih / validasi / uji | 722 / 190 / 912 jendela (dari 96 kondisi = 8 blok × 12 kondisi; tiap kondisi 20 s dikurangi 1 s awal = 19 jendela 1 s) |
| Cara split | **Data uji dipisah per blok**: 4 blok uji (1 blok per kelas, 48 kondisi = 912 jendela) dipisah penuh dari 4 blok latih/validasi (48 kondisi = 912 jendela). Di dalam blok latih/validasi, kondisi diurutkan deterministik per `(kelas, kondisi)` lalu dibagi dengan `src.common.split.indeks(48, porsi_val=0.2)` seed `2026` (38 kondisi latih = 722 jendela; 10 kondisi validasi = 190 jendela). Tidak ada blok yang muncul di latih/validasi dan uji sekaligus. |
| Metrik utama untuk komparasi | **F1-makro** (`val_macro_f1` / `test_macro_f1`, dilengkapi akurasi sebagai metrik sekunder) |
| Praproses yang sama untuk semua model | 1) Filter baris `imu == "kipas"` dan `kanal == "kipas"`; 2) **Buang 1 detik pertama setiap kondisi** (fase peralihan alat); 3) Potong jendela non-overlapping 1 detik pada grid 400 Hz (`arus` 200 Hz diinterpolasi linear ke grid 400 Hz); 4) **Normalisasi z-score per kanal dihitung dari data latih saja** (`mean` dan `std` dari `X_train`), lalu diterapkan ke `X_train`, `X_val`, dan `X_test`. |
| Anggaran komputasi per run | ≤ 20 menit pelatihan di CPU laptop (3 seed pada konfigurasi akhir) |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `32f8c4f2f268315a3cd57fdf11691978` (sesi simulasi dummy, dari `results/baseline/metrics.json`; akan diperbarui lewat PR baru saat data meja asli tersedia) |

## Aturan protokol & pemeriksaan integritas

1. **Pembuangan fase peralihan**: 1 detik pertama (`[t_mulai, t_mulai + 1.0)`) di setiap kondisi 20 detik selalu dibuang sebelum pemotongan jendela 1 detik (`19` jendela/kondisi).
2. **Pemisahan data uji per blok**: Satuan pemisahan data uji adalah `blok`. Seluruh kondisi dan jendela dari satu blok selalu berada pada partisi yang sama (`train+val` atau `test`) dan diperiksa secara otomatis oleh `src.protokol.data.periksa_tanpa_kebocoran()`.
3. **Normalisasi bebas kebocoran**: Statistik normalisasi (`mean`, `std` per kanal) hanya dihitung pada `X_train` dan tidak pernah menggunakan `X_val` maupun `X_test`.
4. **Verifikasi kode**: Jalankan `python -m src.protokol.uji_protokol` untuk memastikan `muat()` mengembalikan split valid dari folder sesi simulasi tanpa kebocoran satuan pemisahan, dan `python -m src.common.baseline --seed 2026` untuk memverifikasi jalur evaluasi `F1-makro`.

Disetujui pembimbing pada: Menunggu peninjauan *pull request* Sprint 0
