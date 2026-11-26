
Tim 26: Deteksi penghalang aliran udara kipas


<table><tr><td>Butir</td><td>Keterangan</td></tr><tr><td>Pilar</td><td>Mesin dan energi</td></tr><tr><td>Pembimbing</td><td>Christyan Tamaro Nadeak</td></tr><tr><td>Anggota dan jalur</td><td>4 orang, satu jalur per anggota: CNN, RNN, Transformer, AE (autoencoder)</td></tr><tr><td>Pembanding non-deep learning</td><td>Fitur statistik + FFT (RMS, puncak, kurtosis, energi pita) + Random Forest</td></tr><tr><td>Jenis tugas</td><td>Klasifikasi 4 kelas</td></tr><tr><td>Masukan model</td><td>Getaran + arus kipas, jendela 1 s</td></tr><tr><td>Metrik utama</td><td>F1-makro</td></tr><tr><td>Purwarupa Sprint 2</td><td>Peringatan ventilasi tertutup</td></tr><tr><td>Jadwal meja</td><td>Operator: Sen 9 Nov 14.00–15.00. Uji purwarupa: Rab 18 Nov 10.30–11.00.</td></tr></table>

Keterangan ini dibaca bersama dokumen Metode Eksperimen Lapangan di Meja Uji Ruang Cerdas 5 V (00-metode-eksperimen-lapangan.md), yang memuat langkah umum, pembagian peran, dan aturan pemakaian data untuk semua tim. 

## 1. Persoalan yang diteliti

Kisi ventilasi yang tertutup debu atau benda membuat kipas bekerja lebih berat dan pendinginan tidak efektif. Tugas tim adalah mendeteksi tingkat tutupan di depan kipas (0%, 25%, 50%, 75%) dari getaran dan arus kipas. Purwarupanya memberi peringatan ‘ventilasi tertutup’. 

## 2. Rencana eksperimen dalam bahasa sehari-hari


Rencana tim ini (rencana/t26-penghalang-kipas.yaml) mengubah 2 faktor. Faktor otomatis disetel skrip, faktor manual dipasang tangan sebelum setiap blok, dan faktor isyarat tampil di layar pada setiap kondisi.


<table><tr><td>Faktor</td><td>Jenis</td><td>Taraf</td><td>Arti di meja</td></tr><tr><td>kipas penghalang</td><td>otomatis manual</td><td>120, 190, 255 tanpa penghalang, kertas menutup 25%, kertas menutup 50%, kertas menutup 75%</td><td>Kecepatan kipas 120, 190, dan 255. Kertas penghalang yang dijepit di depan kisi kipas, menutup 0%, 25%, 50%, atau 75% luas kisi. Garis 25/50/75% ditandai pada kisi.</td></tr></table>

Faktor manual penghalang membentuk 4 kombinasi; karena ulangan_blok: 2, kombinasi itu dipasang ulang 2 kali sehingga ada 8 blok. Urutan blok diacak dengan seed: 1026. Di dalam setiap blok, 3 kombinasi faktor otomatis dan isyarat diulang 4 kali lalu diacak, sehingga satu blok berisi 12 kondisi. Seluruh sesi berisi 96 kondisi. Setiap kondisi direkam 20 detik, diikuti jeda 2 detik ketika alat kembali mati. 

Perekam yang aktif: mikrofon 16 kHz. Sensor ESP32 (getaran, arus, suhu, kelembapan, lux, PIR, dan pintu) selalu merekam pada setiap sesi. 

Waktu rekam bersih sekitar 35 menit, ditambah sekitar 12 menit untuk 8 kali pergantian blok, sehingga perkiraan total sekitar 47 menit. Perkiraan ini muat di dalam slot operator (sekitar 50 menit efektif). 

## 3. Label dan data yang dipakai model

Asal label. Label berasal dari desain (kolom penghalang) dan diperiksa pembimbing sebelum Enter ditekan. 

Data masukan. Getaran imu.csv baris imu = kipas (400 Hz) dan arus daya.csv baris kanal = kipas (200 Hz), jendela 1 detik. Satu detik pertama setiap kondisi dibuang karena alat atau partisipan masih dalam peralihan. 

Satuan pemisahan data uji. Data uji dipisah per blok. Seluruh jendela dari satu satuan berada pada lipatan yang sama, tidak pernah terbagi antara data latih dan data uji. 

## 4. Langkah pengambilan data

## 4.1 Sebelum slot

1. Jalankan rencana dalam mode simulasi: python meja.py --simulasi jalankan ../rencana/t26-penghalang-kipas.yaml. Pastikan skrip mencetak 96 kondisi, lalu uji pemotong jendela tim pada folder sesi simulasi. 

2. Periksa bersama pengelola bahwa sensor getaran (MPU6050) masih menempel kuat dan posisi alat sesuai tanda selotip. 

3. Siapkan tiga lembar kertas yang sudah dipotong sesuai tutupan 25%, 50%, dan 75%, serta penjepit yang sama untuk semua taraf. 

## 4.2 Saat slot

1. Saat blok berganti, layar menampilkan, misalnya, Siapkan: Pasang penghalang di depan kipas: tanpa penghalang. Operator alat memasang keadaan tersebut, pembimbing memeriksa, lalu operator perekam menekan Enter. 

2. Tidak ada yang bersandar pada meja atau menyentuh meja selama kondisi berlangsung; getaran tangan ikut terekam oleh sensor. 

3. Kertas tidak boleh menyentuh bilah kipas; bunyi gesekan menandakan pemasangan salah. 

4. Pencatat menulis nomor kondisi yang terganggu pada lembar sesi. 

## 4.3 Setelah slot

Ikuti bagian 7 dokumen metode umum: tunggu manifest.json, data disalin pengelola, meja dikembalikan ke keadaan acuan, lembar sesi difoto, dan data dimasukkan ke repositori tim. 

## 5. Pemeriksaan mutu data

Selain pemeriksaan umum (bagian 8 dokumen metode), periksa hal berikut pada hari yang sama: 

• Terdapat 96 kondisi (8 blok × 12 kondisi). 

• Arus kipas berubah antar-taraf penghalang pada kecepatan yang sama. 

## 6. Kekeliruan yang perlu dihindari

• Kertas menyentuh bilah, sehingga yang terekam adalah anomali gesekan, bukan tutupan. 

• Posisi kertas bergeser di tengah blok. 

## 7. Kedudukan data sekunder

Model yang dilatih hanya pada data eksternal tidak memenuhi tujuan kasus ini, tetapi dapat menjadi pembanding yang berguna bila diuji langsung pada data uji meja. Bila tim sudah memakai data sekunder, ujilah model tersebut pada data uji meja dan laporkan hasilnya berdampingan dengan model yang dilatih pada data meja (lihat bagian 1.1 dokumen metode). 

## Lampiran: rencana asli

```yaml
# Rencana eksperimen tim 26: Deteksi penghalang aliran udara kipas
# Label: Desain: persentase tutupan kertas (operator) + PWM
tim: 26
judul: Deteksi penghalang aliran udara kipas
pembimbing: Christyan Tamaro Nadeak
pilar: Mesin dan energi
seed: 1026 
```

```yaml
ulangan: 4
ulangan_blok: 2
durasi_kondisi_detik: 20
jeda_detik: 2
rekam:
    audio:
    aktif: true
    sample_rate: 16000
    kamera:
    aktif: false
faktor:
    kipas:
    - 120
    - 190
    - 255
    penghalang:
    jenis: manual
    taraf:
    - tanpa penghalang
    - kertas menutup 25%
    - kertas menutup 50%
    - kertas menutup 75%
    instruksi: 'Pasang penghalang di depan kipas: {taraf}'
metode:
    jenis_tugas: Klasifikasi 4 kelas
masukan_model: Getaran + arus kipas, jendela 1 s
metrik_utama: F1-makro
jalur:
    - CNN
    - RNN
    - Transformer
    - AE
    pembanding_non_dl: Fitur statistik + FFT (RMS, puncak, kurtosis, energi pita)
+ Random Forest 
```