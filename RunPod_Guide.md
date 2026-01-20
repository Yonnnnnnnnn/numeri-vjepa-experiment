# 🚀 RunPod Deployment Walkthrough

Panduan langkah-demi-langkah untuk menjalankan **V-JEPA Event Visualizer** di RunPod.io.

## 1. Persiapan RunPod

1.  **Login/Register:** Buka [runpod.io](https://runpod.io) dan login.
2.  **Top Up Credits:** Pastikan Anda memiliki saldo (min $5 - $10) untuk menyewa GPU.
3.  **Pilih Pod:**
    - Klik menu **"Secure Cloud"** (atau Community Cloud untuk lebih murah, tapi availabilitas bervariasi).
    - Klik **"Deploy"** pada GPU pilihan.
    - **Rekomendasi:** **NVIDIA L40S** (VRAM 48GB). Sangat kencang untuk inference dan harganya bersaing. Jika tidak ada, **L4** atau **A100** juga bisa.

## 2. Konfigurasi Pod

Saat deploy, Anda akan diminta memilih Template:

1.  **Template:** Pilih **RunPod PyTorch 2.1** (atau versi 2.x lainnya).
2.  **Container Disk:** Biarkan default (misal 20GB-40GB).
3.  **Volume Disk:** Biarkan default.
4.  Klik **"Deploy On-Demand"**.
5.  Tunggu hingga status Pod berubah dari `Creating` menjadi `Running` (warna hijau).

## 3. Masuk ke Jupyter Lab

1.  Klik tombol panah biru / **"Connect"** pada Pod Anda.
2.  Pilih **"Connect to Jupyter Lab"** (Tombol oranye).
3.  Tab baru akan terbuka menampilkan antarmuka Jupyter Lab.

## 4. Upload Kode

1.  Di Jupyter Lab (panel kiri), Anda berada di folder `/workspace`.
2.  **Drag & Drop:** Tarik folder `Implementation` dari komputer Anda langsung ke panel file browser di kiri Jupyter Lab.
    - _Alternatif:_ Upload file ZIP lalu unzip di terminal jika folder terlalu besar.
3.  Pastikan struktur folder terlihat seperti ini:
    ```
    /workspace/Implementation/
    ├── main.py
    ├── setup.sh
    ├── requirements.txt
    ├── src/
    └── scripts/
    ```

## 5. Install & Setup (Satu Langkah)

1.  Buka **Terminal** di Jupyter Lab:
    - Klik `File` -> `New` -> `Terminal` (atau klik ikon Terminal di Launcher).
2.  Masuk ke folder project:
    ```bash
    cd Implementation
    ```
3.  Jalankan script setup otomatis:
    ```bash
    bash setup.sh
    ```
    - _Tunggu 2-5 menit._ Script ini akan menginstall FFmpeg, library Python, dan mendownload model AI (Vicuna + V-JEPA/CLIP).

## 6. Upload Video Input

1.  Siapkan video test di komputer Anda (misal `my_video.mp4`).
2.  Drag & drop video tersebut ke dalam folder `Implementation` di Jupyter Lab.

## 7. Jalankan Visualizer

Di terminal yang sama, jalankan perintah:

```bash
python main.py --video my_video.mp4 --output hasil_final.mp4
```

- **Proses:** Anda akan melihat progress bar saat sistem memproses video frame-by-frame.
- **Waktu:** Tergantung durasi video dan GPU.

## 8. Download Hasil

1.  Setelah selesai, file `hasil_final.mp4` akan muncul di panel kiri.
2.  Klik kanan pada file tersebut -> **Download**.
3.  Selesai! Tonton hasil video Side-by-Side (Original vs Event) di komputer Anda.

---

**Tips Hemat Biaya:**
Jangan lupa untuk **Stop** atau **Terminate** Pod Anda di dashboard RunPod setelah selesai agar saldo tidak terpotong terus menerus!
