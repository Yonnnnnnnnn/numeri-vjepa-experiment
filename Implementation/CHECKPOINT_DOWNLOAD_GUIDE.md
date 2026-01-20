# Panduan Mengunduh Checkpoint CountGD

Berikut adalah panduan langkah demi langkah untuk mengunduh semua checkpoint yang dibutuhkan oleh CountGDEngine, khususnya untuk **Google Colab** yang merupakan environment yang Anda gunakan.

## Panduan Khusus Google Colab

### Langkah 1: Persiapan di Colab
1. Buka Google Colab dan buat notebook baru
2. Pasang GPU: Runtime > Change runtime type > Hardware accelerator > GPU
3. Clone repositori proyek ke Colab:
   ```python
   !git clone https://github.com/username/repository-name.git
   cd repository-name
   ```
   *Ganti dengan URL repositori Anda jika sudah di-hosting* atau upload proyek secara manual melalui Colab UI.

### Langkah 2: Buat Direktori Checkpoints
Jalankan di Colab:
```python
!mkdir -p "Techs/CountGD-main/CountGD-main/checkpoints"
```

### Langkah 3: Unduh Weights BERT
Gunakan script yang disediakan untuk mengunduh weights BERT di Colab:
```python
!python "Techs/CountGD-main/CountGD-main/download_bert.py" --output_dir "Techs/CountGD-main/CountGD-main/checkpoints"
```

### Langkah 4: Unduh Weights GroundingDINO di Colab
```python
!wget -P "Techs/CountGD-main/CountGD-main/checkpoints" https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

### Langkah 5: Unduh Weights SAM (Segment Anything Model) di Colab
```python
!wget -P "Techs/CountGD-main/CountGD-main/checkpoints" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Langkah 6: Unduh Weights CountGD di Colab
Weights CountGD tersedia di Google Drive. Anda dapat mengunduhnya secara langsung di Colab menggunakan gdown:

1. Install gdown jika belum terinstal:
   ```python
   !pip install gdown
   ```

2. Unduh weights CountGD:
   ```python
   !gdown --id 1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI -O "Techs/CountGD-main/CountGD-main/checkpoints/checkpoint_fsc147_best.pth"
   ```

## Verifikasi Checkpoint di Colab
Setelah selesai mengunduh, verifikasi file-file checkpoint:
```python
!ls -la "Techs/CountGD-main/CountGD-main/checkpoints"
```

Anda harus melihat file-file berikut:
- `bert-base-uncased/` (direktori)
- `groundingdino_swinb_cogcoor.pth`
- `sam_vit_h_4b8939.pth`
- `checkpoint_fsc147_best.pth`

## Menggunakan Checkpoint di Colab
Setelah semua checkpoint diunduh, Anda dapat menjalankan pipeline seperti biasa. Sistem akan secara otomatis mendeteksi checkpoint yang telah diunduh dan menggunakan model CountGD asli.

## Alternatif: Menggunakan Google Drive untuk Menyimpan Checkpoint
Jika Anda ingin menggunakan checkpoint yang sama across multiple Colab sessions:

1. Sambungkan Google Drive ke Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Buat direktori checkpoint di Drive:
   ```python
   !mkdir -p "/content/drive/MyDrive/CountGD/checkpoints"
   ```

3. Unduh checkpoint ke Drive:
   ```python
   !gdown --id 1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI -O "/content/drive/MyDrive/CountGD/checkpoints/checkpoint_fsc147_best.pth"
   ```

4. Symlink ke direktori proyek:
   ```python
   !ln -s "/content/drive/MyDrive/CountGD/checkpoints" "Techs/CountGD-main/CountGD-main/checkpoints"
   ```

## Panduan untuk Environment Lokal (Opsional)

### Untuk Windows (PowerShell):
```powershell
# Buat direktori
mkdir -p "Techs/CountGD-main/CountGD-main/checkpoints"

# Unduh BERT
python "Techs/CountGD-main/CountGD-main/download_bert.py" --output_dir "Techs/CountGD-main/CountGD-main/checkpoints"

# Unduh GroundingDINO
Invoke-WebRequest -Uri "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" -OutFile "Techs/CountGD-main/CountGD-main/checkpoints/groundingdino_swinb_cogcoor.pth"

# Unduh SAM
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile "Techs/CountGD-main/CountGD-main/checkpoints/sam_vit_h_4b8939.pth"
```

### Untuk Linux/macOS:
```bash
# Buat direktori
mkdir -p "Techs/CountGD-main/CountGD-main/checkpoints"

# Unduh BERT
python "Techs/CountGD-main/CountGD-main/download_bert.py" --output_dir "Techs/CountGD-main/CountGD-main/checkpoints"

# Unduh GroundingDINO
wget -P "Techs/CountGD-main/CountGD-main/checkpoints" https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# Unduh SAM
wget -P "Techs/CountGD-main/CountGD-main/checkpoints" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Catatan Penting
1. **Ukuran File**: Total ukuran checkpoint sekitar 10 GB, pastikan Colab memiliki ruang disk yang cukup
2. **Kecepatan Unduh**: Colab biasanya memiliki kecepatan unduh yang baik, namun pastikan koneksi stabil
3. **Versi Checkpoint**: Jangan mengganti nama file checkpoint, karena nama tersebut diharapkan oleh kode
4. **GPU di Colab**: Pastikan Anda telah memilih GPU sebagai hardware accelerator untuk performa yang optimal

## Troubleshooting
- **Error "File not found"**: Pastikan path file sesuai dengan yang diharapkan
- **Error saat memuat model**: Periksa apakah semua checkpoint telah diunduh dengan benar
- **CUDA out of memory**: Coba gunakan GPU yang lebih besar di Colab (Runtime > Change runtime type > T4 GPU atau V100)
- **Google Drive quota exceeded**: Unduh checkpoint secara manual dan upload ke Drive jika diperlukan

Setelah semua checkpoint diunduh, sistem akan secara otomatis menggunakan model CountGD asli untuk penghitungan yang lebih akurat!