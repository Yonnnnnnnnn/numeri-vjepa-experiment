# 🎓 Google Colab Guide (V2 "Glide-and-Count")

Panduan menjalankan **V2 Inference Pipeline** di Google Colab (Free Tier - T4 GPU).

## Syarat Utama

- **Akun Google** (Standard/Free cukup).
- **Runtime Type:** T4 GPU (Wajib). `Runtime` -> `Change runtime type` -> `T4 GPU`.
- **Hugging Face Token:** Simpan token Anda (misal `hf_...`) untuk download PaliGemma.

## Langkah 1: Mount Drive & Install Dependencies

Buat cell baru dan jalankan kode berikut untuk mempersiapkan environment:

```python
from google.colab import drive
import os

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Setup Persistent Cache (AGAR TIDAK DOWNLOAD ULANG TIAP SESI)
# Kita simpan "otak" model PaliGemma di Drive Anda
os.environ['HF_HOME'] = "/content/drive/MyDrive/Antigravity_V2/checkpoints/hf_cache"
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

# 2. Setup Project Path
# TIPS: Jika error, kita list folder yang ada di MyDrive untuk memverifikasi namanya
print("Folder yang tersedia di MyDrive:", os.listdir("/content/drive/MyDrive"))

PROJECT_PATH = "/content/drive/MyDrive/Antigravity_V2"

if not os.path.exists(PROJECT_PATH):
    print(f"❌ ERROR: Folder {PROJECT_PATH} TIDAK DITEMUKAN!")
    print("Silakan cek sidebar (ikon Folder 📁), cari folder project Anda, klik kanan -> 'Copy path'.")
else:
    %cd $PROJECT_PATH
    print(f"✅ Berhasil masuk ke: {os.getcwd()}")

    # 3. Install System Dependencies
    !apt-get update && apt-get install -y ffmpeg libsm6 libxext6

    # 4. Install Python Dependencies
    # Kita install v2e langsung dari folder Techs yang Anda upload
    !pip install -e Techs/v2e-master/v2e-master
    !pip install transformers timm einops submitit sentencepiece protobuf scikit-learn bitsandbytes accelerate
    !pip install huggingface_hub[hf_xet] addict yapf

    # [CRITICAL] Downgrade NumPy untuk kompatibilitas Numba (v2e)
    # Anda AKAN melihat error "dependency conflict" untuk OpenCV/JAX.
    # ABAIKAN SAJA, ini memang diperlukan untuk project ini.
    !pip install "numpy<2.0"

    # Install bitsandbytes dengan pendekatan yang kompatibel dengan Colab
    !pip install -q bitsandbytes --force-reinstall

# ⚠️ PENTING: Klik tombol "RESTART RUNTIME" yang muncul di output
# setelah instalasi selesai.
```

## Langkah 2: Setup Project Path (Jalankan SETELAH Restart)

Setelah Anda klik "Restart Runtime", posisi folder Anda akan kembali ke `/content`. Jalankan cell ini untuk masuk kembali ke folder project:

```python
import os
PROJECT_PATH = "/content/drive/MyDrive/Antigravity_V2"

if not os.path.exists(PROJECT_PATH):
    print(f"❌ ERROR: Folder {PROJECT_PATH} TIDAK DITEMUKAN!")
else:
    %cd $PROJECT_PATH
    print(f"✅ Berhasil masuk kembali ke: {os.getcwd()}")
```

## Langkah 3: Bersihkan Cache dan File Tidak Perlu

```python
import shutil
import os

# Hapus cache Hugging Face yang korup
hf_cache = "/content/drive/MyDrive/Antigravity_V2/checkpoints/hf_cache"
if os.path.exists(hf_cache):
    print(f"🗑️ Menghapus cache yang korup di: {hf_cache}...")
    shutil.rmtree(hf_cache)
    print("✅ Cache dihapus. Jalankan ulang Cell 4 untuk download PaliGemma yang baru.")
else:
    print("Cache tidak ditemukan, mungkin sudah bersih.")
```

## Langkah 3.1: Login Hugging Face dengan Colab Secrets

### Cara Setup Colab Secret (Hanya sekali)

1. Klik tab **Secrets** di sidebar kiri Colab (ikon kunci 🔑)
2. Klik **Add new secret**
3. Masukkan:
   - **Name**: `HF_TOKEN`
   - **Value**: Masukkan token Hugging Face Anda (dapat diambil dari https://huggingface.co/settings/tokens)
4. Klik **Add**

Setelah secret dibuat, jalankan cell berikut untuk login secara otomatis:

```python
from huggingface_hub import login
import os

# Ambil token dari Colab Secrets
token = os.getenv("HF_TOKEN")

if token:
    login(token)
    print("✅ Token Hugging Face berhasil diambil dari Colab Secrets!")
else:
    # Fallback jika secret tidak ditemukan
    token = input("Masukkan token Hugging Face Anda secara manual: ").strip()
    login(token)
    os.environ["HF_TOKEN"] = token
    print("✅ Token Hugging Face berhasil disimpan!")
```

## Langkah 4: Download Model Weights

### 3.1 Download V-JEPA Weights (Layer 3)

Gunakan skrip yang sudah kita siapkan untuk mendownload bobot model Layer 3.

```python
!python Implementation/scripts/download_v2_weights.py
```

### 3.2 Download CountGD Checkpoints (Layer 4)

Jalankan cell berikut untuk mendownload semua checkpoint yang dibutuhkan oleh CountGDEngine. Checkpoint akan disimpan di Drive, jadi tidak perlu diunduh ulang jika sudah ada:

```python
import os
import subprocess

# 1. Setup Paths
PROJECT_PATH = "/content/drive/MyDrive/Antigravity_V2"
COUNTGD_PATH = f"{PROJECT_PATH}/Techs/CountGD-main/CountGD-main"
CHECKPOINTS_DIR = f"{COUNTGD_PATH}/checkpoints"

# Buat direktori jika belum ada
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# 2. Install gdown
print("📦 Menginstal gdown...")
subprocess.run(["pip", "install", "-q", "gdown"], check=True)

# 3. Download BERT weights (hanya jika belum ada)
bert_dir = f"{CHECKPOINTS_DIR}/bert-base-uncased"
if not os.path.exists(bert_dir):
    print("📥 Mengunduh BERT weights...")
    subprocess.run(["python", f"{COUNTGD_PATH}/download_bert.py", "--output_dir", CHECKPOINTS_DIR], check=True)
else:
    print("✅ BERT weights sudah ada, melewati unduh.")

# 4. Download GroundingDINO weights (hanya jika belum ada)
gdd_path = f"{CHECKPOINTS_DIR}/groundingdino_swinb_cogcoor.pth"
if not os.path.exists(gdd_path):
    print("📥 Mengunduh GroundingDINO weights...")
    subprocess.run(["wget", "-q", "-O", gdd_path, "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"], check=True)
else:
    print("✅ GroundingDINO weights sudah ada, melewati unduh.")

# 5. Download SAM weights (hanya jika belum ada)
sam_path = f"{CHECKPOINTS_DIR}/sam_vit_h_4b8939.pth"
if not os.path.exists(sam_path):
    print("📥 Mengunduh SAM weights...")
    subprocess.run(["wget", "-q", "-O", sam_path, "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"], check=True)
else:
    print("✅ SAM weights sudah ada, melewati unduh.")

# 6. Download CountGD weights dari Google Drive (hanya jika belum ada)
countgd_path = f"{CHECKPOINTS_DIR}/checkpoint_fsc147_best.pth"
if not os.path.exists(countgd_path):
    print("📥 Mengunduh CountGD weights...")
    subprocess.run(["gdown", "--quiet", "--id", "1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI", "-O", countgd_path], check=True)
else:
    print("✅ CountGD weights sudah ada, melewati unduh.")

# 7. Verifikasi download
print("\n✅ Verifikasi Checkpoint CountGD:")
subprocess.run(["ls", "-la", CHECKPOINTS_DIR])
```

## Langkah 5: Jalankan V2 Integration (Inference)

Gunakan skrip `integration_v2.py` untuk menjalankan pipeline 4-layer secara penuh.

```python
# Ganti path video sesuai lokasi di Drive Anda
VIDEO_FILE = "Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4"

# OPSi A: Jalankan Log Kalkulasi (Hanya Teks di Konsol)
# !python Implementation/scripts/integration_v2.py --video $VIDEO_FILE

# OPSI B: Jalankan Visualizer (Menghasilkan VIDEO DASHBOARD .mp4) - REKOMENDASI
# Kita simpan langsung ke Google Drive agar tidak hilang dan mudah diakses
OUTPUT_FILE = "/content/drive/MyDrive/Antigravity_V2/output_results.mp4"

!python Implementation/main.py --video "$VIDEO_FILE" --output "$OUTPUT_FILE"

# Verifikasi file ada di Drive
import os
if os.path.exists(OUTPUT_FILE):
    print(f"✅ SUKSES! Video tersimpan di Drive: {OUTPUT_FILE}")
    print("Silakan buka Google Drive Anda secara manual untuk mendownload/melihat filenya.")
else:
    print("❌ GAGAL: Video tidak ditemukan di Drive.")
```

Cara ganti Video

```
VIDEO_FILE = "/content/drive/MyDrive/Folder_Anda/video_saya.mp4"
```

Jika Video di dalam Folder Project

```
VIDEO_FILE = "data/video_custom_saya.mp4"
```

Path Video Custom

```
VIDEO_FILE = "/content/drive/MyDrive/Antigravity_V2/Video_input/input.mp4"
```

## Tips Visualisasi di Colab

Karena Colab tidak mendukung `cv2.imshow()`, kita harus menampilkan hasil video menggunakan kode HTML. Setelah menjalankan OPSI B di atas, buat cell baru dan tempel kode ini:

```python
from IPython.display import HTML
from base64 import b64encode

# Tampilkan video hasil dashboard
mp4 = open('output_v2.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f"""
<video width=800 controls>
      <source src="{data_url}" type="video/mp4">
</video>
""")
```

**Untuk melihat video hasil di cell Colab:**

```python
from IPython.display import HTML
from base64 import b64encode

def show_video(video_path):
    mp4 = open(video_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f'<video width=600 controls><source src="{data_url}" type="video/mp4"></video>')

# Contoh display hasil
# show_video("path/to/result.mp4")
```

## Troubleshooting T4 GPU (V2)

- **VRAM Limit (15GB):** PaliGemma (6GB) + V-JEPA (5GB) + CountGD (4GB) akan mendekati batas VRAM T4.
- **Optimasi:** Jika terjadi OOM (Out of Memory):
  - Muat PaliGemma dengan `torch_dtype=torch.float16` atau `load_in_4bit=True` (di `vl_jepa_engine.py`).
  - Matikan Layer 1 (v2e emulator visualizer) untuk menghemat RAM.

## 🧹 Bersihkan Disk Colab (Solusi Ruang Disk Penuh)

Jika disk Colab Anda hampir penuh, gunakan cell berikut untuk **melihat file terbesar** dan **membersihkan space**:

```python
# 📋 LANGKAH 1: LIhat Direktori Terbesar di /content
print("📊 Direktori terbesar di /content:")
!du -h /content --max-depth=1 | sort -hr | head -20

# 📋 LANGKAH 2: LIhat File Terbesar di Seluruh VM
print("\n📊 File terbesar di VM (diatas 100MB):")
!find /content -type f -size +100M -exec ls -lh {} \; | sort -hr -k5

# 🧹 LANGKAH 3: Bersihkan Cache dan File Tidak Perlu
print("\n🧹 Membersihkan cache Hugging Face...")
!rm -rf ~/.cache/huggingface

print("🧹 Membersihkan cache pip...")
!pip cache purge

print("🧹 Menghapus file video output lama...")
!rm -f /content/*.mp4 /content/output_*.mp4

print("🧹 Menghapus temporary files di /content...")
!rm -rf /content/tmp* /content/__pycache__ /content/.ipynb_checkpoints

# 🧹 LANGKAH 4: Bersihkan File Checkpoint yang Sudah Tidak Perlu
# Hanya jalankan jika Anda ingin menghapus checkpoint CountGD
# !rm -rf "$PROJECT_PATH/Techs/CountGD-main/CountGD-main/checkpoints/*.pth"

# 📋 LANGKAH 5: Cek Status Disk Sekarang
print("\n📊 Status Disk Setelah Bersihkan:")
!df -h

print("\n✅ Proses bersihkan disk selesai!")
```

### 🔍 Cara Mengidentifikasi Penyebab Disk Penuh:

1. **Lihat Hasil Langkah 1**: Temukan direktori terbesar di `/content` (biasanya folder project Anda)
2. **Lihat Hasil Langkah 2**: Identifikasi file terbesar (biasanya `.pth` checkpoints, `.mp4` video, atau `.zip` file)
3. **Hapus File yang Tidak Perlu**: Gunakan perintah `!rm -f /path/to/file` untuk menghapus file tertentu

### 🎯 Contoh Hapus File Spesifik:

```python
# Hapus file checkpoint tertentu
!rm -f /content/Antigravity_V2/Techs/CountGD-main/CountGD-main/checkpoints/sam_vit_h_4b8939.pth

# Hapus video output lama
!rm -f /content/output_v2_old.mp4

# Hapus folder cache project
!rm -rf /content/Antigravity_V2/checkpoints/hf_cache
```

### Tips Pencegahan Ruang Disk Penuh:

1. **Gunakan Cell 3.2 yang Diperbarui**: Hanya mengunduh checkpoint yang belum ada di Drive
2. **Simpan Video Output ke Drive**: Gunakan path seperti `/content/drive/MyDrive/output_v2.mp4`
3. **Hapus File Setelah Digunakan**: Jangan simpan semua hasil inference lama
4. **Bersihkan Cache Reguler**: Jalankan cell cleanup setiap 2-3 sesi
5. **Batasi Jumlah Frame**: Jika video panjang, gunakan parameter `--max_frames 100` (jika tersedia)
6. **Restart Runtime**: Kadang restart runtime akan membersihkan VM Colab

### ❗ Catatan Penting:

- **Jangan Hapus** folder `/content/drive` (ini adalah mount Google Drive Anda)
- **Hati-Hati** saat menghapus checkpoint - pastikan Anda sudah menyimpannya di Drive
- **Backup Penting**: Simpan checkpoint penting ke Google Drive sebelum menghapus

### 📌 Jika Masih Penuh:

1. Restart Runtime: Runtime > Restart runtime
2. Hapus folder project lama: `!rm -rf /content/Antigravity_V2_old`
3. Gunakan Colab Pro: Untuk mendapatkan VM dengan disk lebih besar (opsional)

---

_Created for: V-JEPA Inventory Project_
