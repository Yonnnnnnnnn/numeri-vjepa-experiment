# 🎓 Google Colab Guide (V2 - GitHub Workflow)

Panduan menjalankan **V2 Inference Pipeline** di Google Colab dengan clone langsung dari **GitHub**.

## Syarat Utama

- **Akun Google** (Standard/Free cukup).
- **Runtime Type:** T4 GPU (Wajib). `Runtime` -> `Change runtime type` -> `T4 GPU`.
- **Hugging Face Token:** Simpan token Anda (misal `hf_...`) untuk download PaliGemma.

---

## Langkah 0: Bersihkan Runtime (Opsional tapi Disarankan)

Jika Anda mendapatkan error NumPy 2.x, lakukan ini:
**Menu: Runtime -> Disconnect and delete runtime.** Lalu hubungkan kembali.

## Langkah 1: Clone Repository & Instalasi (PENTING)

Jalankan cell ini. Jika repository Anda **Private**, masukkan **GitHub Token** saat diminta.

```python
import os
import subprocess

# 1. Konfigurasi
REPO_URL = "https://github.com/Yonnnnnnnnn/numeri-vjepa-experiment.git"
PROJECT_DIR = "/content/numeri-vjepa-experiment"

def setup_repo():
    if os.path.exists(PROJECT_DIR):
        print(f"✅ Repository sudah ada di: {PROJECT_DIR}")
        %cd $PROJECT_DIR
        !git pull origin master
        return True

    print("📥 Cloning repository...")
    # Coba clone publik dulu
    res = subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR])

    if res.returncode != 0:
        print("\n⚠️ ERROR: Gagal clone. Repo mungkin private.")
        token = input("Masukkan GitHub Personal Access Token (PAT): ").strip()
        if token:
            REPO_URL_TOKEN = REPO_URL.replace("https://", f"https://{token}@")
            !git clone {REPO_URL_TOKEN} {PROJECT_DIR}
        else:
            print("❌ Tidak ada token. Proses dibatalkan.")
            return False

    %cd $PROJECT_DIR
    return True

if setup_repo():
    # 2. Install System Dependencies
    print("📦 Installing system dependencies...")
    !apt-get update && apt-get install -y ffmpeg libsm6 libxext6 -qq

    # 3. Install Python Dependencies
    print("🐍 Installing python dependencies...")

    # [CRITICAL] Force Install NumPy 1.26.4 PERTAMA KALI
    !pip install numpy==1.26.4 --force-reinstall -q

    # [CRITICAL] Update Build Tools to avoid 'tokenizers' build failure
    !pip install --upgrade pip setuptools wheel -q
    !pip install tokenizers -q

    # [CRITICAL] Dependencies for SAM2 & DepthAnything
    !pip install hydra-core omegaconf -q

    # Use modern stable transformers (robustified by bertwarper.py fix)
    !pip install transformers -q

    !pip install -e Techs/v2e-master/v2e-master -q
    !pip install -e Techs/sam2-main/sam2-main -q
    !pip install timm einops submitit sentencepiece protobuf scikit-learn bitsandbytes accelerate -q
    !pip install huggingface_hub[hf_xet] addict yapf langgraph pydantic pydantic-settings scipy -q

    import numpy as np
    print(f"📊 NumPy Version installed: {np.__version__}")
    if np.__version__.startswith("2"):
         print("⚠️ WARNING: NumPy masih 2.x terbaca di kernel ini.")

    print("\n✅ Instalasi Selesai!")
    print("🚀 PENTING: Lakukan RESTART MANUAL SEKARANG.")
    print("Menu: Runtime -> Restart session")

else:
    print("❌ Setup gagal.")
```

---

## Langkah 2: Verifikasi & Path (Jalankan SETELAH Restart)

Setelah restart runtime, jalankan cell ini untuk memastikan NumPy < 2.0 aktif:

```python
import os
import numpy as np

# 1. Cek NumPy
print(f"📊 NumPy Version: {np.__version__}")
# Numba 0.60 mendukung NumPy 2.0.x, tapi bermasalah di 2.1+ atau 2.4+
if np.__version__.startswith("2") and not (np.__version__.startswith("2.0") or np.__version__.startswith("2.1")):
    print("❌ ERROR: NumPy versi 2.x (selain 2.0/2.1) terdeteksi! Jalankan: !pip install 'numpy<2.1' lalu RESTART lagi.")
else:
    print("✅ NumPy versi kompatibel.")

# 2. Masuk ke folder project
PROJECT_DIR = "/content/numeri-vjepa-experiment"
if os.path.exists(PROJECT_DIR):
    %cd $PROJECT_DIR
    print(f"✅ Berhasil masuk ke: {os.getcwd()}")
else:
    print("❌ ERROR: Folder project tidak ditemukan.")
```

---

## Langkah 3: Login Hugging Face dengan Colab Secrets

### Setup Colab Secret (Hanya sekali)

1. Klik tab **Secrets** di sidebar kiri Colab (ikon kunci 🔑)
2. Klik **Add new secret**
3. Masukkan:
   - **Name**: `HF_TOKEN`
   - **Value**: Token Hugging Face Anda (dari https://huggingface.co/settings/tokens)
4. **Toggle "Notebook access"** ke ON

```python
from google.colab import userdata
from huggingface_hub import login

try:
    token = userdata.get('HF_TOKEN')
    login(token)
    print("✅ Login Hugging Face berhasil!")
except Exception:
    token = input("Masukkan token Hugging Face: ").strip()
    login(token)
    print("✅ Token Hugging Face berhasil disimpan!")
```

---

## Langkah 4: Download Model Weights

### 4.1 Download V-JEPA Weights

```python
!python Implementation/scripts/download_v2_weights.py
```

### 4.2 Download CountGD Checkpoints

```python
import os
import subprocess

PROJECT_DIR = "/content/numeri-vjepa-experiment"
COUNTGD_PATH = f"{PROJECT_DIR}/Techs/CountGD-main/CountGD-main"
CHECKPOINTS_DIR = f"{COUNTGD_PATH}/checkpoints"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Install gdown
subprocess.run(["pip", "install", "-q", "gdown"], check=True)

# Download BERT weights directly using huggingface_hub
bert_dir = f"{CHECKPOINTS_DIR}/bert-base-uncased"
if not os.path.exists(bert_dir):
    print("📥 Mengunduh BERT weights (Snapshot)...")
    from huggingface_hub import snapshot_download
    try:
        # Mengunduh langsung tanpa lewat class transformers untuk menghindari error import
        snapshot_download(repo_id="google-bert/bert-base-uncased", local_dir=bert_dir, local_dir_use_symlinks=False)
        print("✅ BERT download complete.")
    except Exception as e:
        print(f"❌ ERROR Download BERT: {e}")
else:
    print("✅ BERT weights sudah ada.")

# Download GroundingDINO weights
gdd_path = f"{CHECKPOINTS_DIR}/groundingdino_swinb_cogcoor.pth"
if not os.path.exists(gdd_path):
    print("📥 Mengunduh GroundingDINO weights...")
    subprocess.run(["wget", "-q", "-O", gdd_path, "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"], check=True)
else:
    print("✅ GroundingDINO weights sudah ada.")

# 4.2.1 Download CountGD Core weights (FSC-147)
cgd_path = f"{CHECKPOINTS_DIR}/checkpoint_fsc147_best.pth"
if not os.path.exists(cgd_path):
    print("📥 Mengunduh CountGD Core weights (Gdrive)...")
    # File ID dari warning log user
    !gdown --id 1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI -O {cgd_path} -q
else:
    print("✅ CountGD Core weights sudah ada.")

# Download SAM weights
sam_path = f"{CHECKPOINTS_DIR}/sam_vit_h_4b8939.pth"
if not os.path.exists(sam_path):
    print("📥 Mengunduh SAM weights...")
    subprocess.run(["wget", "-q", "-O", sam_path, "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"], check=True)
else:
    print("✅ SAM weights sudah ada.")

# 4.3 Download Depth-Anything V2 weights
depth_dir = f"{PROJECT_DIR}/Techs/Depth-Anything-V2-main/Depth-Anything-V2-main/checkpoints"
os.makedirs(depth_dir, exist_ok=True)
depth_vits_path = f"{depth_dir}/depth_anything_v2_vits.pth"

if not os.path.exists(depth_vits_path):
    print("📥 Mengunduh Depth-Anything V2 (ViT-S) weights...")
    subprocess.run(["wget", "-q", "-O", depth_vits_path, "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"], check=True)
else:
    print("✅ Depth-Anything V2 weights sudah ada.")

# 4.4 Download Demo Video (Wajib untuk Langkah 5)
video_dir = f"{PROJECT_DIR}/Techs/sam2-main/sam2-main/demo/data/gallery"
os.makedirs(video_dir, exist_ok=True)
video_path = f"{video_dir}/02_cups.mp4"

if not os.path.exists(video_path):
    print("📥 Mengunduh Demo Video (02_cups.mp4)...")
    subprocess.run(["wget", "-q", "-O", video_path, "https://raw.githubusercontent.com/facebookresearch/sam2/main/demo/data/gallery/02_cups.mp4"], check=False)
    if os.path.exists(video_path) and os.path.getsize(video_path) < 1000:
        print("⚠️ GitHub LFS detected. Mengunduh dari source alternatif...")
        subprocess.run(["wget", "-q", "-O", video_path, "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_sample.mp4"], check=True)
        print("✅ Video alternatif berhasil diunduh.")

print("\n✅ Semua checkpoint & video siap!")
!ls -la {CHECKPOINTS_DIR}
!ls -la {video_dir}
!ls -la {depth_dir}
```

---

## Langkah 5: End-to-End Validation (Logic + Visual)

Jalankan cell ini untuk melihat **Logika (Teks)** dan menghasilkan **Video (Visual)** secara berurutan.

```python
# 1. Konfigurasi File (Gunakan Path Absolut untuk Keamanan)
PROJECT_DIR = "/content/numeri-vjepa-experiment"
VIDEO_FILE = f"{PROJECT_DIR}/Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4"
OUTPUT_FILE = "/content/output_v2.mp4"

print("🧠 BAGIAN 1: Menjalankan Recursive Intent Logic (LangGraph)...")
print("-" * 50)
# Kita tambahkan flag --video agar path-nya absolut
!python Implementation/run_recursive_system.py --video "$VIDEO_FILE"

print("\n\n👁️ BAGIAN 2: Menghasilkan Video Visualisasi (MP4)...")
print("-" * 50)
!python Implementation/main.py --video "$VIDEO_FILE" --output "$OUTPUT_FILE"

print("\n\n✅ Pengujian Selesai!")
```

### Lihat Video Hasil (Visualisasi SAM2 + CountGD)

JANGAN jalankan ini sebelum Langkah 5 di atas selesai.

### Lihat Video Hasil di Colab

```python
from IPython.display import HTML
from base64 import b64encode

def show_video(video_path):
    mp4 = open(video_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f'<video width=600 controls><source src="{data_url}" type="video/mp4"></video>')

show_video("/content/output_v2.mp4")
```

---

## Tips & Troubleshooting

### Gunakan Video Custom

```python
# Upload video ke Colab, lalu:
VIDEO_FILE = "/content/my_video.mp4"
```

### VRAM Limit (T4 = 15GB)

- PaliGemma (6GB) + V-JEPA (5GB) + CountGD (4GB) = ~15GB
- Jika OOM: Gunakan `load_in_4bit=True` di `vl_jepa_engine.py`

### Bersihkan Disk Colab

```python
!rm -rf ~/.cache/huggingface
!pip cache purge
!rm -f /content/*.mp4
```

---

## Simpan Hasil ke Google Drive (Opsional)

Jika ingin menyimpan hasil ke Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy hasil ke Drive
!cp /content/output_v2.mp4 "/content/drive/MyDrive/output_v2.mp4"
print("✅ Video disimpan ke Google Drive!")
```

---

_Created for: V-JEPA Inventory Project (GitHub Workflow)_
