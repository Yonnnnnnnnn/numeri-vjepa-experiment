# 🎓 Google Colab Guide (V2 - GitHub Workflow)

Panduan menjalankan **V2 Inference Pipeline** di Google Colab dengan clone langsung dari **GitHub**.

## Syarat Utama

- **Akun Google** (Standard/Free cukup).
- **Runtime Type:** T4 GPU (Wajib). `Runtime` -> `Change runtime type` -> `T4 GPU`.
- **Hugging Face Token:** Simpan token Anda (misal `hf_...`) untuk download PaliGemma.

---

## Langkah 1: Clone Repository dari GitHub

Buat cell baru dan jalankan kode berikut untuk clone repository:

```python
import os

# 1. Clone Repository dari GitHub
REPO_URL = "https://github.com/Yonnnnnnnnn/numeri-vjepa-experiment.git"
PROJECT_DIR = "/content/numeri-vjepa-experiment"

if os.path.exists(PROJECT_DIR):
    print(f"✅ Repository sudah ada di: {PROJECT_DIR}")
    %cd $PROJECT_DIR
    !git pull origin master
else:
    print("📥 Cloning repository...")
    !git clone $REPO_URL $PROJECT_DIR
    %cd $PROJECT_DIR

print(f"📁 Current directory: {os.getcwd()}")

# 2. Install System Dependencies
!apt-get update && apt-get install -y ffmpeg libsm6 libxext6 -qq

# 3. Install Python Dependencies
!pip install -e Techs/v2e-master/v2e-master -q
!pip install transformers timm einops submitit sentencepiece protobuf scikit-learn bitsandbytes accelerate -q
!pip install huggingface_hub[hf_xet] addict yapf langgraph pydantic -q

# [CRITICAL] Downgrade NumPy untuk kompatibilitas Numba (v2e)
!pip install "numpy<2.0" -q

# Install bitsandbytes dengan pendekatan yang kompatibel dengan Colab
!pip install -q bitsandbytes --force-reinstall

print("\n✅ Instalasi selesai!")
print("⚠️ PENTING: Klik tombol 'RESTART RUNTIME' yang muncul di output")
```

---

## Langkah 2: Setup Path (Jalankan SETELAH Restart)

Setelah restart runtime, jalankan cell ini untuk masuk kembali ke folder project:

```python
import os

PROJECT_DIR = "/content/numeri-vjepa-experiment"

if os.path.exists(PROJECT_DIR):
    %cd $PROJECT_DIR
    print(f"✅ Berhasil masuk ke: {os.getcwd()}")
else:
    print("❌ ERROR: Folder project tidak ditemukan. Jalankan ulang Langkah 1.")
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

# Download BERT weights
bert_dir = f"{CHECKPOINTS_DIR}/bert-base-uncased"
if not os.path.exists(bert_dir):
    print("📥 Mengunduh BERT weights...")
    subprocess.run(["python", f"{COUNTGD_PATH}/download_bert.py", "--output_dir", CHECKPOINTS_DIR], check=True)
else:
    print("✅ BERT weights sudah ada.")

# Download GroundingDINO weights
gdd_path = f"{CHECKPOINTS_DIR}/groundingdino_swinb_cogcoor.pth"
if not os.path.exists(gdd_path):
    print("📥 Mengunduh GroundingDINO weights...")
    subprocess.run(["wget", "-q", "-O", gdd_path, "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"], check=True)
else:
    print("✅ GroundingDINO weights sudah ada.")

# Download SAM weights
sam_path = f"{CHECKPOINTS_DIR}/sam_vit_h_4b8939.pth"
if not os.path.exists(sam_path):
    print("📥 Mengunduh SAM weights...")
    subprocess.run(["wget", "-q", "-O", sam_path, "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"], check=True)
else:
    print("✅ SAM weights sudah ada.")

# Download CountGD weights
countgd_path = f"{CHECKPOINTS_DIR}/checkpoint_fsc147_best.pth"
if not os.path.exists(countgd_path):
    print("📥 Mengunduh CountGD weights...")
    subprocess.run(["gdown", "--quiet", "--id", "1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI", "-O", countgd_path], check=True)
else:
    print("✅ CountGD weights sudah ada.")

print("\n✅ Semua checkpoint siap!")
!ls -la {CHECKPOINTS_DIR}
```

---

## Langkah 5: Jalankan V2 Integration (Inference)

```python
# Video demo (sudah termasuk di repo)
VIDEO_FILE = "Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4"

# Output ke folder lokal Colab
OUTPUT_FILE = "/content/output_v2.mp4"

!python Implementation/main.py --video "$VIDEO_FILE" --output "$OUTPUT_FILE"

# Verifikasi
import os
if os.path.exists(OUTPUT_FILE):
    print(f"✅ SUKSES! Video tersimpan di: {OUTPUT_FILE}")
else:
    print("❌ GAGAL: Video tidak ditemukan.")
```

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
