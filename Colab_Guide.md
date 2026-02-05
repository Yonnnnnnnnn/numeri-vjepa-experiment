# 🎓 Google Colab Guide (V2 - GitHub Workflow)

Panduan menjalankan **V2 Inference Pipeline** di Google Colab dengan clone langsung dari **GitHub**.

## Syarat Utama

- **Akun Google** (Standard/Free cukup).
- **Runtime Type:** T4 GPU (Wajib). `Runtime` -> `Change runtime type` -> `T4 GPU`.
- **Hugging Face Token:** Simpan token Anda (misal `hf_...`) untuk download PaliGemma.

---

## Langkah 0: Bersihkan & Siapkan Runtime (WAJIB - Python 3.12 Compat)

> ⚠️ **PENTING**: Setelah menjalankan cell ini, Anda **HARUS** melakukan `Runtime -> Restart session` **SEBELUM** menjalankan cell lain!

Untuk menghindari konflik **NumPy 2.x** dan **Torchvision**, jalankan ini di cell pertama:

```python
# 1. Uninstall paket bermasalah sepenuhnya (Python 3.12 di Colab)
!pip uninstall -y numpy torch torchvision torchaudio jax jaxlib

# Clear pip cache to handle corrupted downloads
!pip cache purge

# 2. Install versi stabil (Python 3.12 & SAM-2 Compat)
# Using PyTorch 2.6.0 with CUDA 11.8 to support SAM2 mask functions
!pip install numpy==1.26.4
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
!pip install jax==0.4.33 jaxlib==0.4.33

print("=" * 60)
print("🚨 WAJIB: Klik Menu 'Runtime' -> 'Restart session' SEKARANG!")
print("          Lalu lanjutkan ke Langkah 0.5 (Verifikasi)")
print("=" * 60)
```

---

## Langkah 0.5: Verifikasi Environment (Jalankan SETELAH Restart)

> ⚠️ Jalankan cell ini **HANYA SETELAH** Anda melakukan Restart Session dari Langkah 0.

```python
# Verifikasi NumPy & Torch versions
import numpy as np
import torch
import torchvision

print(f"NumPy: {np.__version__} (expected: 1.26.4)")
print(f"Torch: {torch.__version__} (expected: 2.6.0+cu118)")
print(f"Torchvision: {torchvision.__version__} (expected: 0.21.0+cu118)")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA: {torch.version.cuda} - GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA not available! Make sure you're using a GPU runtime.")

# Verify torchvision ops (this is the critical test)
try:
    from torchvision.ops import StochasticDepth
    print("✅ Torchvision Ops verified!")
except Exception as e:
    print(f"❌ Torchvision Ops Error: {e}")
```

## Langkah 1: Clone Repository & Instalasi (PENTING)

Jalankan cell ini. Jika repository Anda **Private**, masukkan **GitHub Token** saat diminta.

```python
import sys
import os
import subprocess

# Base path
REPO_PATH = "/content/numeri-vjepa-experiment"
PROJECT_DIR = REPO_PATH
REPO_URL = "https://github.com/Yonnnnnnnnn/numeri-vjepa-experiment.git"

def setup_repo():
    # 0. Emergency Patch for Torchvision Circular Import
    import sys
    try:
        import torchvision
        import torchvision.ops
        import torch.nn as nn
        if not hasattr(torchvision.ops, 'StochasticDepth'):
            print("🩹 Applying Emergency Patch: torchvision.ops.StochasticDepth")
            class StochasticDepth(nn.Module):
                def __init__(self, p=0.1, mode="batch"): super().__init__()
                def forward(self, x): return x
            torchvision.ops.StochasticDepth = StochasticDepth
    except Exception:
        pass

    if os.path.exists(PROJECT_DIR):
        print(f"✅ Repository sudah ada di: {PROJECT_DIR}")
        %cd $PROJECT_DIR
        !git pull origin master
    else:
        print("📥 Cloning repository...")
        res = subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR])
        if res.returncode != 0:
            token = input("Masukkan GitHub PAT: ").strip()
            REPO_URL_TOKEN = REPO_URL.replace("https://", f"https://{token}@")
            !git clone {REPO_URL_TOKEN} {PROJECT_DIR}
        %cd $PROJECT_DIR

    # 1. Pastikan folder Techs Lengkap (Auto-Clone jika hilang)
    TECHS_DIR = os.path.join(PROJECT_DIR, "Techs")
    os.makedirs(TECHS_DIR, exist_ok=True)

    needed_techs = {
        "Depth-Anything-V2-main/Depth-Anything-V2-main": "https://github.com/depth-anything/Depth-Anything-V2.git",
        "sam2-main/sam2-main": "https://github.com/facebookresearch/sam2.git",
        "v2e-master/v2e-master": "https://github.com/YosuaNa/v2e.git", # Fork with Python 3.12 fixes
        "CountVid-main/CountVid-main": "https://github.com/niki-amini-naieni/CountVid.git"
    }

    for folder, url in needed_techs.items():
        full_path = os.path.join(TECHS_DIR, folder)
        if not os.path.exists(full_path):
            print(f"⚠️ Folder {folder} hilang! Mendownload dari source...")
            # Kita buat manual nested structure-nya agar cocok dengan path engine.py
            target_parent = os.path.dirname(full_path)
            os.makedirs(target_parent, exist_ok=True)
            !git clone {url} {full_path}

    # 2. Fix Techs Package Init
    TECH_LIBS = [
        f"{PROJECT_DIR}/Techs/Depth-Anything-V2-main/Depth-Anything-V2-main/depth_anything_v2",
        f"{PROJECT_DIR}/Techs/CountVid-main/CountVid-main",
    ]
    for lib in TECH_LIBS:
        if os.path.exists(lib):
            init_file = os.path.join(lib, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Auto-generated package init\n")
                print(f"✅ Created missing __init__.py for {os.path.basename(lib)}")

    # 3. Add to sys.path
    if PROJECT_DIR not in sys.path:
        sys.path.append(PROJECT_DIR)
        sys.path.append(os.path.join(PROJECT_DIR, "Implementation"))
        DEPTH_PARENT = os.path.join(PROJECT_DIR, "Techs/Depth-Anything-V2-main/Depth-Anything-V2-main")
        if DEPTH_PARENT not in sys.path:
            sys.path.append(DEPTH_PARENT)
            print(f"✅ Added {DEPTH_PARENT} to sys.path")

    return True

if setup_repo():
    # 2. Install System Dependencies
    print("📦 Installing system dependencies...")
    !apt-get update && apt-get install -y ffmpeg libsm6 libxext6 -qq

    # 3. Install Python Dependencies
    print("🐍 Installing python dependencies...")
    !pip install --upgrade pip setuptools wheel -q

    # Kita paksa paksa numpy==1.26.4 di setiap install agar pip tidak meng-upgrade-nya
    # [CRITICAL] Fix for 'Failed building wheel for tokenizers' on Python 3.12
    !pip install "tokenizers>=0.19" "transformers>=4.44.0" numpy==1.26.4 --only-binary :all: -q

    # Dependencies for Model Engines
    !pip install hydra-core omegaconf numpy==1.26.4 -q
    !pip install -e Techs/v2e-master/v2e-master -q
    !pip install -e Techs/sam2-main/sam2-main -q
    !pip install timm einops submitit sentencepiece protobuf scikit-learn bitsandbytes accelerate numpy==1.26.4 -q
    !pip install huggingface_hub[hf_xet] addict yapf langgraph pydantic pydantic-settings scipy numpy==1.26.4 -q

    import numpy as np
    import torch
    print(f"✅ Instalasi Selesai! NumPy: {np.__version__}, Torch: {torch.__version__}")
    if np.__version__.startswith('2'):
        print("⚠️ WARNING: NumPy ter-upgrade ke 2.x! Menjalankan Emergency Downgrade...")
        !pip install numpy==1.26.4 -q
        import importlib
        importlib.reload(np)
        print(f"✅ NumPy Downgraded: {np.__version__}")

    print("🚀 PENTING: Lakukan RESTART SESSION (Menu: Runtime -> Restart session)")
```

## ,StartLine:75,TargetContent:

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
PROJECT_DIR = "/content/numeri-vjepa-experiment"
%cd $PROJECT_DIR
!python Implementation/scripts/download_v2_weights.py
```

> [!NOTE]
> Per **2026-02-02**: Perbaikan "Visualizer vs Logic" (Coordinate Scaling & StochasticDepth Patch) sekarang sudah diinjeksi langsung ke dalam file `.py`. Anda tidak perlu lagi melakukan patching manual di cell Colab. Cukup jalankan Langkah 5 di bawah.

### 4.2 Download CountVid Checkpoints

```python
import os
import subprocess

PROJECT_DIR = "/content/numeri-vjepa-experiment"
COUNTVID_PATH = f"{PROJECT_DIR}/Techs/CountVid-main/CountVid-main"
CHECKPOINTS_DIR = f"{COUNTVID_PATH}/checkpoints"

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

# 4.2.1 Download CountVid Core weights (FSC-147)
cvid_path = f"{CHECKPOINTS_DIR}/countgd_box.pth"
if not os.path.exists(cvid_path):
    print("📥 Mengunduh CountVid Core weights (Gdrive)...")
    # File ID untuk CountVid (AAAI 2026)
    !gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O {cvid_path} -q
else:
    print("✅ CountVid Core weights sudah ada.")

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
# 1. Konfigurasi File
PROJECT_DIR = "/content/numeri-vjepa-experiment"
VIDEO_FILE = f"{PROJECT_DIR}/Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4"
OUTPUT_FILE = "/content/output_v2.mp4"

# Pastikan kita di folder yang benar
%cd $PROJECT_DIR

print("🧠 BAGIAN 1: Menjalankan Recursive Intent Logic (LangGraph)...")
print("-" * 50)
# Gunakan path absolut ke script
!python {PROJECT_DIR}/Implementation/run_recursive_system.py --video "$VIDEO_FILE"

print("\n\n👁️ BAGIAN 2: Menghasilkan Video Visualisasi (MP4)...")
print("-" * 50)
# Gunakan path absolut ke script
!python {PROJECT_DIR}/Implementation/main.py --video "$VIDEO_FILE" --output "$OUTPUT_FILE"

print("\n\n✅ Pengujian Selesai!")
```

### Lihat Video Hasil (Visualisasi SAM2 + CountVid)

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

### Error Lama Masih Muncul di Log? (Stale Code)

Jika Anda sudah melakukan `git pull` tapi error lama masih muncul:

1. Klik menu **Runtime** -> **Restart session**.
2. Jalankan ulang **Langkah 1**.
3. Jalankan ulang perintah Anda.

Ini karena Colab menyimpan kode lama di RAM sampai Anda me-restart session.

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
