# Panduan Mengunduh Checkpoint CountVid (Migrasi Baru)

Berikut adalah panduan langkah demi langkah untuk mengunduh semua checkpoint yang dibutuhkan oleh **CountVidEngine** (CountVid-main) di Google Colab.

> **PENTING**: Panduan ini menggantikan panduan CountGD lama.

## Panduan Khusus Google Colab

### Langkah 1: Persiapan di Colab

1. Buka Google Colab dan buat notebook baru
2. Pasang GPU: Runtime > Change runtime type > Hardware accelerator > GPU
3. Pastikan dependensi terinstall:
   ```python
   !pip install submitit addict termcolor yapf timm scipy
   ```

### Langkah 2: Buat Direktori Checkpoints

Jalankan di Colab:

```python
!mkdir -p "Techs/CountVid-main/CountVid-main/checkpoints"
```

### Langkah 3: Unduh Weights BERT

Gunakan script yang disediakan untuk mengunduh weights BERT (HuggingFace) di Colab:

```python
!python "Techs/CountVid-main/CountVid-main/download_bert.py" --output_dir "Techs/CountVid-main/CountVid-main/checkpoints"
```

### Langkah 4: Unduh Weights CountVid (CountGD-Box)

Weights utama CountVid ("CountGD-Box") tersedia di Google Drive.

```python
!pip install gdown
!gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O "Techs/CountVid-main/CountVid-main/checkpoints/countgd_box.pth"
```

### Langkah 5: Verifikasi

Verifikasi file-file checkpoint:

```python
!ls -la "Techs/CountVid-main/CountVid-main/checkpoints"
```

Anda harus melihat:

- `bert-base-uncased/` (direktori)
- `countgd_box.pth` (sekitar 1.3 GB)

## Troubleshooting

- **Jika gdown gagal**: Coba unduh manual dari [Link Google Drive ini](https://drive.google.com/file/d/1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD/view?usp=sharing), lalu upload ke folder `Techs/CountVid-main/CountVid-main/checkpoints` di Colab.
