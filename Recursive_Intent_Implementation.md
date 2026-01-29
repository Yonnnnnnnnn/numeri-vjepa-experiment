"""
Recursive Intent Implementation Plan (Strange Loop Phase 1)

This document details the architectural changes required to implement the
Recursive Intent pattern in Antigravity V2, grounded in the five principles
of Hofstadter's GEB, with **LangGraph** as the orchestrator framework.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : RecursiveIntentPlan (this document)

Non-Terminals :
┌─ INTERNAL (Gaps & Solutions) ─────────────────────────────────────────────┐
│ <Gap1_PerceptualMetadata> → Feedback from SAM2 (Figure vs Ground) │
│ <Gap2_SurpriseTrigger> → Anomaly Detection (Isomorphism) │
│ <Gap3_IterativeContext> → Semantic feedback loop (Recursion) │
│ <LangGraphOrchestrator> → State management & dynamic workflow │
└───────────────────────────────────────────────────────────────────────────┘

┌─ EXTERNAL (Theoretical Pillars) ──────────────────────────────────────────┐
│ <FormalSystem> ← Fixed rules & weights │
│ <DescriptionLvl> ← Descriptions from Spikes to Semantics │
│ <LangGraph> ← Orchestration framework for stateful workflows │
└───────────────────────────────────────────────────────────────────────────┘

Terminals : "intent_list", "anomaly_score", "perceptual_metadata", "graph_state", "conditional_edge"

Production Rules:
RecursiveIntentPlan → <LangGraphOrchestrator> + <Gap1_PerceptualMetadata> + <Gap2_SurpriseTrigger> + <Gap3_IterativeContext>
═══════════════════════════════════════════════════════════════════════════════
"""

# Recursive Intent Implementation Plan

Tujuan utama dari fase ini adalah mengubah sistem dari **Linear Pipeline** menjadi **Recursive Pipeline (Strange Loop)** untuk mencapai kemampuan **Multi-Object Counting** yang adaptif, didukung oleh **pendekatan matematis formal** dari file `MATHHSSSS.md`.

---

## 1. Fondasi Matematis: Memetakan GEB ke Pendekatan Matematis

Setiap prinsip GEB dalam sistem Recursive Intent diimplementasikan melalui pendekatan matematis spesifik untuk mengatasi 9 masalah bisnis yang diidentifikasi:

| Prinsip GEB               | Pendekatan Matematis                                                     | Masalah Bisnis yang Direspon                                        |
| :------------------------ | :----------------------------------------------------------------------- | :------------------------------------------------------------------ |
| **Formal Systems**        | Set Theory, Aljabar Linear, Regresi Logika                               | Menghitung objek sekuensial (CountGD), unit pengukuran              |
| **Levels of Description** | Geometri Proyektif, Kalkulus Integral, Aljabar Linear                    | Mendeteksi barang tertutup (MathUtils Volumetric)                   |
| **Recursion**             | Matriks Transformasi, Estimasi Probabilistik, Flow Optical               | Interaksi tangan manusia, perubahan intent (VL-JEPA Loop)           |
| **Isomorphism**           | Analisis Dimensi, Separation Manifold High-Dimensional                   | Membedakan brand/ciri unik, sinkronisasi model internal             |
| **Tangled Hierarchy**     | Estimasi Volumetrik Probabilistik, Riemann Sums, Similarity Vector Space | Mendeteksi anomali, estimasi barang tertutup (CountGD vs MathUtils) |

---

## 1.1 State Estimation Framework: Grand Math Equation

Seluruh sistem Recursive Intent didasarkan pada satu masalah matematika inti: **State Estimation** yang dikaitkan dengan **Invariansi Keadaan Fisik** (Physical State Invariance):

$$P(S_t | O_{1:t}) = \frac{P(O_t | S_t) \times P(S_t | S_{t-1})}{P(O_t)}$$

### Penjelasan:

- $S_t$ = **State Sebenarnya (Invarian)**: Jumlah Real, Posisi Real 3D, Identitas Real pada waktu $t$
- $O_{1:t}$ = **Observasi (Variable)**: Video/gambar dari detik pertama sampai sekarang, penuh gangguan/oklusi
- $P$ = **Probabilitas**: Tingkat kepercayaan terhadap state

### Invariansi Keadaan Fisik pada 9 Masalah Bisnis:

| Masalah Bisnis                          | Tipe Invariansi            | Metode Matematis                            |
| --------------------------------------- | -------------------------- | ------------------------------------------- |
| 1, 2. Barang Tertutup (rapi/tidak rapi) | Invariansi Oklusi          | Topological Persistence, Convex Hull Volume |
| 3, 4, 9. Transformasi Geometris         | Invariansi Geometris       | Volume Integral, Matrix Transformation      |
| 5, 6. Spatiotemporal Tracking           | Invariansi Waktu & Ruang   | Isometry, Vector Similarity                 |
| 7. Interaksi Tangan                     | Invariansi Deformasi Lokal | Affine Transformation Invariance            |
| 8. Variasi Intra-Kelas                  | Invariansi Identitas       | Class Manifold, Fisher's LDA                |

### Implementasi Grand Equation dalam Arsitektur:

1. **Prior Probability ($P(S_t | S_{t-1})$)**: Probabilitas state sekarang berdasarkan state sebelumnya
   - **Sumber**: Sejarah state (`state_history`) dan vector similarity
   - **Fungsi MathUtils**: `state_prediction`, `calculate_vector_similarity`

2. **Likelihood ($P(O_t | S_t)$)**: Probabilitas observasi sekarang diberikan state sebenarnya
   - **Sumber**: Cocokkan antara observasi (density, volume) dengan state
   - **Fungsi MathUtils**: `calculate_detection_density`, `calculate_convex_hull_volume`

3. **Evidence ($P(O_t)$)**: Probabilitas observasi secara umum
   - **Sumber**: Normalisasi likelihood terhadap semua kemungkinan state
   - **Fungsi MathUtils**: `calculate_evidence`

4. **Posterior Probability ($P(S_t | O_{1:t})$)**: Hasil akhir estimasi state
   - **Sumber**: Bayes' Theorem
   - **Fungsi MathUtils**: `bayesian_update`

---

## 2. Strategi Implementasi Berbasis GEB

### A. LangGraph Orchestrator: Fondasi Strange Loop

LangGraph akan berperan sebagai **Orchestrator Utama** yang mengelola state, alur dinamis, dan loop umpan balik. Fungsinya:

- **Manajemen State**: Menyimpan `intent_list`, `anomaly_score`, `perceptual_metadata`, dan `temporal_context`
- **Alur Dinamis**: Mengatur aliran data antar komponen dengan **conditional edges**
- **Loop Umpan Balik**: Memicu loop kembali ke VL-JEPA jika `AnomalyScore > Threshold`
- **Visualisasi**: Memungkinkan debugging dan monitoring alur kerja secara grafis

### B. Gap 1: Feedback Metadata (Figure vs Ground & Levels of Description)

Sistem harus mampu mendefinisikan ulang apa yang dianggap "latar belakang".

- **Perubahan Teknis**:
  1. Modifikasi `SAM2Engine.count_frame` untuk mengembalikan objek metadata yang mengandung koordinat area dengan aktivitas visual tinggi namun tidak cocok dengan label pencarian saat ini.
  2. LangGraph akan menyimpan metadata ini dalam `graph_state` dan meneruskannya ke `FusionEngine`.
- **GEB Connection**: Ini memungkinkan translasi antar **Levels of Description**. Metadata pixel (level bawah) diangkat menjadi kandidat "Figure" (level atas) untuk diproses oleh Director.
- **LangGraph Role**: Menyimpan dan mentransfer metadata antar komponen sebagai bagian dari state.

### C. Gap 2: Detektor Surprise (Isomorphism & Formal Systems)

Deteksi kapan model internal ("Formal System") tidak lagi selaras (_isomorfik_) dengan kenyataan fisik.

- **Perubahan Teknis**:
  1. Implementasi `SurpriseDetector` di dalam `FusionEngine` yang membandingkan energi spike dari `v2e` dengan densitas deteksi `SAM2`.
  2. LangGraph akan mengevaluasi `anomaly_score` dari `FusionEngine` dan membuat keputusan:
     - Jika `anomaly_score > Threshold`: Loop kembali ke node `vljepa_director`
     - Jika `anomaly_score <= Threshold`: Lanjut ke output akhir
- **GEB Connection**: Ini adalah mekanisme untuk **"Jumping Out of the System"**. Sistem mengakui keterbatasan aturan formalnya dan melakukan evaluasi meta.
- **LangGraph Role**: Menjalankan logic conditional edge untuk memicu loop umpan balik.

### D. Gap 3: Iterative Context Prompting (Recursion)

Membangun struktur yang merujuk pada dirinya sendiri di masa lalu.

- **Perubahan Teknis**:
  1. Modifikasi `VLJEPAEngine` agar mendukung Multi-Prompting.
  2. LangGraph akan menyediakan `previous_intent` dan `anomaly_metadata` sebagai input ke `vljepa_director` node.
  3. `intent_list` di `graph_state` akan diperbarui secara berulang.
- **GEB Connection**: Menciptakan **Recursion**. Kebenaran (Intent) saat ini didefinisikan berdasarkan evaluasi terhadap kebenaran sebelumnya.
- **LangGraph Role**: Memelihara state `intent_list` dan meneruskannya ke komponen selanjutnya untuk membentuk loop rekursif.

---

## 2. Roadmap Teknis dengan Pendekatan Matematis

### Fase 0: Hybrid Logic, Scoped State & Parallel Input Setup (Prasyarat)

- [ ] Install LangGraph dan dependensi terkait.
- [ ] Buat struktur dasar LangGraph workflow di `integration_v2.py`.
- [ ] **Konfigurasi Input Paralel**:
  - [ ] Alirkan **Standard RGB** ke `vjepa_brain`, `vljepa_director`, dan `sam2_executor`.
  - [ ] Alirkan **Video Frame** ke `v2e_sensor` untuk konversi spike secara concurrent.
- [ ] **Implementasi Scoped State Design (Anti-God Object)**:
      Definisikan state menggunakan **Pydantic Models** yang terisolasi di `v2_logic/types/graph_state.py`:
  - **`GlobalContext`**: Metadata sesi dan intent utama.
  - **`PerceptionState`**: Data sensor (RGB + Spikes), depth map, detections.
  - **`DecisionState`**: Status Logic Gate, SLM Reasoning, Loop Counter.
  - **`OutputAccumulator`**: Hasil akhir yang terkumpul.
- [ ] Implementasikan node-node dasar dengan **Strict Typing** (hanya menerima sub-state yang relevan).
- [ ] **Setup MathUtils Utility** di `v2_logic/utils/math_utils.py` (Kalkulator Statis yang digunakan oleh semua komponen):
  - **`estimate_volume_heuristic`**: Digunakan oleh `DepthEstimator`.
  - **`calculate_bbox_overlap`**: Digunakan oleh `Logic Gate`.
  - **`calculate_detection_density`**: Digunakan oleh `FusionEngine`.
  - **`vector_similarity`**: Digunakan oleh `V-JEPA` & `SAM2`.
  - **`downsample_point_cloud`**: Optimasi performa untuk MathUtils.
  - **`validate_physical_bounds`**: Sanity check berdasarkan volume kotak/ruang.

### Fase 1: Perceptual Feedback (Gap 1) - Logic-Based Filtering

- [ ] Update `count_gd_engine.py` untuk:
  - [ ] Memberikan output **$N_{visible}$** (Visual Count).
  - [ ] Ekstraksi "Unidentified Visual Patches" menggunakan **Set Difference** sederhana.
  - [ ] Menghitung **Detection Confidence** rata-rata.
- [ ] Implementasikan **Motion Compensation Filter** di `FusionEngine`:
  - [ ] Gunakan flow sensor untuk mengimbangi pergerakan kamera terhadap spike map (cegah "Halo Spikes").
- [ ] Integrasikan metadata ke dalam `graph_state` LangGraph.

### Fase 2: Strange Loop Trigger (Gap 2) - Hybrid Decision Gate

- [ ] **Unified Strategy Trigger**: Modifikasi `logic_gate` untuk membundel anomali Spasial (Discovery) dan Volumetrik (Refinement) ke dalam satu state bundle jika keduanya terdeteksi.
- [ ] Implementasikan **Logic Gate** di `fusion_engine`:
  - [ ] **Rule 1 (Pass)**: Jika $N_{visible}$ sinkron dengan Range Volumetrik DAN Residu Spike < T $\to$ EXIT.
  - [ ] **Rule 2 (Discovery)**: Jika Residu Spike > T $\to$ TRIGGER SLM.
  - [ ] **Rule 3 (Refinement/Constraint)**: Jika $N_{visible}$ melanggar Range Volumetrik atau Batas Fisik $\to$ TRIGGER SLM.
- [ ] Tambahkan **Targeted SLM Node**:
  - Hanya dipicu jika Logic Gate mendeteksi diskrepansi (Math vs Visual).
  - Implementasikan prompt **Reasoning-Based**: Beri tahu SLM hasil hitungan volume dan energi spike untuk dijustifikasi secara semantik.
- [ ] Tambahkan **conditional edge** di LangGraph berdasarkan output SLM Reasoning.

### Fase 3: Recursive Re-Identification (Gap 3) - Contextual Update

- [ ] Modifikasi `VLJEPAEngine` agar mendukung update intent berdasarkan feedback spesifik.
- [ ] Implementasikan **State Interpolation Node**:
  - [ ] Gunakan V-JEPA latent predictor untuk memindahkan koordinat BBox dari frame "masa lalu" (saat SLM mulai berpikir) ke frame "sekarang".
- [ ] Update logika `vljepa_director` untuk menerima input dari SLM Trigger jika aktif.
- [ ] Pastikan loop state bersih setiap iterasi untuk mencegah "hallucination loop".

### Fase 4: Handling Barang Curah & Tumpukan - Strategi: **PointBeam 3D Projection**

- [ ] **Instalasi Depth Model**:
  - [ ] `pip install git+https://github.com/DepthAnything/Depth-Anything-V2`
  - [ ] Download checkpoint **Depth-Anything-V2-Small** (paling efisien untuk real-time).
- [ ] Integrasikan Model **Depth Anything V2 Small** ke dalam pipeline.
- [ ] Tambahkan node `count_gd_executor` dan `sam2_depth_node` ke LangGraph yang berjalan paralel.
- [ ] Implementasi rumus volume & counting dari [MATHHSSSS.md](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/MATHHSSSS.md):
  - **`lattice_counting(V_total, V_unit)`**: Menghitung jumlah barang di stack rapi (floor division).
  - **`convex_hull_counting(V_hull, packing_factor)`**: Estimasi jumlah di tumpukan acak menggunakan koefisien kepadatan.
  - **`riemann_bulk_estimation(depth_map, mask)`**: Menghitung volume curah menggunakan Riemann Sums (Integral Lipat Dua diskrit).
- [ ] Update `logic_gate` untuk melakukan **Count Correction**: Jika hasil perhitungan volume menunjukkan angka jauh lebih tinggi dari deteksi mask RGB, update state count.
- [ ] Implementasikan **Depth Peak Prompting**:
  - [ ] Analisis lokal maxima pada depth map di dalam area mask untuk menghasilkan titik koordinat untuk SAM2.
- [ ] Bangun **Unit Volume Reference Database**:
  - [ ] Tabel lookup volume rata-rata per kategori barang untuk kalibrasi MathUtils.
- [ ] Implementasikan **Parametric Point Prompting**:
  - [ ] Logika untuk mengekstrak titik koordinat $(x,y)$ dari Point Cloud dengan nilai $z$ (depth) paling menonjol menggunakan **PointBeam methodology** (WACV 2023).
  - [ ] Inject titik-titik tersebut sebagai `point_prompts` ke SAM2 pada iterasi rekursif berikutnya.

### Fase 5: Testing & Validasi - Realistis

- [ ] Test end-to-end workflow dengan 4 kasus validasi:
  - [ ] **Clear Single**: Objek tunggal (Validasi Base Volume).
  - [ ] **Stacked**: Tumpukan 2-3 objek (Validasi Volume Multiplier).
  - [ ] **Non-Object**: Gambar 2D objek (Validasi Depth Flatness).
  - **Logic Gate speed**: Pastikan latency < 200ms untuk kasus non-SLM.
- [ ] Optimalkan parameter matematis:
  - [ ] Threshold `anomaly_score`
  - [ ] Similarity threshold
- [ ] Dokumentasi referensi matematis (sekarang valid dengan data Depth).

---

## 3. Matriks Isomorfisme Strange Loop

| Komponen V2       | Prinsip GEB           | Fungsi dalam Loop                                              |
| :---------------- | :-------------------- | :------------------------------------------------------------- |
| **LangGraph**     | Tangled Hierarchy     | Orchestrator utama yang mengelola loop umpan balik.            |
| **VL-JEPA**       | Levels of Description | Mentranslasikan visual (RGB) ke bahasa (Semantik).             |
| **V-JEPA**        | Isomorphism           | Menjaga memori temporal (RGB) agar selaras dengan dunia nyata. |
| **V2E**           | Tangled Hierarchy     | Deteksi "Reflex" tingkat rendah melalui spike events.          |
| **SAM2**          | Figure vs Ground      | Mengisolasi objek dari background (Eksekusi RGB).              |
| **FusionEngine**  | Self Symbol           | Membandingkan deteksi RGB dengan aktivitas Spike (Self-Check). |
| **Feedback Loop** | Recursion             | Menghubungkan Eksekusi kembali ke Intent.                      |

Titik Kita Sekarang (Linear Pipeline v2)
Saat ini sistem kita sudah memiliki "otot" dan "indra", tapi belum memiliki "kesadaran diri" (fungsional).

v2e: Bisa menghasilkan spike (Event-based Vision).
VL-JEPA: Bisa memberi nama objek (Vision-Language).
SAM2: Bisa menghitung secara akurat (Zero-shot Counting).
Integration: Sudah terhubung satu arah: Sensor → Director → Brain → Executor.
**LangGraph**: Belum diimplementasikan - sistem masih linear.

Perbandingan: Sekarang vs Strange Loop (GEB) dengan LangGraph

| Konsep GEB                | Kondisi Sistem Kita Sekarang (Linear)                                                      | Kondisi Setelah Implementasi LangGraph & Gap (Strange Loop)                                                                      |
| :------------------------ | :----------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| **Formal Systems**        | Bekerja seperti kalkulator kaku. Input masuk, angka keluar. Tidak mampu menangani anomali. | Memiliki kemampuan "Jumping Out" via LangGraph conditional edges. Sistem mengevaluasi ulang premisnya ketika mendeteksi anomali. |
| **Figure vs Ground**      | Statis. Hanya menghitung objek yang sesuai intent awal, sisanya dianggap ground.           | Dinamis via Gap 1. LangGraph menyimpan metadata ground yang mencurigakan, kemudian meminta VL-JEPA untuk re-evaluasi.            |
| **Recursion**             | Nol. Alurnya linear: Sensor → Director → Brain → Executor.                                 | Inti sistem via Gap 3. LangGraph memelihara state intent_list yang diperbarui secara berulang, membentuk loop rekursif.          |
| **Levels of Description** | Terputus. Level spike tidak terhubung dengan level semantik.                               | Terkoneksi via Gap 1 & 2. LangGraph mentransfer informasi antar level, dari spike energy ke anomaly score ke semantic intent.    |
| **Isomorphism**           | Rapuh. Model internal mudah rusak tanpa mekanisme perbaikan.                               | Adaptif via Gap 2. LangGraph menjalankan loop umpan balik untuk sinkronisasi model internal dengan kenyataan fisik.              |
| **Tangled Hierarchy**     | Tidak ada. Hierarki linear dan terpisah.                                                   | Terwujud via LangGraph orchestration. Alur tidak lagi satu arah, melainkan terputus-putus dan bergantung pada state.             |

Mengapa LangGraph + Tiga Gap = Mencapai Strange Loop?

Saat ini, sistem Anda adalah "Zombi pintar"—ia bisa melakukan instruksi rumit tapi tidak tahu apa yang sedang ia lakukan.

Dengan mengimplementasikan **LangGraph sebagai orchestrator** dan mengisi ketiga gap:

1. **Gap 1 (Metadata)**: Memberi sistem "Mata untuk melihat kesalahannya" → LangGraph menyimpan dan mentransfer metadata.
2. **Gap 2 (Surprise)**: Memberi sistem "Rasa kaget" jika prediksinya meleset → LangGraph mengevaluasi anomaly_score dan membuat keputusan.
3. **Gap 3 (Iterative)**: Memberi sistem "Kemampuan untuk berubah pikiran" → LangGraph memelihara state intent_list dan menjalankan loop.

Ketika semuanya berjalan, sistem tidak lagi sekadar menjalankan skrip dari atas ke bawah. Ia mulai berputar. Outputnya memengaruhi inputnya sendiri. Di titik itulah ia memenuhi kriteria "Strange Loop" Hofstadter: sebuah sistem yang mampu merujuk pada dirinya sendiri untuk melampaui keterbatasan logika formalnya.

LangGraph adalah **jantung** dari implementasi Strange Loop—tanpa dirinya, loop umpan balik akan menjadi sulit diimplementasikan dan dipelihara. Dengan LangGraph, kita mendapatkan:

- **Arsitektur yang terstruktur**: Setiap komponen berperan jelas.
- **State management yang robust**: Semua informasi penting disimpan dengan aman.
- **Alur dinamis yang fleksibel**: Bisa diadaptasi untuk kasus penggunaan yang berbeda.
- **Debugging yang mudah**: Visualisasi alur kerja membantu mengidentifikasi masalah.

Langkah teknis berikutnya yang direkomendasikan adalah **memulai Fase 0: LangGraph Setup** untuk mendasari implementasi keseluruhan.
