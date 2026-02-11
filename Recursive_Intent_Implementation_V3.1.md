# Recursive Intent Implementation V3.1 (Complete System Architecture)

> **Status**: APPROVED for Production
> **Version**: 3.1
> **Methodology**: Hybrid Geometric-Semantic Reconciliation (Triangulation Engine)

This document outlines the **End-to-End Architecture** of the Recursive Intent System, integrating legacy sensors (Fusion, V2E) with the new V3 Volumetric Engine.

---

## 1. The Triangulation Engine (Core Philosophy)

The system is a **Reconciliation Engine** that triangulates "Truth" from three independent Sovereignties:

1.  **Vision ($N_{vis}$)**: "What can be seen directly?" (CountVid/SAM2).
2.  **Geometry ($N_{vol}$)**: "What is physically probable?" (**V-JEPA Trajectory Memory** + **Alpha Search** + **DINOv2 Texture Analysis**).
3.  **Events (Discovery)**: "What is moving in the dark?" (V2E Spikes).

> [!TIP]
> **V3.1 Physical Insights**:
>
> - **$\rho$ (Density)**: Bukan sekadar angka acak, melainkan hasil analisis "Visual Turbidity" & "Specular Highlights" oleh DINOv2. Pola kilatan cahaya yang kacau menunjukkan kepadatan tinggi (minim airgap).
> - **V-JEPA Memory**: Bertindak sebagai lem spasial. Menggunakan _latent displacement_ (seberapa jauh kamera bergeser) untuk menjahit point cloud dari berbagai frame menjadi satu bungkusan utuh.
> - **Golden Alpha**: Sidik jari fisik per jenis barang. Jika Golden Alpha barang baru mirip dengan barang yang sudah tersimpan, sistem melakukan rekonsiliasi identitas secara fisik, bukan hanya visual.

---

## 2. Complete Component Inventory (21 Components)

### A. The Brain (Orchestration & Logic)

| Component      | Type                   | Production Role                                                                                                                                      |
| :------------- | :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LangGraph**  | **State Orchestrator** | **The Nervous System**: Mengatur alur eksekusi (Nodes & Edges), menyimpan Global Context (Intent, Registry), dan memicu Loop (Discovery/Refinement). |
| **Logic Gate** | **The Guard**          | **Comparator**: Gerbang logika yang membandingkan $N_{vis}$ vs $N_{vol}$ vs `AnomalyScore`. Menentukan apakah sistem "Converged" atau "Recurse".     |
| **MathUtils**  | **Calculation Kernel** | **The Calculator**: Menghitung rumus final $N = (V_{stack} \times \rho) / V_{\mu}$ dan utilitas geometri lainnya.                                    |
| **SLM (Qwen)** | **Verifier / Judge**   | **The Oracle**: Menyediakan data fisik ($V_{\mu}$), melakukan _Reasoning_ konflik, dan memutuskan strategi Refinement.                               |

### B. The Senses (Input Sensors)

| Component           | Type                | Production Role                                                                                 |
| :------------------ | :------------------ | :---------------------------------------------------------------------------------------------- |
| **V2E Engine**      | **Event Sensor**    | **Motion Detector**: Mengubah video menjadi event spikes diskrit untuk deteksi anomali.         |
| **Event Gen**       | **Kernel Utility**  | **Spike Generator**: Utilitas low-level pembantu V2E untuk mensintesis event dari frame gray.   |
| **SAM2**            | **Intent Masker**   | **Visual Filter**: Memisahkan piksel objek target dari latar belakang berdasarkan Intent aktif. |
| **DepthEverything** | **Geometry Sensor** | **Metric Eye**: Menghasilkan Depth Map yang diskalakan ke CM berdasarkan referensi objek.       |
| **CountVid**        | **Visual Counter**  | **The Eye**: Menghitung objek yang terlihat mata ($N_{vis}$) dengan sensitivitas dinamis.       |

### C. Discovery & Identification (Sovereignty 1 & 2)

| Component       | Type                 | Production Role                                                                                         |
| :-------------- | :------------------- | :------------------------------------------------------------------------------------------------------ |
| **VL-JEPA**     | **Director Agent**   | **Semantic Anchor**: Mengidentifikasi objek pertama dan menetapkan Intent awal via _Focused Attention_. |
| **VLM Wrapper** | **Low-level Bridge** | **Qwen Interface**: Backend kuantisasi 4-bit untuk menjalankan VLM/SLM di infrastruktur T4 GPU.         |
| **Embedding**   | **Regional Analyst** | **Feature Extractor**: Mengambil fitur DINOv2 dari potongan gambar (crops) untuk identifikasi SKU.      |
| **Clustering**  | **Group Engine**     | **Category Divider**: Mengelompokkan embedding visual ke dalam kategori SKU yang berbeda.               |
| **ReID Engine** | **Identity Tracker** | **Persistence Link**: Menjaga ID objek tetap konsisten antar frame menggunakan V-JEPA Latent Pose.      |

### D. Geometry & Physics (Sovereignty 3)

| Component         | Type                 | Production Role                                                                                           |
| :---------------- | :------------------- | :-------------------------------------------------------------------------------------------------------- |
| **V-JEPA**        | **Spatio-Temporal**  | **Pose Proxy & Glue**: Menyimpan `PersistentLatentContext` (ingatan) untuk menyatukan Point Clouds.       |
| **AlphaShape**    | **Hull Wrapper**     | **Volume Engine**: Membungkus point cloud menjadi mesh tertutup menggunakan `Golden Alpha` binary search. |
| **DINOv2 Engine** | **Texture Analyst**  | **Density Sensor**: Menganalisis tekstur permukaan tumpukan untuk mencari anomali oklusi.                 |
| **MLP Regressor** | **Density Head**     | **Logic Translator**: Menerjemahkan visual chaos DINOv2 menjadi angka kepadatan benda ($\rho$).           |
| **Fusion Engine** | **Anomaly Watchdog** | **The Shield**: Memvalidasi sinkronisasi antara visual masks vs event spikes untuk deteksi benda asing.   |
| **Temporal Mem**  | **Legacy Layer**     | **V2 Fallback**: Modul penyimpanan peak-count lama (akan digantikan sepenuhnya oleh V-JEPA V3.1).         |

### E. Data Infrastructure

| Component       | Type             | Production Role                                                                                                |
| :-------------- | :--------------- | :------------------------------------------------------------------------------------------------------------- |
| **Graph State** | **Pydantic DTO** | **The Registry**: Struktur data formal yang menyimpan seluruh variabel sistem (GoldenAlpha, Rho, V_unit, dll). |

### 2.2. Target V3.1 Interaction Matrix (The Blueprint)

Tabel ini menggambarkan target arsitektur V3.1 yang akan kita bangun. Berbeda dengan matriks Section 5.2, di sini seluruh komponen "Otak Kiri" (**AlphaShape**, **MLP**, **Math Kernel**) sudah terhubung sepenuhnya dalam rantai produksi.

| Komponen A               | Komponen B             |  Tipe  | Nama Interaksi     | Input -> Output                                 |
| :----------------------- | :--------------------- | :----: | :----------------- | :---------------------------------------------- |
| **V-JEPA Brain**         | **AlphaShape Wrapper** | `-->`  | Volumetric Mesh    | Persistent Latent -> Point Cloud Mapping        |
| **AlphaShape Wrapper**   | **MathUtils**          | `-->`  | Convex Strategy    | Alpha Search -> $V_{stack}$ (CM³)               |
| **DINOv2 Engine**        | **MLP Regressor**      | `-->`  | Density Extraction | Texture Latents -> $\rho$ (Density)             |
| **MLP Regressor**        | **MathUtils**          | `-->`  | Final Physics      | Predicted $\rho \to$ Volumetric Formulas        |
| **SLM (Physics Oracle)** | **Graph State**        | `-->`  | Physical Constants | Physics Reasoning -> $V_{\mu}$ (Unit Volume)    |
| **MathUtils**            | **Logic Gate**         | `-->`  | Sovereign 2 Truth  | $(V_{stack} \times \rho) / V_{\mu} \to N_{vol}$ |
| **SAM2 (Union Mask)**    | **Fusion Engine**      | `-->`  | Multi-Shield       | Aggregated Masks -> Multi-Object Residue        |
| **Logic Gate**           | **LangGraph**          | `<-->` | Loop Command       | Anomaly Vector <-> Recurse/Converge             |
| **VL-JEPA (Focused)**    | **Logic Gate**         | `<-->` | Direct Discovery   | ROI Crops <-> New Identity                      |
| **Graph State**          | **All Components**     | `<-->` | V3.1 Registry      | GoldenAlpha, Rho, V_unit, MetricScale           |

---

## 3. End-to-End Interaction Flow (The 13-Step Protocol)

### Phase 0: The Anchor (Calibration)

1.  **Input Stream**: Video masuk ke **V-JEPA** & **V2E** (didukung **Event Gen**) secara paralel.
    - _Action_: V-JEPA mulai membangun `LatentTrajectory` dan memori spasial per barang.
    - _Output_: `LatentRepresentation` & `EventSpikes`.
2.  **Intent Discovery**: **LangGraph** meminta **VL-JEPA** (via **VLM Wrapper**) melihat frame awal.
    - _Action_: VL-JEPA melakukan identifikasi semantik awal.
    - _Output_: `CurrentIntent = "Cup"`.
3.  **Physical Lookup**: **LangGraph** query ke **SLM** dan mencatat di **Graph State**. - _Action_: SLM mencari standar fisik: "Gelas = 350ml". - _Output_: `V_unit = 350.0` disimpan di Registry.
    3.5. **Golden Alpha Calibration**: **MathUtils** + **AlphaShape**. - _Action_: Binary Search variabel $\alpha$ sampai $V_{calc\_hull} \approx V_{unit}$ (disebut rekonsiliasi $V_{calc} = V_{target}$). - _Output_: `GoldenAlpha` disimpan sebagai **Physical Signature** untuk verifikasi identitas nanti.

### Phase 1: Orbit & Perception (The Sweep)

4.  **Tracking & Visual Counting**: LangGraph trigger **CountVid** & **ReID Engine**.
    - _Action_: ReID menggunakan V-JEPA Latent Pose untuk menjaga konsistensi ID objek.
    - _Output_: `N_visible`.
5.  **Semantic Masking**: LangGraph trigger **SAM2**.
    - _Action_: SAM2 membuat mask "Hanya Tumpukan" sesuai Intent aktif.
6.  **Depth Mapping**: Mask dikirim ke **DepthEverything**.
    - _Output_: `MetricDepthMap` (dalam satuan CM).
7.  **Sovereignty 3 (Occupancy)**: Depth + RGB ke **Embedding** $\to$ **Clustering** $\to$ **MLP**.
    - _Action_: DINOv2 menganalisis "Visual Turbidity" & "Specular Highlights" (kilatan cahaya) pada RGB untuk memprediksi kepadatan/airgap.
    - _Output_: $\rho$ (Density) yang dipengaruhi oleh interaksi cahaya pada tumpukan.
8.  **3D Reconstruction**: **V-JEPA Memory** + **AlphaShape**.
    - _Action_: Menggunakan _latent displacement_ (geseran kamera) untuk menjahit poin depth dari berbagai angle. `GoldenAlpha` digunakan untuk melakukan simulasi "Bungkusan Rapat" tumpukan agar $V_{stack}$ akurat.
    - _Output_: `V_stack`.

### Phase 2: The Audit (Calculation)

9.  **Math Engine**: Menerima seluruh variabel fisik dari **Graph State**.
    - _Formula_: $N_{vol} = \frac{V_{stack} \times \rho}{V_{unit}}$.
    - _Output_: `N_volumetric`.
10. **Anomaly Check**: **Fusion Engine** membandingkan Mask SAM2 vs V2E Spikes.
    - _Action_: Menghitung `AnomalyScore` berdasarkan event spike yang muncul di luar area masking.
    - _Output_: `AnomalyScore` -> Trigger Discovery Loop jika tinggi.

### Phase 3: The Reconciliation (Strange Loop)

11. **Logic Gate**: Membandingkan 3 Kebenaran (**Sovereign Consensus**).
    - Jika $|N_{vis} - N_{vol}| > 0.5$ **OR** `AnomalyScore` tinggi $\to$ **RECURSE**.
12. **Discovery Loop (Identity Discovery)**:
    - **Trigger**: Logic Gate melihat `AnomalyScore` tinggi.
    - **Action**: **VL-JEPA** fokus ke koordinat anomali. Ia mengecek posisi latent dan membandingkan **Golden Alpha** barang baru vs database. Jika asing, ia membuat intent baru.
    - _Output_: `NewIntent` -> Restart ke Phase 0.
13. **Refinement Loop (Accuracy Refinement)**:
    - **SLM Judge** menganalisis konflik ($N_{vis} \neq N_{vol}$).
    - _Action_: Memerintah update intent spesifik atau mengubah parameter ($\rho$, threshold CountVid).
    - _Effect_: Loop sampai status = "Converged".

---

## 4. Modified, New and Unmodified Components (Brutally Honest Audit)

### 4.1. MODIFIED (Existing Core Upgraded)

| Component           | What is Modified? (Critical Rationale)                                                                               |
| :------------------ | :------------------------------------------------------------------------------------------------------------------- |
| **LangGraph**       | **Loop Logic**: Harus mengintegrasikan alur 13-langkah V3.1. Penambahan node `Discovery` dan `Refinement`.           |
| **LogicGate**       | **Routing Engine**: Sekarang harus bisa me-route anomali langsung ke VL-JEPA (Semboyan: Latent-First).               |
| **MathUtils**       | **V3 Kernel**: Menambahkan integrasi `alphashape` (binary search Golden Alpha) dan rumus 3DC final.                  |
| **SLM (Qwen)**      | **The Oracle**: Prompt engineering diubah total dari "asisten umum" menjadi "pakar fisika & verifikator".            |
| **V-JEPA**          | **Spasial Memory**: Implementasi `PersistentContext` (ctxt tokens) untuk "ingatan" antar frame.                      |
| **VL-JEPA**         | **Focused Attention**: Kemampuan menerima koordinat crop dari LogicGate untuk identifikasi benda asing secara cepat. |
| **SAM2**            | **Multi-Masking**: Harus bisa menggabungkan mask dari berbagai Intent agar tidak bentrok (Union Mask).               |
| **DepthEverything** | **Metric Scaling**: Output di-normalize ke CM berdasarkan kalibrasi Phase 0; bukan lagi angka relatif 0-1.           |
| **Fusion Engine**   | **Multi-Shield**: Harus di-upgrade agar bisa membandingkan gabungan semua mask aktif vs event spikes.                |
| **CountVid**        | **Dynamic Sensitivity**: Harus dimodifikasi agar bisa menerima feedback threshold dari SLM Judge secara real-time.   |
| **ReID Engine**     | **Latent-Pose**: Harus di-upgrade dari IoU-only menjadi V-JEPA Latent feature matching untuk tracking di tumpukan.   |
| **Graph State**     | **V3.1 Schema**: Penambahan field krusial Pydantic (`GoldenAlpha`, `Rho`, `MetricScale`, `V_unit`).                  |
| **Temporal Memory** | **Legacy Replacement**: Dirombak total (atau di-replace) oleh arsitektur Spasial Memory V3.1 berbasis V-JEPA.        |

### 4.2. NEW (To be Created / Wrap)

| Component         | Purpose                                                                                            |
| :---------------- | :------------------------------------------------------------------------------------------------- |
| **AlphaShape**    | Library/Wrapper untuk menghitung volume bungkusan tumpukan dengan parameter **Golden Alpha**.      |
| **DINOv2 Engine** | Backend ekstraksi fitur tekstur (kilatan cahaya/specularities) dari tumpukan (RGB + Depth).        |
| **MLP Regressor** | Jaringan saraf ringan (sklearn) untuk menerjemahkan fitur DINOv2 menjadi angka kepadatan ($\rho$). |

### 4.3. UNMODIFIED (Stable Utilities & Sensors)

| Component            | Rationale                                                                               |
| :------------------- | :-------------------------------------------------------------------------------------- |
| **V2E Engine**       | Sensor event agnostik. Tetap berfungsi sebagai generator spike tanpa perubahan kode.    |
| **Clustering**       | Algoritma grouping SKU berdasarkan embedding visual sudah cukup stabil.                 |
| **Embedding Engine** | Berfungsi sebagai casing DINOv2 regional. Tidak butuh perubahan logika internal.        |
| **VLM Wrapper**      | Bridge low-level untuk Qwen 4-bit sudah stabil di infrastruktur T4 GPU.                 |
| **Event Gen Kernel** | Utilitas pembantu untuk generate visual spikes. Tidak dipengaruhi oleh logika rekursif. |

---

## 5. Current Condition of System (Brutally Honest Audit)

Setelah melakukan audit pada direktori `Implementation/`, berikut adalah realita kondisi sistem kita saat ini dibandingkan dengan V3.1 Spec:

1.  **DINOv2**:
    - **Ada**: Source code library di `Techs/dinov2-main` dan script pengujian `test_dinov2.py`.
    - **TIDAK ADA**: `Dinov2Engine.py` di `v2_logic/models/`. Kita belum punya "Mesin" yang siap dihubungkan ke LangGraph untuk ekstraksi tekstur secara rutin.
2.  **AlphaShape**:
    - **Ada**: Library `alphashape` (pip installed).
    - **TIDAK ADA**: `AlphaHullWrapper.py`. Logika untuk melakukan _Binary Search_ mencari Golden Alpha belum diimplementasikan.
3.  **MLP Regressor**:
    - **TIDAK ADA**: Sama sekali belum ada model `.joblib` atau `.pth` untuk regresi kepadatan ($\rho$), dan belum ada wrapper-nya.
4.  **V-JEPA**:
    - **Ada**: `v_jepa_engine.py` versi dasar.
    - **TIDAK ADA**: Logika `PersistentLatentContext` (Spatio-Temporal Memory). Saat ini V-JEPA hanya bekerja frame-by-frame, belum bisa jadi "Jembatan Ingatan".
5.  **Fusion Engine**:
    - **Ada**: `fusion_engine.py` versi legacy.
    - **Masalah**: Hanya mendukung single-mask. Tidak bisa menangani _Multi-Intent Discovery_ tanpa modifikasi total.
6.  **DepthEverything**:
    - **Ada**: `depth_engine.py`.
    - **Masalah**: Belum memiliki modul Kalibrasi Metric. Angka depth masih bersifat relatif, bukan CM absolut yang dibutuhkan MathUtils.
7.  **CountVid**:
    - **Ada**: `count_vid_engine.py` (Adapter).
    - **Masalah**: Threshold sensitivitas masih statis (`CONF_THRESH = 0.23`). Belum bisa menerima input dinamis dari SLM Judge.
8.  **SAM2**:
    - **Ada**: `segmentation_engine.py`.
    - **Masalah**: Masih bersifat "General Segmentation". Belum ada logika _Intent Filtering_ atau _Union Masking_ untuk mendukung tumpukan benda yang berbeda.
9.  **VL-JEPA**:
    - **Ada**: `vl_jepa_engine.py`.
    - **Masalah**: Belum mendukung "Focused Attention" (crop-based identification). Masih melihat frame secara utuh.
10. **SLM (Qwen)**:
    - **Ada**: `slm_engine.py`.
    - **Masalah**: Prompt masih bersifat generik (Spatial/Volumetric). Belum ada prompt spesifik untuk estimasi $V_{\mu}$ (Physics Oracle).
11. **LogicGate**:
    - **Ada**: `logic_gate.py`.
    - **Masalah**: Logika routing masih linier (Loop ke SLM). Belum punya jalur cepat "Discovery" langsung ke VL-JEPA saat `AnomalyScore` tinggi.
12. **LangGraph**:
    - **Ada**: `recursive_flow.py`.
    - **Masalah**: Struktur graf masih versi V2. Harus dirombak total untuk mengakomodasi workflow 13-langkah V3.1 dan _Global Registry_.
13. **MathUtils**:
    - **Ada**: `math_utils.py` (Basic Hull).
    - **Masalah**: Belum ada kernel untuk **Golden Alpha Binary Search** dan rumus **3DC Volumetric**.
14. **V2E**:
    - **Ada**: `v2e_engine.py` (Agnostic Sensor).
    - **Kondisi**: Aman, tidak perlu modifikasi besar.
15. **ReID Engine**:
    - **Ada**: `reid_engine.py`.
    - **Masalah**: Menggunakan IoU dan CLIP-like features. Belum terintegrasi dengan _Latent Pose_ V-JEPA untuk tracking yang lebih stabil di V3.
16. **Clustering Engine**:
    - **Ada**: `clustering_engine.py`.
    - **Kondisi**: Berfungsi dengan baik untuk grouping SKU berdasarkan embedding.
17. **Temporal Memory (Legacy)**:
    - **Ada**: `temporal_memory.py`.
    - **Kondisi**: Modul V2 yang menggunakan V-JEPA ViT-Huge. Akan di-_deprecated_ atau dirombak total oleh Phase 1 (Spatio-Temporal Memory V3.1).
18. **Graph State (Pydantic)**:
    - **Ada**: `graph_state.py`.
    - **Masalah**: Definisi `GlobalContext` dan `PerceptionState` masih minim. Harus ditambahkan field untuk `GoldenAlpha`, `Rho`, `V_unit`, dan `MetricScale`.
19. **Embedding Engine**:
    - **Ada**: `embedding_engine.py`.
    - **Status**: Sebenarnya menggunakan DINOv2 (meskipun docstring menyebut CLIP). Ini adalah "casing" untuk ekstraksi fitur regional.
20. **VLM Wrapper**:
    - **Ada**: `vlm_wrapper.py`.
    - **Status**: Backend low-level untuk Qwen2-VL (4-bit quantization). Sudah stabil untuk T4 GPU.
21. **Event Gen Kernel**:
    - **Ada**: `v2_logic/kernels/event_gen.py`.
    - **Status**: Utilitas pembantu untuk generate synthetic spikes dari frame gray.

**Sertifikasi Audit**: 100% Seluruh komponen (21 item) di direktori `Implementation/` telah diperiksa.

**Kesimpulan Akhir**: Sistem saat ini memiliki "Panca Indra" yang sudah terhubung, tapi "Otak Kiri" (Logika Volumetrik) dan "Memori Spasial"-nya masih kosong. Kita sedang berada di transisi dari _Object Detection_ menuju _Physical Reality Estimation_.

### 5.2. Schematic Interaction Matrix (Current System Only)

Berikut adalah tabel interaksi antar-komponen yang mencerminkan kondisi **saat ini** di direktori `Implementation/` (berdasarkan `recursive_flow.py` dan `engine_v2.py`), tanpa menyertakan fitur V3.1 yang belum dibangun.

| Komponen A                 | Komponen B            |  Tipe  | Nama Interaksi       | Input -> Output                     |
| :------------------------- | :-------------------- | :----: | :------------------- | :---------------------------------- |
| **LangGraph (Controller)** | **V2E Engine**        | `-->`  | Parallel Trigger     | image_rgb -> spike_map              |
| **Event Gen Kernel**       | **V2E Engine**        | `-->`  | Spike Synthesis      | frame_gray -> synthetic_events      |
| **LangGraph (Controller)** | **V-JEPA**            | `-->`  | Brain Encoding       | image_rgb -> latent_representation  |
| **V-JEPA**                 | **VL-JEPA Director**  | `-->`  | Dynamic Intent       | latent_rep -> intent_list           |
| **VL-JEPA Director**       | **CountVid Engine**   | `-->`  | Parallel Execution   | intent_list -> n_visible            |
| **VL-JEPA Director**       | **SAM2 Engine**       | `-->`  | Parallel Execution   | intent_list -> segmentation_masks   |
| **SAM2 Engine**            | **DepthEverything**   | `-->`  | Depth Fusing         | masks -> depth_map_stats            |
| **V2E Engine**             | **Fusion Engine**     | `-->`  | Spike Audit          | spike_map -> residual_energy        |
| **SAM2 Engine**            | **Fusion Engine**     | `-->`  | Mask Validation      | masks -> explained_spike_blobs      |
| **Fusion Engine**          | **ReID Engine**       | `<-->` | ID Persistence       | bboxes <-> tracked_object_ids       |
| **SAM2 Engine**            | **Embedding Engine**  | `-->`  | Feature Source       | image_crops -> dinov2_features      |
| **Embedding Engine**       | **Clustering Engine** | `-->`  | Grouping             | features -> sku_clusters            |
| **Fusion Engine**          | **Logic Gate**        | `-->`  | Convergence Audit    | n_vis, residual -> gate_action      |
| **Logic Gate**             | **SLM Engine**        | `-->`  | Anomaly Reasoning    | anomaly_context -> hypothesis_text  |
| **SLM Engine**             | **VL-JEPA Director**  | `-->`  | Recursive Feedback   | hypothesis_text -> updated_intent   |
| **VLM Wrapper**            | **VL-JEPA / SLM**     | `<-->` | Model Hosting        | prompt/image <-> 4-bit quant output |
| **MathUtils**              | **Logic Gate**        | `-->`  | Geometric Logic      | n_vis, n_vol -> anomaly_type        |
| **Graph State**            | **All Components**    | `<-->` | Registry Sync        | state_reads <-> state_writes        |
| **Temporal Mem**           | **Fusion Engine**     | `-->`  | Persistence Fallback | historical_features -> peak_count   |
| **AlphaShape**             | **Library (N/A)**     | `---`  | Placeholder          | Belum terhubung ke flow sistem.     |
| **MLP Regressor**          | **Model (N/A)**       | `---`  | Placeholder          | Belum terhubung ke flow sistem.     |
| **DINOv2 Engine**          | **Library (N/A)**     | `---`  | Placeholder          | Digunakan via Embedding Engine.     |

**Catatan Diagnostik**: Saat ini, interaksi antara **SAM2/Depth** dan **CountVid** berjalan secara paralel setelah mendapat "Sinyal Intent" dari Director. Keduanya menyumbangkan angka independen ($N_{vis}$ dan $N_{vol\_range}$) ke Fusion Engine untuk dibandingkan oleh Logic Gate.

---

## 6. Implementation Roadmap (Phase Breakdown)

Berdasarkan audit 21 komponen, berikut adalah roadmap implementasi granular untuk mencapai arsitektur V3.1:

### Phase 1: The Spatio-Temporal Foundation

- [x] **V-JEPA Memory**: Implementasi `PersistentLatentContext` di `vjepa_engine.py`. Fokus pada penyimpanan trajectory laten antar frame.
- [x] **Latent-Pose ReID**: Upgrade `reid_engine.py` untuk menggunakan fitur laten V-JEPA guna menjaga konsistensi ID pada tumpukan padat.

### Phase 2: The Physical Density Engine

- [x] **DINOv2 Engine**: Membangun `dinov2_engine.py` sebagai wrapper untuk ekstraksi fitur tekstur (kilatan cahaya/specularities).
- [x] **Density Predictor**: Inisialisasi dan integrasi `MLP Regressor` (sklearn) untuk menerjemahkan fitur DINOv2 menjadi angka $\rho$ (occupancy).

### Phase 3: The Geometric Kernel

- [x] **Golden Alpha AlphaShape**: Membangun `alphashape_wrapper.py` untuk Binary Search variabel $\alpha$ (rekonsiliasi $V_{calc} = V_{target}$).
- [x] **3DC Volumetric Math**: Update `math_utils.py` dengan rumus final $(V_{stack} \times \rho) / V_{\mu}$ dan integrasi dengan AlphaHull.

### Phase 4: Perception & Validation Upgrades

- [x] **Multi-Intent SAM2**: Implementasi _Union Masking_ di `segmentation_engine.py` agar bisa menghitung tumpukan barang berbeda secara simultan.
- [x] **Multi-Shield Fusion**: Upgrade `fusion_engine.py` untuk membandingkan akumulasi mask vs event spikes.
- [x] **Dynamic CountVid**: Modul sensitivitas adaptif di `count_vid_engine.py` yang menerima feedback dari SLM Judge.

### Phase 5: The 13-Step Orchestration

- [ ] **V3.1 Graph State**: Ekspansi schema Pydantic di `graph_state.py` (Registry untuk GoldenAlpha, Rho, dll).
- [ ] **Recursive Flow Overhaul**: Rombak total `recursive_flow.py` untuk mengaktifkan 13-langkah protokol dan jalur cepat Discovery.

### Phase 6: Deployment & Convergence Test

- [ ] **3-Object Triangulation**: Uji coba pada kasus tumpukan 3 objek untuk memverifikasi konvergensi antara $N_{vis}$ dan $N_{vol}$.
- [ ] **Legacy Cleanup**: Deprecated `temporal_memory.py` dan kode V2 yang sudah tidak relevan.
