# Test Strategy: Vision & Loop Verification

This document outlines the strategy for verifying the Antigravity V2 pipeline.

## 1. Unit Testing

- **VJEPAEngine**: Verify weight loading, encoding shape, and prediction consistency.
- **Preprocessing**: Test spike-to-frame conversion and normalization.

## 2. Integration Testing

- **Recursive Flow**: Verify end-to-end execution from raw video to final count using `run_recursive_system.py`.
- **Hybrid Decision Gate**: Validate that Logic Gate correctly routes frames based on spatial (V2E) and volumetric (SAM2+Depth) anomalies.
- **V-JEPA Memory**: Verify that temporal latents provide consistent context for the countvid_executor_node.

## 4. Runtime Debugging & Stability

- **Persistence Verification**: Test the `temporal_filter_fast` mechanism to ensure object IDs are correctly tracked across window `w`.
- **Multiprocessing**: Verify cross-platform compatibility (Windows spawn vs Unix fork) for object persistence checks by ensuring clean state passing in `starmap_args`.
- **Traceability**: Utilize full traceback logging in `count_vid_engine.py` to diagnose device-specific serialization errors (e.g., `BatchEncoding.to()` failures).
- **Bootstrap Robustness Testing**: Memverifikasi bahwa seluruh _Logical Graph_ tetap mencapai node akhir (`exit`) meskipun subsistem sekunder (seperti `SegmentationEngine`) gagal dimuat (Graceful Degradation).
- **Dispatcher Diagnostics**: Memantau log `[CountVid Patch]` selama inisialisasi untuk memvalidasi apakah sistem menggunakan jalur _Modern_, _Positional_, atau _Nuclear Fallback_.
