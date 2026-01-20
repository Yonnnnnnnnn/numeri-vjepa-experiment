"""
Main Inference Engine (Segmentation Pipeline)

Handles Video I/O, SAM Segmentation, CLIP Embedding, K-Means Clustering,
V-JEPA Temporal Memory, and Table Visualization.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : InferenceEngine (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <run_inference_loop>  → Main processing function                         │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <SegmentationEngine>  ← from v2_logic.models.segmentation_engine          │
  │  <EmbeddingEngine>     ← from v2_logic.models.embedding_engine             │
  │  <ClusteringEngine>    ← from v2_logic.models.clustering_engine            │
  │  <TemporalMemory>      ← from v2_logic.models.temporal_memory              │
  │  <VLMInferenceModel>   ← Qwen2.5 for labeling (optional)                  │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, int, float, numpy.ndarray, torch.Tensor

Production Rules:
  run_inference_loop → setup + loop(segment + embed + cluster + track) + output
═══════════════════════════════════════════════════════════════════════════════

Pattern: Pipeline
- Sequential processing with multiple stages
"""

# pylint: disable=no-member
# pylint: disable=import-error

import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass

from v2_logic.kernels.event_gen import rgb_to_event_triton
from v2_logic.models.segmentation_engine import SegmentationEngine
from v2_logic.models.embedding_engine import EmbeddingEngine
from v2_logic.models.clustering_engine import ClusteringEngine, ClusterInfo
from v2_logic.models.temporal_memory import TemporalMemory


@dataclass
class PerCategoryCount:
    """Count per item category with temporal tracking."""

    label: str
    visible: int
    from_memory: int
    total: int


from v2_logic.models.temporal_memory import (
    TemporalMemory,
    load_vjepa_model,
    extract_frame_features,
)


class InventoryTracker:
    """Tracks per-category counts across frames using V-JEPA."""

    def __init__(self, device="cuda"):
        self.device = device
        self.memories: Dict[str, TemporalMemory] = {}
        self.vjepa_model = load_vjepa_model(device)
        self.all_labels: set = set()

    def update(
        self, clusters: List[ClusterInfo], frame_tensor: torch.Tensor
    ) -> List[PerCategoryCount]:
        """Update counts from current frame's clusters using V-JEPA features."""
        # 1. Extract V-JEPA features for the current frame (once for all categories)
        features = extract_frame_features(self.vjepa_model, frame_tensor)

        # 2. Get current visible counts
        current_counts = {}
        for cluster in clusters:
            label = (
                cluster.label
                if cluster.label != "Unknown"
                else f"Type {cluster.cluster_id}"
            )
            current_counts[label] = current_counts.get(label, 0) + cluster.count
            self.all_labels.add(label)

        # 3. Update memory for ALL known labels
        results = []
        for label in sorted(self.all_labels):
            visible = current_counts.get(label, 0)

            # Get or create memory for this label
            if label not in self.memories:
                self.memories[label] = TemporalMemory()

            # Update memory state
            ctx = self.memories[label].update(features, visible)

            # Create result
            results.append(
                PerCategoryCount(
                    label=label,
                    visible=visible,
                    from_memory=ctx.occluded_estimate,  # Intelligent occlusion estimate
                    total=(
                        ctx.peak_count
                        if ctx.feature_similarity >= 0.7
                        else (visible + ctx.occluded_estimate)
                    ),
                    # Note: Total is usually visible + occluded.
                    # If scene is stable, we trust peak. If scene changed, we trust decaying peak.
                    # Simplified: peak_count is usually the robust history.
                    # But if items are removed, we should show the decayed version.
                    # Let's use: visible + occluded (which is derived from effective_peak)
                )
            )

        return results


def draw_table(
    image: np.ndarray,
    counts: List[PerCategoryCount],
    start_pos: tuple = (10, 30),
    row_height: int = 25,
    col_widths: tuple = (150, 60, 60, 60),
) -> np.ndarray:
    """Draw a count table on the image."""
    output = image.copy()
    x, y = start_pos

    # Table colors
    header_color = (50, 50, 50)
    bg_color = (30, 30, 30)
    text_color = (0, 255, 0)
    border_color = (100, 100, 100)

    # Calculate table dimensions
    total_width = sum(col_widths)
    total_height = (len(counts) + 2) * row_height  # +2 for header and total

    # Draw background
    cv2.rectangle(output, (x, y), (x + total_width, y + total_height), bg_color, -1)
    cv2.rectangle(output, (x, y), (x + total_width, y + total_height), border_color, 1)

    # Draw header
    headers = ["Item", "Visible", "Memory", "Total"]
    cx = x
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        cv2.rectangle(output, (cx, y), (cx + width, y + row_height), header_color, -1)
        cv2.putText(
            output,
            header,
            (cx + 5, y + row_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
        )
        cx += width

    # Draw rows
    for row_idx, item in enumerate(counts):
        ry = y + (row_idx + 1) * row_height
        cx = x

        values = [
            item.label[:15],  # Truncate label
            str(item.visible),
            str(item.from_memory),
            str(item.total),
        ]

        for val, width in zip(values, col_widths):
            cv2.putText(
                output,
                val,
                (cx + 5, ry + row_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
            cx += width

    # Draw total row
    total_y = y + (len(counts) + 1) * row_height
    cv2.rectangle(
        output, (x, total_y), (x + total_width, total_y + row_height), header_color, -1
    )

    total_visible = sum(c.visible for c in counts)
    total_memory = sum(c.from_memory for c in counts)
    total_all = sum(c.total for c in counts)

    totals = ["TOTAL", str(total_visible), str(total_memory), str(total_all)]
    cx = x
    for val, width in zip(totals, col_widths):
        cv2.putText(
            output,
            val,
            (cx + 5, total_y + row_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )
        cx += width

    return output


def run_inference_loop(
    video_path: str,
    output_path: str,
    threshold: float = 0.1,
    segment_interval: int = 10,
    sam_checkpoint: Optional[str] = None,
):
    """
    Main inference loop with SAM + CLIP + Clustering.

    Args:
        video_path: Path to input video
        output_path: Path to output video
        threshold: Event detection threshold
        segment_interval: Run segmentation every N frames
        sam_checkpoint: Path to SAM2 checkpoint (required for segmentation)
    """
    print("=" * 60)
    print(" INFERENCE VISUALIZER: SAM + CLIP + V-JEPA")
    print("=" * 60)

    # 1. Setup Video I/O
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

    # 2. Setup Components
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/4] Loading Segmentation Engine (SAM2)...")
    segmenter = SegmentationEngine(checkpoint_path=sam_checkpoint, device=device)

    print(f"[2/4] Loading Embedding Engine (DINOv2)...")
    embedder = EmbeddingEngine(device=device)

    print(f"[3/4] Initializing Clustering Engine (DBSCAN)...")
    # Tuning: eps=0.7 (merged), min_samples=5 (less noise/fragmentation)
    clusterer = ClusteringEngine(eps=0.7, min_samples=5)

    print(f"[4/4] Initializing Inventory Tracker...")
    tracker = InventoryTracker()

    print("=" * 60)
    print("All components loaded. Starting processing...")

    # 3. Processing Loop
    prev_frame_tensor = None
    last_counts: List[PerCategoryCount] = []
    last_seg_result = None  # Persistent across frames for visualization
    pbar = tqdm(total=total_frames, desc="Processing")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = (
            torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(device)

        if prev_frame_tensor is None:
            prev_frame_tensor = frame_tensor
            pbar.update(1)
            continue

        # A. Generate Events (for visualization)
        event_tensor = rgb_to_event_triton(frame_tensor, prev_frame_tensor, threshold)

        # B. Segmentation + Clustering (every N frames)
        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame_idx % segment_interval == 0:
            # Segment
            seg_result = segmenter.segment_frame(frame_rgb)
            last_seg_result = seg_result  # Keep reference for mask visualization

            if seg_result.num_segments > 0:
                # Get cropped regions
                crops = seg_result.get_cropped_regions(frame_rgb)

                # Embed crops
                embeddings = embedder.embed_regions(crops)

                # Cluster
                labels, clusters = clusterer.fit_predict(embeddings)

                # Update tracker with V-JEPA/frame features
                last_counts = tracker.update(clusters, frame_tensor)

        # C. Visualization
        # Create mask overlay on original frame
        frame_with_masks = frame.copy()

        if last_seg_result is not None and last_seg_result.num_segments > 0:
            # Generate distinct colors for each mask
            np.random.seed(42)  # Consistent colors across frames
            colors = [
                tuple(np.random.randint(50, 255, 3).tolist())
                for _ in range(last_seg_result.num_segments)
            ]

            # Overlay each mask with transparency
            for idx, (mask, box, score) in enumerate(
                zip(
                    last_seg_result.masks, last_seg_result.boxes, last_seg_result.scores
                )
            ):
                color = colors[idx % len(colors)]

                # Create colored mask overlay
                mask_overlay = np.zeros_like(frame)
                mask_overlay[mask] = color

                # Blend with alpha
                alpha = 0.4
                frame_with_masks = cv2.addWeighted(
                    frame_with_masks, 1.0, mask_overlay, alpha, 0
                )

                # Draw bounding box
                x1, y1, x2, y2 = box
                cv2.rectangle(frame_with_masks, (x1, y1), (x2, y2), color, 2)

                # Draw segment ID and score
                label_text = f"#{idx} ({score:.2f})"
                cv2.putText(
                    frame_with_masks,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

            # Draw segment count
            cv2.putText(
                frame_with_masks,
                f"Segments: {last_seg_result.num_segments}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Draw ROI boundary (focus area) - always visible
        roi_margin = 0.15  # Match segmentation_engine default
        roi_x1 = int(width * roi_margin)
        roi_y1 = int(height * roi_margin)
        roi_x2 = int(width * (1 - roi_margin))
        roi_y2 = int(height * (1 - roi_margin))
        cv2.rectangle(
            frame_with_masks, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2
        )
        cv2.putText(
            frame_with_masks,
            "FOCUS AREA",
            (roi_x1 + 5, roi_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        # Event visualization (right panel)
        event_vis = torch.zeros_like(frame_tensor)
        event_vis[event_tensor > 0] = 1.0
        event_vis[event_tensor < 0] = 0.5

        event_img = event_vis.squeeze(0).permute(1, 2, 0).cpu().numpy()
        event_img = (event_img * 255).astype(np.uint8)

        # Concatenate side-by-side: left=masks, right=events
        combined_frame = np.hstack((frame_with_masks, event_img))

        # Draw table overlay
        if last_counts:
            combined_frame = draw_table(combined_frame, last_counts)

        out_writer.write(combined_frame)

        prev_frame_tensor = frame_tensor
        pbar.update(1)

    # Final Stats
    pbar.close()
    print("=" * 60)
    print("Final Inventory Count:")
    for item in last_counts:
        print(
            f"  {item.label}: {item.total} ({item.visible} visible + {item.from_memory} memory)"
        )
    print("=" * 60)

    cap.release()
    out_writer.release()
    print(f"Processing Complete. Output saved to {output_path}")


if __name__ == "__main__":
    pass
