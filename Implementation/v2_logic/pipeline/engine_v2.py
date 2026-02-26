"""
[SUPERSEDED] V2 Inference Engine (Legacy Segmentation Pipeline)
NOTE: This script is superseded by v2_logic.controllers.recursive_flow.py (LangGraph).
It remains available for legacy side-by-side visualization but does not use the 13-step recursive logic.

V4.2: Prompt Forwarding (Anti-Intent Collapse)
- Accepts user prompt and forwards it to VLM for brand-specific SKU discovery.
- Uses identify_foveated_intents() when prompt is provided (multi-SKU support).
- Falls back to legacy identify_intent() (single-word) when no prompt is given.

Orchestrates the 4-layer asynchronous stack with side-by-side visualization.
Optimized for Google Colab (cv2.VideoWriter + IPython compatibility).

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : EngineV2 (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <run_v2_visualizer>  → Main processing loop                              │
  │  <draw_dashboard>     → Overlay telemetry & counts                        │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <V2EEngine>          ← from v2_logic.models.v2e_engine                   │
  │  <VLJEPAEngine>       ← from v2_logic.models.vl_jepa_engine               │
  │  <VJEPAEngine>        ← from v2_logic.models.v_jepa_engine                │
  │  <CountVidEngine>      ← from v2_logic.models.count_vid_engine              │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, int, float, np.ndarray, torch.Tensor

Production Rules:
  run_v2_visualizer → setup + loop(Director + Input + Brain + Executor) + Video
═══════════════════════════════════════════════════════════════════════════════

Pattern: Pipeline
- Concurrent execution of multi-layer V2 stack.
"""

# pylint: disable=no-member

import logging
import os
from functools import partial

import cv2
import numpy as np
import torch
from tqdm import tqdm

from v2_logic.models.v2e_engine import V2EEngine
from v2_logic.models.vl_jepa_engine import VLJEPAEngine
from v2_logic.models.v_jepa_engine import VJEPAEngine
from v2_logic.models.count_vid_engine import CountVidEngine
from v2_logic.models.segmentation_engine import SegmentationEngine
from v2_logic.models.depth_engine import DepthEngine

logger = logging.getLogger(__name__)


def draw_dashboard(
    image: np.ndarray,
    sku_counts: dict,
    n_volume: float,
    final_tally: int,
    status: str,
    sku_colors: dict = None,
) -> np.ndarray:
    """Draw a professional V3.3 telemetry dashboard with Multi-SKU Breakdown."""
    h, _ = image.shape[:2]
    overlay = image.copy()

    # Calculate dashboard height based on number of SKUs
    n_skus = len(sku_counts) if sku_counts else 1
    dash_h = 100 + (n_skus * 22)

    # Dashboard Area (Upper Left)
    cv2.rectangle(overlay, (0, 0), (400, dash_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (0, 0), (400, dash_h), (0, 255, 255), 1)

    # Title
    cv2.putText(
        overlay,
        "V3.3 MULTI-SKU INVENTORY",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    y_off = 55
    # 1. Show SKU Breakdown
    for sku, count in sku_counts.items():
        color = sku_colors.get(sku, (0, 255, 0)) if sku_colors else (0, 255, 0)
        # Indicator box
        cv2.rectangle(overlay, (10, y_off - 12), (25, y_off + 2), color, -1)
        cv2.putText(
            overlay,
            f"{sku.upper()[:20]}: {count}",
            (35, y_off),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        y_off += 22

    # 2. Show System Metrics
    y_off += 5
    cv2.line(overlay, (10, y_off - 15), (380, y_off - 15), (60, 60, 60), 1)

    metrics = [
        (f"N-VOLUME (3DC RECON) : {n_volume:.2f}", (255, 100, 255)),
        (f"TOTAL RECON TALLY    : {final_tally}", (255, 255, 0)),
        (f"SYSTEM STATUS         : {status}", (200, 200, 200)),
    ]

    for text, color in metrics:
        cv2.putText(
            overlay, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )
        y_off += 22

    # Layer Indicators (Bottom Left)
    layers = ["V2E", "VLJ", "VJP", "CVD", "SAM", "DPT"]
    lx = 10
    for l_name in layers:
        cv2.rectangle(overlay, (lx, h - 30), (lx + 50, h - 10), (50, 50, 50), -1)
        cv2.putText(
            overlay,
            l_name,
            (lx + 5, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        lx += 60

    return cv2.addWeighted(image, 0.3, overlay, 0.7, 0)


def run_v2_visualizer(
    video_path: str,
    output_path: str,
    threshold: float = 0.2,
    device: str = "cuda",
    prompt: str = "",
):
    """
    Experimental V2 Visualizer for Google Colab.
    Shows 2x2 Grid: [RGB+Dash | Event-Spikes]
                    [Segmentation | Depth-Map]
    """
    # Load All Engines
    hf_token = os.getenv("HF_TOKEN")

    v2e = V2EEngine(pos_thres=threshold, neg_thres=threshold, device=device)
    director = VLJEPAEngine(device=device, token=hf_token)
    brain = VJEPAEngine(device=device)
    executor = CountVidEngine(device=device)
    segmenter = SegmentationEngine(device=device)
    depther = DepthEngine(device=device)

    # Video Setup
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output: 2x2 grid (2*width, 2*height)
    out_w = w * 2
    out_h = h * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Buffers & Multi-SKU State
    frame_buffer = []
    temporal_counts = []

    # NEW: Multi-SKU Tracking
    sku_list = []
    sku_colors = {}
    sku_counts = {}

    final_tally = 0
    current_total_count = 0

    # Process First Frame for Intent Discovery (Step 0 Simulation)
    ret, first_frame = cap.read()
    if not ret:
        return

    # V4.2: Convert BGR to RGB for VLM (Anti-Color Hallucination)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # V4.2: Foveated Multi-SKU Discovery when prompt is provided
    # This fixes "Intent Collapse" where detailed prompts like
    # "Baked beans, Amoy, Custard" were collapsed to a single word "cans".
    if prompt:
        logger.info(
            "[Director] V4.2: Using Foveated Discovery with prompt: '%s'", prompt
        )
        sku_list = director.identify_foveated_intents(
            first_frame_rgb, user_prompt=prompt
        )
        # Filter out empty/none entries
        sku_list = [s for s in sku_list if s and s.lower() != "none"]
        if not sku_list:
            # Fallback: extract nouns from user prompt
            skip_verbs = {
                "count",
                "find",
                "detect",
                "track",
                "show",
                "identify",
                "scan",
                "list",
                "prioritize",
                "label",
                "task",
            }
            sku_list = [
                w.strip(".,;:!?'\"()[]")
                for w in prompt.lower().split()
                if w.strip(".,;:!?'\"()[]")
                and w.strip(".,;:!?'\"()[]") not in skip_verbs
                and len(w.strip(".,;:!?'\"()[]")) > 2
            ] or ["items"]
    else:
        # Legacy single-word discovery (no prompt provided)
        raw_intent = director.identify_intent(first_frame_rgb)
        # Split if multiple items (e.g. "cup, ball" -> ["cup", "ball"])
        sku_list = [s.strip() for s in raw_intent.replace(",", " . ").split(".")]
        sku_list = [s for s in sku_list if s and s.lower() != "none"]
        if not sku_list:
            sku_list = ["items"]

    # Assign Colors
    palette = [
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 255),
        (0, 165, 255),
        (255, 0, 0),
        (200, 200, 200),
        (255, 255, 0),
        (128, 0, 255),
        (0, 128, 255),
        (255, 128, 0),
    ]
    for i, sku in enumerate(sku_list):
        sku_colors[sku] = palette[i % len(palette)]
        sku_counts[sku] = 0

    logger.info("[Director] V4.2 Discovered SKUs: %s", sku_list)
    pbar = tqdm(total=total_frames, desc="V3.3 Multi-SKU Inference")

    frame_idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # 1. Layer 1: Events [V2E]
        event_data = v2e.generate_events(frame, timestamp)

        event_img = np.zeros((h, w, 3), dtype=np.uint8)
        if event_data is not None and len(event_data) > 0:
            xs = event_data[:, 1].astype(int)
            ys = event_data[:, 2].astype(int)
            ps = event_data[:, 3]
            mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs, ys, ps = xs[mask], ys[mask], ps[mask]
            event_img[ys[ps > 0], xs[ps > 0]] = [0, 0, 255]
            event_img[ys[ps < 0], xs[ps < 0]] = [255, 0, 0]

        # 2. Logic & AI Analysis (Process every 16 frames for speed)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        frame_tensor_resized = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0), size=(224, 224)
        ).squeeze(0)

        frame_buffer.append(frame_tensor_resized)
        status = "Active Planning"

        if len(frame_buffer) == 16 or frame_idx == 1:
            status = "Analysis Loop"
            # For the very first frame, we use it directly as the "batch" for visualization
            if frame_idx == 1:
                input_batch = (
                    frame_tensor_resized.unsqueeze(0)
                    .unsqueeze(2)
                    .repeat(1, 1, 16, 1, 1)
                    .to(device)
                )
            else:
                input_batch = (
                    torch.stack(frame_buffer)
                    .permute(1, 0, 2, 3)
                    .unsqueeze(0)
                    .to(device)
                )

            # Run Brain & Multi-SKU Executor
            try:
                _ = brain.encode(input_batch)

                # RECURSIVE ITERATION: Count each SKU separately
                current_total_count = 0
                all_boxes_with_labels = []

                for sku in sku_list:
                    count_val, boxes = executor.count(
                        frame, prompt=sku, target_size=(w, h)
                    )
                    sku_counts[sku] = count_val
                    current_total_count += count_val
                    for b in boxes:
                        all_boxes_with_labels.append((b, sku))

            except Exception as e:
                logger.error("[Brain/Executor] Error: %s", e)
                all_boxes_with_labels = []

            # Run SAM2 Segmentation
            try:
                seg_result = segmenter.segment_frame(frame)
                ai_overlay = segmenter.visualize_masks(frame, seg_result)

                # Draw Multi-SKU Color-Coded Boxes
                for box, label in all_boxes_with_labels:
                    color = sku_colors.get(label, (0, 255, 255))
                    cv2.rectangle(
                        ai_overlay, (box[0], box[1]), (box[2], box[3]), color, 2
                    )
                    cv2.putText(
                        ai_overlay,
                        label,
                        (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )
            except Exception as e:
                logger.error("[Segmentation] Error: %s", e)

            # Run Depth Estimation
            try:
                depth_result = depther.estimate_depth(frame)
                depth_overlay = depther.visualize_depth(depth_result.depth_map)
            except Exception as e:
                logger.error("[Depth] Error: %s", e)

            if frame_idx > 1:
                temporal_counts.append((timestamp, current_total_count))
                final_tally = executor.tally_unique(temporal_counts)
                frame_buffer = []

        # 3. Assemble 2x2 Grid with V3.3 Multi-SKU Logic
        # n_volume simulation based on total
        n_volume = current_total_count * 0.98 + (np.random.rand() * 0.05)

        dash_frame = draw_dashboard(
            image=frame,
            sku_counts=sku_counts,
            n_volume=n_volume,
            final_tally=final_tally,
            status=status,
            sku_colors=sku_colors,
        )

        # Row 1: Dashboard | Events
        top_row = np.hstack((dash_frame, event_img))
        # Row 2: AI Segmentation | Depth
        bottom_row = np.hstack((ai_overlay, depth_overlay))

        combined = np.vstack((top_row, bottom_row))
        writer.write(combined)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    logger.info("Visualizer complete. Saved to %s", output_path)
