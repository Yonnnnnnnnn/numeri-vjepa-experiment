"""
Math Utilities Module

Utility class for mathematical operations supporting State Estimation framework.
Organized by their role in the Grand Equation: P(S_t | O_{1:t})

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : MathUtils (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <MathUtils>  → Static utility class with mathematical operations        │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <numpy>       ← from numpy (Array operations)                           │
  │  <ConvexHull>  ← from scipy.spatial (Volume calculations)                │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : numpy.ndarray, float, int, Dict, List, Tuple

Production Rules:
  MathUtils → static methods for similarity, state prediction, volume, IoU, etc.
═══════════════════════════════════════════════════════════════════════════════

Pattern: Utility
- Collection of pure static functions with no internal state.
- Used across multiple components for mathematical computations.
"""

import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module
from typing import Dict, List, Tuple, Optional


class MathUtils:
    """
    Utility class for mathematical operations supporting State Estimation framework.
    Organized by their role in the Grand Equation: P(S_t | O_{1:t})
    """

    # =========================================================================
    # For P(S_t | S_{t-1}) - State Transition Probability
    # =========================================================================

    @staticmethod
    def calculate_vector_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors for object re-identification.
        Higher values mean more similar vectors (same object).

        Args:
            vec1: First feature vector
            vec2: Second feature vector

        Returns:
            Cosine similarity score (0-1)
        """
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def state_prediction(
        previous_state: Dict, motion_vector: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Predict next state from previous state using motion vector.

        Args:
            previous_state: Previous state dictionary with positions and counts
            motion_vector: Optional motion vector for object tracking

        Returns:
            Predicted next state
        """
        predicted_state = previous_state.copy()

        # Update positions based on motion vector if available
        if motion_vector is not None and "positions" in previous_state:
            predicted_positions = []
            for pos in previous_state["positions"]:
                predicted_pos = np.array(pos) + motion_vector
                predicted_positions.append(predicted_pos.tolist())
            predicted_state["positions"] = predicted_positions

        return predicted_state

    # =========================================================================
    # For P(O_t | S_t) - Observation Likelihood
    # =========================================================================

    @staticmethod
    def calculate_convex_hull_volume(points: List[List[float]]) -> float:
        """
        Calculate volume of convex hull from 3D points for occluded object estimation.

        Args:
            points: List of 3D points

        Returns:
            Volume of convex hull
        """
        if len(points) < 4:
            return 0.0
        try:
            hull = ConvexHull(points)
            return hull.volume
        except:
            return 0.0

    @staticmethod
    def calculate_detection_density(
        bboxes: List[Dict], frame_size: Tuple[int, int]
    ) -> float:
        """
        Calculate detection density (ratio of detected area to frame area).

        Args:
            bboxes: List of bounding boxes [{'x': ..., 'y': ..., 'w': ..., 'h': ...}]
            frame_size: Tuple of (width, height)

        Returns:
            Detection density (0-1)
        """
        frame_area = frame_size[0] * frame_size[1]
        if frame_area == 0:
            return 0.0

        detected_area = 0
        for bbox in bboxes:
            detected_area += bbox["w"] * bbox["h"]

        return min(1.0, detected_area / frame_area)

    @staticmethod
    def calculate_riemann_sum(depth_map: np.ndarray, resolution: float) -> float:
        """
        Calculate volume using Riemann sum from depth map for granular materials.

        Args:
            depth_map: 2D array of depth values
            resolution: Resolution of each pixel in real-world units

        Returns:
            Estimated volume
        """
        return np.sum(depth_map) * (resolution**2)

    @staticmethod
    def matrix_transformation(
        local_coords: List[float], transform_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Transform local coordinates to global using transformation matrix.

        Args:
            local_coords: Local 2D/3D coordinates
            transform_matrix: Transformation matrix (4x4 for 3D, 3x3 for 2D)

        Returns:
            Transformed global coordinates
        """
        # Convert to homogeneous coordinates
        if len(local_coords) == 2:
            homo_coords = np.append(local_coords, 1)
        elif len(local_coords) == 3:
            homo_coords = np.append(local_coords, 1)
        else:
            raise ValueError("Local coordinates must be 2D or 3D")

        # Apply transformation
        transformed = np.dot(transform_matrix, homo_coords)

        # Convert back to regular coordinates
        return (
            transformed[:-1] / transformed[-1]
            if transformed[-1] != 0
            else transformed[:-1]
        )

    # =========================================================================
    # For Bayesian Update - Core of Grand Equation
    # =========================================================================

    @staticmethod
    def bayesian_update(prior: float, likelihood: float, evidence: float) -> float:
        """
        Implement Bayes' theorem for state estimation:
        P(S_t | O_t) = (P(O_t | S_t) * P(S_t)) / P(O_t)

        Args:
            prior: Prior probability P(S_t)
            likelihood: Likelihood P(O_t | S_t)
            evidence: Evidence P(O_t)

        Returns:
            Posterior probability P(S_t | O_t)
        """
        if evidence == 0:
            return 0.0
        return (likelihood * prior) / evidence

    @staticmethod
    def calculate_evidence(likelihoods: List[float], priors: List[float]) -> float:
        """
        Calculate evidence P(O_t) = sum(P(O_t | S_i) * P(S_i)) for all possible states S_i.

        Args:
            likelihoods: List of likelihoods P(O_t | S_i) for all states
            priors: List of priors P(S_i) for all states

        Returns:
            Evidence P(O_t)
        """
        if len(likelihoods) != len(priors):
            raise ValueError("Likelihoods and priors must have the same length")

        evidence = 0.0
        for lh, pr in zip(likelihoods, priors):
            evidence += lh * pr

        return evidence

    # =========================================================================
    # For State Consistency - Invariance Preservation
    # =========================================================================

    @staticmethod
    def check_state_consistency(
        current_state: Dict, previous_state: Dict, invariants: List[str] = None
    ) -> float:
        """
        Check if current state is consistent with previous state based on invariants.

        Args:
            current_state: Current state dictionary
            previous_state: Previous state dictionary
            invariants: List of invariants to check (e.g., 'count', 'volume', 'identity')

        Returns:
            Consistency score (0-1), higher means more consistent
        """
        if invariants is None:
            invariants = ["count", "volume", "identity"]

        consistency_scores = []

        # Check count invariance
        if (
            "count" in invariants
            and "count" in current_state
            and "count" in previous_state
        ):
            count_diff = abs(current_state["count"] - previous_state["count"])
            # Allow small count differences (occlusion changes)
            count_consistency = max(
                0.0,
                1.0
                - (
                    count_diff / max(current_state["count"], previous_state["count"], 1)
                ),
            )
            consistency_scores.append(count_consistency)

        # Check volume invariance
        if "volume" in invariants:
            current_vol = (
                current_state.get("mathematical_context", {})
                .get("convex_hull", {})
                .get("volume", 0.0)
            )
            previous_vol = (
                previous_state.get("mathematical_context", {})
                .get("convex_hull", {})
                .get("volume", 0.0)
            )

            if max(current_vol, previous_vol) > 0:
                vol_diff = abs(current_vol - previous_vol)
                vol_consistency = max(
                    0.0, 1.0 - (vol_diff / max(current_vol, previous_vol))
                )
                consistency_scores.append(vol_consistency)

        # Check identity invariance (using vector similarity)
        if "identity" in invariants:
            current_fps = (
                current_state.get("mathematical_context", {})
                .get("vector_features", {})
                .get("object_fingerprints", [])
            )
            previous_fps = (
                previous_state.get("mathematical_context", {})
                .get("vector_features", {})
                .get("object_fingerprints", [])
            )

            if current_fps and previous_fps:
                # Calculate average similarity between fingerprints
                similarities = []
                for cf in current_fps:
                    for pf in previous_fps:
                        sim = MathUtils.calculate_vector_similarity(
                            np.array(cf["features"]), np.array(pf["features"])
                        )
                        similarities.append(sim)

                if similarities:
                    identity_consistency = np.mean(similarities)
                    consistency_scores.append(identity_consistency)

        return np.mean(consistency_scores) if consistency_scores else 1.0

    @staticmethod
    def calculate_packing_density(
        convex_hull_volume: float, single_object_volume: float, total_count: int
    ) -> float:
        """
        Calculate packing density for occluded object estimation.

        Args:
            convex_hull_volume: Volume of convex hull containing all objects
            single_object_volume: Average volume of a single object
            total_count: Total number of objects

        Returns:
            Packing density (0-1)
        """
        if single_object_volume == 0 or total_count == 0:
            return 0.0

        total_object_volume = single_object_volume * total_count
        return (
            min(1.0, total_object_volume / convex_hull_volume)
            if convex_hull_volume > 0
            else 0.0
        )

    @staticmethod
    def calculate_anomaly_score(spike_energy: float, detection_density: float) -> float:
        """
        Calculate anomaly score by comparing spike energy with detection density.
        High score means inconsistent observations (possible new objects or occlusions).

        Args:
            spike_energy: Energy from V2E spike events
            detection_density: Density of detected objects

        Returns:
            Anomaly score (0-1)
        """
        denominator = max(spike_energy, detection_density, 1e-6)
        return abs(spike_energy - detection_density) / denominator

    # =========================================================================
    # For Fisher's Linear Discriminant - Brand Differentiation
    # =========================================================================

    @staticmethod
    def calculate_fisher_discriminant(
        class1_features: List[np.ndarray], class2_features: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate Fisher's Linear Discriminant for brand differentiation.

        Args:
            class1_features: Features from class 1 (e.g., Coca-Cola bottles)
            class2_features: Features from class 2 (e.g., Pepsi bottles)

        Returns:
            Fisher's discriminant vector
        """
        # Convert to numpy arrays
        X1 = np.array(class1_features)
        X2 = np.array(class2_features)

        # Calculate means
        m1 = np.mean(X1, axis=0)
        m2 = np.mean(X2, axis=0)

        # Calculate within-class scatter matrices
        S1 = np.cov(X1.T) * (len(X1) - 1)
        S2 = np.cov(X2.T) * (len(X2) - 1)
        Sw = S1 + S2

        # Calculate between-class scatter matrix
        Sb = np.outer((m2 - m1), (m2 - m1))

        # Calculate discriminant vector
        try:
            Sw_inv = np.linalg.inv(Sw)
            w = Sw_inv.dot(m2 - m1)
            return w / np.linalg.norm(w)  # Normalize
        except np.linalg.LinAlgError:
            # If Sw is singular, use pseudoinverse
            Sw_pinv = np.linalg.pinv(Sw)
            w = Sw_pinv.dot(m2 - m1)
            return w / np.linalg.norm(w) if np.linalg.norm(w) > 0 else np.zeros_like(m1)

    # =========================================================================
    # For Dimensional Analysis - Unit Estimation
    # =========================================================================

    @staticmethod
    def calculate_dimensions(bboxes: List[Dict], camera_matrix: np.ndarray) -> Dict:
        """
        Calculate real-world dimensions from bounding boxes and camera matrix.

        Args:
            bboxes: List of bounding boxes with 3D coordinates
            camera_matrix: Camera intrinsic matrix

        Returns:
            Dictionary with estimated dimensions
        """
        if not bboxes:
            return {"length": 0.0, "width": 0.0, "height": 0.0}

        # Calculate average dimensions from all bounding boxes
        lengths = []
        widths = []
        heights = []

        for bbox in bboxes:
            if "bbox_3d" in bbox:
                # Assume bbox_3d contains min and max coordinates
                if (
                    isinstance(bbox["bbox_3d"], dict)
                    and "min" in bbox["bbox_3d"]
                    and "max" in bbox["bbox_3d"]
                ):
                    min_coords = np.array(bbox["bbox_3d"]["min"])
                    max_coords = np.array(bbox["bbox_3d"]["max"])

                    lengths.append(max_coords[0] - min_coords[0])
                    widths.append(max_coords[1] - min_coords[1])
                    heights.append(max_coords[2] - min_coords[2])

        return {
            "length": np.mean(lengths) if lengths else 0.0,
            "width": np.mean(widths) if widths else 0.0,
            "height": np.mean(heights) if heights else 0.0,
        }

    # =========================================================================
    # For Phase 0: New Helper Functions (Recursive Intent)
    # =========================================================================

    @staticmethod
    def estimate_volume_heuristic(
        depth_map: np.ndarray,
        mask: np.ndarray,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> float:
        """
        Estimate volume from depth map using point cloud back-projection.
        Uses camera intrinsics for 3D projection (Vectorized).

        Args:
            depth_map: 2D array of depth values (in meters).
            mask: Binary mask indicating object region.
            fx, fy: Camera focal lengths.
            cx, cy: Camera principal point (default: image center).

        Returns:
            Estimated volume in cubic meters.
        """
        if depth_map.shape != mask.shape:
            raise ValueError("Depth map and mask must have the same shape")

        h, w = depth_map.shape
        if cx is None:
            cx = w / 2
        if cy is None:
            cy = h / 2

        # Apply mask
        masked_depth = depth_map * mask

        # Vectorized Grid Generation
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)

        # Vectorized Back-projection
        # Valid points mask (z > 0 and in object mask)
        valid_mask = masked_depth > 0

        if not np.any(valid_mask):
            return 0.0

        z = masked_depth[valid_mask]
        x = (uu[valid_mask] - cx) * z / fx
        y = (vv[valid_mask] - cy) * z / fy

        # Stack into (N, 3) array
        valid_points = np.stack((x, y, z), axis=1)

        # Downsample if too many points (for ConvexHull performance)
        if len(valid_points) > 10000:
            # Simple random sampling for speed vs VoxelGrid
            indices = np.random.choice(len(valid_points), 5000, replace=False)
            valid_points = valid_points[indices]

        if len(valid_points) < 4:
            return 0.0

        return MathUtils.calculate_convex_hull_volume(valid_points.tolist())

    @staticmethod
    def calculate_bbox_overlap(
        bbox1: Dict[str, float], bbox2: Dict[str, float]
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bbox with keys 'x', 'y', 'w', 'h'.
            bbox2: Second bbox with keys 'x', 'y', 'w', 'h'.

        Returns:
            IoU score (0-1).
        """
        x1 = max(bbox1["x"], bbox2["x"])
        y1 = max(bbox1["y"], bbox2["y"])
        x2 = min(bbox1["x"] + bbox1["w"], bbox2["x"] + bbox2["w"])
        y2 = min(bbox1["y"] + bbox1["h"], bbox2["y"] + bbox2["h"])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = bbox1["w"] * bbox1["h"]
        area2 = bbox2["w"] * bbox2["h"]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def downsample_point_cloud(
        points: np.ndarray, voxel_size: float = 0.01
    ) -> np.ndarray:
        """
        Downsample point cloud using voxel grid filtering for performance.
        Inspired by CountNet3D heuristics.

        Args:
            points: Nx3 array of 3D points.
            voxel_size: Size of each voxel for downsampling.

        Returns:
            Downsampled point cloud.
        """
        if len(points) == 0:
            return points

        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # Use unique voxels to downsample
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

        return points[unique_indices]

    @staticmethod
    def validate_physical_bounds(
        estimated_count: int,
        bounding_box_volume: float,
        unit_volume: float,
        packing_factor: float = 0.64,  # Random packing ~64%
    ) -> Tuple[bool, str]:
        """
        Sanity check: Ensure estimated count is physically plausible.
        Inspired by CountNet3D physical constraints.

        Args:
            estimated_count: Estimated number of objects.
            bounding_box_volume: Volume of the containing box.
            unit_volume: Average volume of a single object.
            packing_factor: Maximum packing density (default: 0.64 for random).

        Returns:
            Tuple of (is_valid, reason).
        """
        if unit_volume <= 0:
            return False, "Unit volume must be positive"

        max_possible_count = int((bounding_box_volume * packing_factor) / unit_volume)

        if estimated_count > max_possible_count:
            return (
                False,
                f"Count {estimated_count} exceeds physical limit {max_possible_count}",
            )

        return True, "Physical bounds satisfied"

    @staticmethod
    def lattice_counting(total_volume: float, unit_volume: float) -> Tuple[int, int]:
        """
        Count objects in a neat stack using lattice division.

        Args:
            total_volume: Total volume of the stack.
            unit_volume: Volume of a single unit.

        Returns:
            Tuple of (min_count, max_count) range.
        """
        if unit_volume <= 0:
            return (0, 0)

        exact_count = total_volume / unit_volume
        min_count = int(np.floor(exact_count * 0.9))  # 10% tolerance
        max_count = int(np.ceil(exact_count * 1.1))

        return (max(0, min_count), max_count)

    @staticmethod
    def find_depth_peaks(
        depth_map: np.ndarray,
        mask: np.ndarray,
        min_distance: int = 20,
        threshold_rel: float = 0.1,
    ) -> List[Tuple[int, int]]:
        """
        Find local maxima in depth map for PointBeam prompting.
        Used to generate SAM2 point prompts for stacked objects.

        Args:
            depth_map: 2D array of depth values.
            mask: Binary mask indicating region of interest.
            min_distance: Minimum distance between peaks.
            threshold_rel: Relative threshold for peak detection.

        Returns:
            List of (y, x) coordinates of depth peaks.
        """
        from scipy.ndimage import maximum_filter

        masked_depth = depth_map * mask

        # Find local maxima using maximum filter
        local_max = maximum_filter(masked_depth, size=min_distance)
        peaks = (masked_depth == local_max) & (masked_depth > 0)

        # Apply threshold
        threshold = np.max(masked_depth) * threshold_rel
        peaks = peaks & (masked_depth > threshold)

        # Get coordinates
        peak_coords = np.argwhere(peaks)

        return [tuple(coord) for coord in peak_coords]
