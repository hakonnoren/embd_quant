"""Quantization utilities for embeddings."""
import numpy as np
from typing import Optional
from sentence_transformers.quantization import quantize_embeddings


class QuantizationHandler:
    """Handles various embedding quantization schemes."""

    def __init__(self):
        self.int8_ranges: Optional[np.ndarray] = None

    def calibrate_int8(self, calibration_embeddings: np.ndarray) -> None:
        """Compute min/max ranges for int8 quantization from calibration data."""
        min_vals = calibration_embeddings.min(axis=0)
        max_vals = calibration_embeddings.max(axis=0)
        self.int8_ranges = np.vstack([min_vals, max_vals])  # Shape: (2, dim)

    def quantize_to_int8(
        self,
        embeddings: np.ndarray,
        calibration_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Quantize embeddings to int8 using sentence-transformers.

        Args:
            embeddings: Float32 embeddings to quantize
            calibration_embeddings: Optional calibration data for computing ranges

        Returns:
            Int8 quantized embeddings
        """
        if calibration_embeddings is not None:
            return quantize_embeddings(
                embeddings, precision="int8", calibration_embeddings=calibration_embeddings
            )
        elif self.int8_ranges is not None:
            return quantize_embeddings(
                embeddings, precision="int8", ranges=self.int8_ranges
            )
        else:
            # Fall back to using embeddings themselves for calibration
            return quantize_embeddings(embeddings, precision="int8")

    def compute_binary_thresholds(self, calibration_embeddings: np.ndarray) -> np.ndarray:
        """Compute per-dimension median thresholds from calibration data.

        Returns:
            Threshold vector of shape (dim,) — the per-dimension median.
        """
        self.binary_thresholds = np.median(calibration_embeddings, axis=0)
        return self.binary_thresholds

    def center_for_binary(self, embeddings: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Subtract per-dimension thresholds so that binarization at 0 is equivalent to binarization at median."""
        return embeddings - thresholds

    # --- Quaternary (2-bit, quartile-based) ---

    def calibrate_quaternary(self, corpus_embeddings: np.ndarray) -> dict:
        """Compute quartile boundaries and per-bucket centroids.

        Returns dict with 'boundaries' (3, d) and 'centroids' (4, d).
        """
        boundaries = np.percentile(corpus_embeddings, [25, 50, 75], axis=0).astype(np.float32)  # (3, d)

        # Assign codes to compute centroids
        codes = np.zeros_like(corpus_embeddings, dtype=np.uint8)
        codes[corpus_embeddings >= boundaries[0]] = 1
        codes[corpus_embeddings >= boundaries[1]] = 2
        codes[corpus_embeddings >= boundaries[2]] = 3

        dim = corpus_embeddings.shape[1]
        centroids = np.zeros((4, dim), dtype=np.float32)
        for c in range(4):
            mask = (codes == c)
            for j in range(dim):
                col_mask = mask[:, j]
                if col_mask.any():
                    centroids[c, j] = corpus_embeddings[col_mask, j].mean()

        self.quaternary_boundaries = boundaries
        self.quaternary_centroids = centroids
        return {"boundaries": boundaries, "centroids": centroids}

    def quantize_to_quaternary(self, embeddings: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
        """Quantize to 2-bit codes using quartile boundaries.

        Returns (n, d) uint8 array with values in {0, 1, 2, 3}.
        """
        codes = np.zeros(embeddings.shape, dtype=np.uint8)
        codes[embeddings >= boundaries[0]] = 1
        codes[embeddings >= boundaries[1]] = 2
        codes[embeddings >= boundaries[2]] = 3
        return codes

    def reconstruct_quaternary(self, codes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Reconstruct float vectors from quaternary codes using centroids."""
        dim = centroids.shape[1]
        return centroids[codes, np.arange(dim)]

    # --- Lloyd-Max Gaussian (2-bit, MSE-optimal for Gaussian) ---

    # Fixed Gaussian-optimal constants
    LLOYD_BOUNDS = np.array([-0.9816, 0.0, 0.9816])
    LLOYD_LEVELS = np.array([-1.5104, -0.4528, 0.4528, 1.5104], dtype=np.float32)

    def calibrate_lloyd_max(self, corpus_embeddings: np.ndarray) -> dict:
        """Compute per-dimension median and std for Lloyd-Max quantization.

        Returns dict with 'medians' (d,) and 'stds' (d,).
        """
        medians = np.median(corpus_embeddings, axis=0).astype(np.float32)
        stds = np.std(corpus_embeddings, axis=0).astype(np.float32)
        stds = np.clip(stds, 1e-10, None)
        self.lloyd_medians = medians
        self.lloyd_stds = stds
        return {"medians": medians, "stds": stds}

    def quantize_to_lloyd_max(self, embeddings: np.ndarray, medians: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Quantize to 2-bit Lloyd-Max codes.

        Returns (n, d) uint8 array with values in {0, 1, 2, 3}.
        """
        z = (embeddings - medians) / stds
        codes = np.digitize(z, self.LLOYD_BOUNDS).astype(np.uint8)
        return codes

    def reconstruct_lloyd_max(self, codes: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Reconstruct in standardized space — caller must handle scoring trick."""
        return self.LLOYD_LEVELS[codes]

    def quantize_to_binary(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantize embeddings to binary (1-bit) representation.

        Uses ubinary format for packed uint8 representation.
        Original dim=D becomes D/8 bytes.

        Returns:
            Binary embeddings packed as uint8 array
        """
        return quantize_embeddings(embeddings, precision="ubinary")

    def truncate_matryoshka(
        self, embeddings: np.ndarray, target_dim: int
    ) -> np.ndarray:
        """
        Truncate embeddings to target dimension (Matryoshka).

        Simply takes the first target_dim dimensions.
        Re-normalizes after truncation for cosine similarity.

        Args:
            embeddings: Original embeddings
            target_dim: Target dimension to truncate to

        Returns:
            Truncated and re-normalized embeddings
        """
        truncated = embeddings[:, :target_dim]
        # Re-normalize
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return truncated / norms

    @staticmethod
    def compute_memory_bytes(n_vectors: int, dim: int, precision: str) -> int:
        """
        Calculate memory usage in bytes.

        Args:
            n_vectors: Number of vectors
            dim: Dimension of vectors (before packing for binary)
            precision: One of 'float32', 'int8', 'binary'

        Returns:
            Memory usage in bytes
        """
        if precision == "float32":
            return n_vectors * dim * 4
        elif precision == "int8":
            return n_vectors * dim * 1
        elif precision in ["binary", "binary_rescore", "binary_median"]:
            # Packed binary: dim/8 bytes per vector
            return n_vectors * (dim // 8)
        elif precision in ["quaternary", "lloyd_max"]:
            # 2-bit: dim/4 bytes per vector (4 codes per byte)
            return n_vectors * max(1, dim // 4)
        else:
            raise ValueError(f"Unknown precision: {precision}")
