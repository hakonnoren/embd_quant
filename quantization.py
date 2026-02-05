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
        elif precision in ["binary", "binary_rescore"]:
            # Packed binary: dim/8 bytes per vector
            return n_vectors * (dim // 8)
        else:
            raise ValueError(f"Unknown precision: {precision}")
