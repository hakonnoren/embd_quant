"""
Random orthogonal rotation for binary quantization.

Random rotation helps make coordinates more uniformly distributed,
which significantly improves quantization quality for structured data.
"""

import numpy as np
from typing import Optional


class RandomOrthogonalRotation:
    """
    Random orthogonal rotation using QR decomposition.

    Generates a uniformly random orthogonal matrix via Stewart's method.
    O(D^3) to generate, O(D^2) per rotation.
    """

    def __init__(self, d: int, seed: Optional[int] = None):
        """
        Initialize random orthogonal rotation.

        Args:
            d: Dimension of vectors
            seed: Random seed for reproducibility
        """
        self.d = d
        self.seed = seed
        self.P = self._generate_orthogonal_matrix()

    def _generate_orthogonal_matrix(self) -> np.ndarray:
        """Generate uniformly random orthogonal matrix."""
        rng = np.random.default_rng(self.seed)

        # Random Gaussian matrix
        G = rng.standard_normal((self.d, self.d))

        # QR decomposition
        Q, R = np.linalg.qr(G)

        # Adjust signs for uniform distribution (Mezzadri's algorithm)
        d = np.sign(np.diag(R))
        d[d == 0] = 1
        Q = Q * d

        return Q.astype(np.float32)

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply rotation: y = Px.

        Args:
            vectors: Input vectors, shape (n, d) or (d,)

        Returns:
            Rotated vectors
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        single = vectors.ndim == 1

        if single:
            vectors = vectors.reshape(1, -1)

        rotated = vectors @ self.P.T

        return rotated.flatten() if single else rotated

    def inverse_rotate(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply inverse rotation: x = P^T y.

        Args:
            vectors: Rotated vectors

        Returns:
            Original vectors
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        single = vectors.ndim == 1

        if single:
            vectors = vectors.reshape(1, -1)

        original = vectors @ self.P

        return original.flatten() if single else original


class FastHadamardRotation:
    """
    Fast pseudo-random rotation using Walsh-Hadamard transform.

    R = H * D where:
    - H is the normalized Hadamard matrix (orthonormal)
    - D is a diagonal matrix of random ±1 signs

    O(D log D) per rotation instead of O(D^2).
    Dimension must be a power of 2.
    """

    def __init__(self, d: int, n_rounds: int = 3, seed: Optional[int] = None):
        """
        Initialize fast Hadamard rotation.

        Args:
            d: Dimension (must be power of 2)
            n_rounds: Number of HD rounds (more = better mixing)
            seed: Random seed
        """
        if d & (d - 1) != 0:
            raise ValueError(f"Dimension must be power of 2, got {d}")

        self.d = d
        self.n_rounds = n_rounds
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Precompute random signs for each round (shared across all vectors)
        self.signs = [
            rng.choice([-1.0, 1.0], size=d).astype(np.float32)
            for _ in range(n_rounds)
        ]

    @staticmethod
    def _fwht_batch(X: np.ndarray) -> np.ndarray:
        """
        Fast Walsh-Hadamard transform for batch of vectors.

        In-place transform, normalized to preserve norms.
        """
        X = np.array(X, dtype=np.float32, copy=True)
        n, d = X.shape
        h = 1

        while h < d:
            X = X.reshape(n, -1, 2 * h)
            a = X[:, :, :h].copy()
            b = X[:, :, h:2*h]
            X[:, :, :h] = a + b
            X[:, :, h:2*h] = a - b
            X = X.reshape(n, d)
            h *= 2

        X *= np.float32(1.0 / np.sqrt(d))
        return X

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply fast random rotation.

        Args:
            vectors: Input vectors, shape (n, d) or (d,)

        Returns:
            Rotated vectors
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        single = vectors.ndim == 1

        if single:
            vectors = vectors.reshape(1, -1)

        X = vectors.copy()
        for r in range(self.n_rounds):
            X = X * self.signs[r][None, :]
            X = self._fwht_batch(X)

        return X.flatten() if single else X

    def inverse_rotate(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply inverse rotation.

        For H*D, inverse is D^{-1} * H^{-1} = D * H (since D and H are self-inverse).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        single = vectors.ndim == 1

        if single:
            vectors = vectors.reshape(1, -1)

        X = vectors.copy()
        for r in range(self.n_rounds - 1, -1, -1):
            X = self._fwht_batch(X)
            X = X * self.signs[r][None, :]

        return X.flatten() if single else X


def create_rotation(
    d: int,
    method: str = "hadamard",
    seed: Optional[int] = None,
    **kwargs,
):
    """
    Factory function to create a rotation object.

    Args:
        d: Dimension
        method: "qr" for exact random orthogonal, "hadamard" for fast
        seed: Random seed
        **kwargs: Additional arguments (e.g., n_rounds for hadamard)

    Returns:
        Rotation object with rotate() and inverse_rotate() methods
    """
    if method == "qr":
        return RandomOrthogonalRotation(d, seed=seed)
    elif method == "hadamard":
        n_rounds = kwargs.get("n_rounds", 3)
        return FastHadamardRotation(d, n_rounds=n_rounds, seed=seed)
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown method: {method}")
