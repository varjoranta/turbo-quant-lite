"""TurboQuant vector quantization — numpy only, no PyTorch.

Implements the core algorithm from:
  Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization
  with Near-optimal Distortion Rate." 2025. arXiv:2504.19874

Codebook values from Lloyd-Max optimal solution for the Beta distribution
on the unit hypersphere at d=512, verified against the Rust reference
implementation (github.com/aisar-labs/turboquant-rs).
"""

from __future__ import annotations

import math
import struct

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Precomputed codebooks (Lloyd-Max optimal for Beta distribution at d=512)
# Centroids are in [-1, 1] for unit-sphere coordinates.
# ---------------------------------------------------------------------------

_C1 = 0.03516  # E[|X|] for Beta(d=512)

_CODEBOOKS: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}


def _build_symmetric(positives: list[float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build symmetric centroids and midpoint boundaries from positive half."""
    centroids = np.array([-c for c in reversed(positives)] + positives)
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries


def _get_codebook(bits: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if bits in _CODEBOOKS:
        return _CODEBOOKS[bits]

    if bits == 1:
        centroids = np.array([-_C1, _C1])
        boundaries = np.array([0.0])
    elif bits == 2:
        centroids, boundaries = _build_symmetric([0.01999, 0.06672])
    elif bits == 3:
        centroids, boundaries = _build_symmetric([0.01082, 0.03338, 0.05934, 0.09501])
    elif bits == 4:
        centroids, boundaries = _build_symmetric(
            [0.005668, 0.01713, 0.02900, 0.04161, 0.05547, 0.07143, 0.09136, 0.12066]
        )
    else:
        raise ValueError(f"Supported bit widths: 1-4, got {bits}")

    _CODEBOOKS[bits] = (centroids, boundaries)
    return centroids, boundaries


# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------


def _random_orthogonal(dim: int, seed: int) -> NDArray[np.float64]:
    """Uniformly random orthogonal matrix via QR with Haar measure correction."""
    rng = np.random.RandomState(seed)
    a = rng.randn(dim, dim)
    q, r = np.linalg.qr(a)
    # Sign correction for uniform Haar measure
    q *= np.sign(np.diag(r))
    return q


# ---------------------------------------------------------------------------
# TurboQuant
# ---------------------------------------------------------------------------


class TurboQuant:
    """Data-oblivious vector quantizer.

    Args:
        dim: Vector dimensionality (must match all vectors encoded/decoded).
        bits: Quantization bit width (1-4). Default 4.
        seed: Random seed for rotation matrix. Must be the same for encode and decode.
    """

    def __init__(self, dim: int, bits: int = 4, seed: int = 42) -> None:
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.rotation = _random_orthogonal(dim, seed)
        self.centroids, self.boundaries = _get_codebook(bits)

    def encode(self, x: NDArray[np.floating]) -> tuple[NDArray[np.uint8], float]:
        """Quantize a single vector.

        Returns (indices, norm) where indices is uint8 array of codebook indices
        and norm is the original L2 norm.
        """
        x = np.asarray(x, dtype=np.float64)
        norm = float(np.linalg.norm(x))
        if norm == 0.0:
            return np.zeros(self.dim, dtype=np.uint8), 0.0
        y = self.rotation @ (x / norm)
        indices = np.searchsorted(self.boundaries, y).astype(np.uint8)
        return indices, norm

    def decode(self, indices: NDArray[np.uint8], norm: float) -> NDArray[np.float64]:
        """Reconstruct a vector from quantized representation."""
        if norm == 0.0:
            return np.zeros(self.dim, dtype=np.float64)
        y_hat = self.centroids[indices]
        return norm * (self.rotation.T @ y_hat)

    def encode_batch(
        self, xs: NDArray[np.floating]
    ) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
        """Quantize a batch of vectors. xs shape: (n, dim).

        Returns (all_indices, all_norms) with shapes (n, dim) and (n,).
        """
        xs = np.asarray(xs, dtype=np.float64)
        norms = np.linalg.norm(xs, axis=1)
        safe_norms = np.where(norms > 0, norms, 1.0)
        normalized = xs / safe_norms[:, np.newaxis]
        rotated = (self.rotation @ normalized.T).T  # (n, dim)
        all_indices = np.searchsorted(self.boundaries, rotated).astype(np.uint8)
        return all_indices, norms

    def decode_batch(
        self, all_indices: NDArray[np.uint8], all_norms: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Reconstruct a batch of vectors."""
        y_hat = self.centroids[all_indices]  # (n, dim)
        reconstructed = (self.rotation.T @ y_hat.T).T  # (n, dim)
        return all_norms[:, np.newaxis] * reconstructed

    def similarity(self, query: NDArray[np.floating], indices: NDArray[np.uint8], norm: float) -> float:
        """Approximate inner product between a full-precision query and a quantized vector.

        Faster than decode + dot because it skips the inverse rotation
        (orthogonal rotation preserves inner products).
        """
        query = np.asarray(query, dtype=np.float64)
        q_rot = self.rotation @ query
        y_hat = self.centroids[indices]
        return float(norm * np.dot(q_rot, y_hat))

    def similarity_batch(
        self,
        query: NDArray[np.floating],
        all_indices: NDArray[np.uint8],
        all_norms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Approximate inner products between a query and a batch of quantized vectors."""
        query = np.asarray(query, dtype=np.float64)
        q_rot = self.rotation @ query  # (dim,)
        y_hat = self.centroids[all_indices]  # (n, dim)
        return all_norms * (y_hat @ q_rot)  # (n,)

    def compressed_size_bytes(self) -> int:
        """Size in bytes of one compressed vector (packed indices + norm)."""
        return 4 + math.ceil(self.dim * self.bits / 8)

    def compression_ratio(self) -> float:
        """Compression ratio vs float32."""
        return (self.dim * 4) / self.compressed_size_bytes()


# ---------------------------------------------------------------------------
# Binary serialization
# ---------------------------------------------------------------------------


def pack(indices: NDArray[np.uint8], norm: float, bits: int) -> bytes:
    """Pack quantized vector into compact bytes.

    Format: [4 bytes float32 norm][bit-packed indices]
    """
    header = struct.pack("<f", norm)
    if bits == 4:
        # Pack two 4-bit indices per byte
        paired = indices.reshape(-1, 2) if len(indices) % 2 == 0 else np.append(indices, [0]).reshape(-1, 2)
        packed = (paired[:, 0] | (paired[:, 1] << 4)).astype(np.uint8)
        return header + packed.tobytes()
    if bits == 2:
        # Pack four 2-bit indices per byte
        padded = indices
        remainder = len(indices) % 4
        if remainder:
            padded = np.append(indices, np.zeros(4 - remainder, dtype=np.uint8))
        grouped = padded.reshape(-1, 4)
        packed = (grouped[:, 0] | (grouped[:, 1] << 2) | (grouped[:, 2] << 4) | (grouped[:, 3] << 6)).astype(
            np.uint8
        )
        return header + packed.tobytes()
    if bits == 1:
        # Pack eight 1-bit indices per byte
        padded = indices
        remainder = len(indices) % 8
        if remainder:
            padded = np.append(indices, np.zeros(8 - remainder, dtype=np.uint8))
        packed = np.packbits(padded, bitorder="little")
        return header + packed.tobytes()
    if bits == 3:
        # Pack indices as uint8 (no bit-packing for 3-bit; wastes ~60% but keeps it simple)
        return header + indices.tobytes()
    raise ValueError(f"Unsupported bits: {bits}")


def unpack(data: bytes, dim: int, bits: int) -> tuple[NDArray[np.uint8], float]:
    """Unpack bytes back to (indices, norm)."""
    (norm,) = struct.unpack("<f", data[:4])
    payload = data[4:]
    if bits == 4:
        raw = np.frombuffer(payload, dtype=np.uint8)
        low = raw & 0x0F
        high = (raw >> 4) & 0x0F
        indices = np.empty(len(raw) * 2, dtype=np.uint8)
        indices[0::2] = low
        indices[1::2] = high
        indices = indices[:dim]
    elif bits == 2:
        raw = np.frombuffer(payload, dtype=np.uint8)
        b0 = raw & 0x03
        b1 = (raw >> 2) & 0x03
        b2 = (raw >> 4) & 0x03
        b3 = (raw >> 6) & 0x03
        indices = np.empty(len(raw) * 4, dtype=np.uint8)
        indices[0::4] = b0
        indices[1::4] = b1
        indices[2::4] = b2
        indices[3::4] = b3
        indices = indices[:dim]
    elif bits == 1:
        raw = np.frombuffer(payload, dtype=np.uint8)
        indices = np.unpackbits(raw, bitorder="little")[:dim]
    elif bits == 3:
        indices = np.frombuffer(payload, dtype=np.uint8)[:dim]
    else:
        raise ValueError(f"Unsupported bits: {bits}")
    return indices, norm
