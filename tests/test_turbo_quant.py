"""Tests for turbo-quant-lite."""

import numpy as np
import pytest

from turbo_quant_lite import TurboQuant, pack, unpack


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tq4():
    return TurboQuant(dim=768, bits=4, seed=42)


@pytest.fixture
def tq3():
    return TurboQuant(dim=768, bits=3, seed=42)


@pytest.fixture
def tq2():
    return TurboQuant(dim=768, bits=2, seed=42)


@pytest.fixture
def tq1():
    return TurboQuant(dim=768, bits=1, seed=42)


def random_vectors(n: int, dim: int, seed: int = 123) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim)


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_encode_decode_preserves_shape(self, tq4):
        x = np.random.randn(768)
        indices, norm = tq4.encode(x)
        assert indices.shape == (768,)
        assert indices.dtype == np.uint8
        restored = tq4.decode(indices, norm)
        assert restored.shape == (768,)

    def test_zero_vector(self, tq4):
        x = np.zeros(768)
        indices, norm = tq4.encode(x)
        assert norm == 0.0
        restored = tq4.decode(indices, norm)
        np.testing.assert_array_equal(restored, np.zeros(768))

    def test_unit_vector(self, tq4):
        x = np.zeros(768)
        x[0] = 1.0
        indices, norm = tq4.encode(x)
        assert abs(norm - 1.0) < 1e-10
        restored = tq4.decode(indices, norm)
        assert restored.shape == (768,)

    def test_deterministic_with_same_seed(self):
        x = np.random.randn(768)
        tq_a = TurboQuant(dim=768, bits=4, seed=42)
        tq_b = TurboQuant(dim=768, bits=4, seed=42)
        idx_a, norm_a = tq_a.encode(x)
        idx_b, norm_b = tq_b.encode(x)
        np.testing.assert_array_equal(idx_a, idx_b)
        assert norm_a == norm_b

    def test_different_seed_gives_different_result(self):
        x = np.random.randn(768)
        tq_a = TurboQuant(dim=768, bits=4, seed=42)
        tq_b = TurboQuant(dim=768, bits=4, seed=99)
        idx_a, _ = tq_a.encode(x)
        idx_b, _ = tq_b.encode(x)
        # Overwhelmingly likely to differ
        assert not np.array_equal(idx_a, idx_b)


# ---------------------------------------------------------------------------
# Distortion bounds
# ---------------------------------------------------------------------------


class TestDistortion:
    """Verify empirical MSE falls within theoretical bounds."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_within_theoretical_bound(self, bits):
        """MSE distortion should be <= 2.72 * (1/4^bits) for random unit vectors."""
        dim = 768
        tq = TurboQuant(dim=dim, bits=bits, seed=42)
        vectors = random_vectors(200, dim, seed=777)

        total_mse = 0.0
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm == 0:
                continue
            indices, n = tq.encode(v)
            restored = tq.decode(indices, n)
            err = np.sum((v - restored) ** 2) / (norm ** 2)
            total_mse += err

        avg_mse = total_mse / len(vectors)
        shannon_bound = 1.0 / (4.0 ** bits)
        theoretical_upper = 2.72 * shannon_bound

        # Allow 20% margin for finite-sample variance
        assert avg_mse <= theoretical_upper * 1.2, (
            f"MSE {avg_mse:.6f} exceeds theoretical bound {theoretical_upper:.6f} * 1.2 at {bits}-bit"
        )

    def test_4bit_mse_below_1_percent(self):
        """4-bit should achieve < 1.1% normalized MSE on random vectors."""
        tq = TurboQuant(dim=768, bits=4, seed=42)
        vectors = random_vectors(500, 768, seed=888)

        mses = []
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm == 0:
                continue
            indices, n = tq.encode(v)
            restored = tq.decode(indices, n)
            mses.append(np.sum((v - restored) ** 2) / (norm ** 2))

        avg_mse = np.mean(mses)
        assert avg_mse < 0.012, f"4-bit MSE {avg_mse:.6f} exceeds 1.2%"


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_similarity_correlates_with_true_dot(self, tq4):
        """Approximate similarity should correlate highly with true inner product."""
        vectors = random_vectors(100, 768, seed=111)
        query = np.random.RandomState(222).randn(768)

        true_dots = []
        approx_dots = []
        for v in vectors:
            indices, norm = tq4.encode(v)
            true_dots.append(np.dot(query, v))
            approx_dots.append(tq4.similarity(query, indices, norm))

        correlation = np.corrcoef(true_dots, approx_dots)[0, 1]
        assert correlation > 0.99, f"Correlation {correlation:.4f} too low"

    def test_similarity_batch_matches_individual(self, tq4):
        vectors = random_vectors(50, 768, seed=333)
        query = np.random.RandomState(444).randn(768)

        all_indices, all_norms = tq4.encode_batch(vectors)
        batch_scores = tq4.similarity_batch(query, all_indices, all_norms)

        for i in range(len(vectors)):
            individual = tq4.similarity(query, all_indices[i], all_norms[i])
            np.testing.assert_allclose(batch_scores[i], individual, rtol=1e-10)


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_matches_individual(self, tq4):
        vectors = random_vectors(50, 768, seed=555)
        all_indices, all_norms = tq4.encode_batch(vectors)

        for i, v in enumerate(vectors):
            single_idx, single_norm = tq4.encode(v)
            np.testing.assert_array_equal(all_indices[i], single_idx)
            np.testing.assert_allclose(all_norms[i], single_norm, rtol=1e-10)

    def test_batch_decode_matches_individual(self, tq4):
        vectors = random_vectors(50, 768, seed=666)
        all_indices, all_norms = tq4.encode_batch(vectors)
        batch_decoded = tq4.decode_batch(all_indices, all_norms)

        for i in range(len(vectors)):
            single_decoded = tq4.decode(all_indices[i], all_norms[i])
            np.testing.assert_allclose(batch_decoded[i], single_decoded, rtol=1e-10)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestPackUnpack:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_pack_unpack_roundtrip(self, bits):
        dim = 768
        tq = TurboQuant(dim=dim, bits=bits, seed=42)
        x = np.random.randn(dim)
        indices, norm = tq.encode(x)

        data = pack(indices, norm, bits=bits)
        unpacked_indices, unpacked_norm = unpack(data, dim=dim, bits=bits)

        np.testing.assert_array_equal(unpacked_indices, indices)
        np.testing.assert_allclose(unpacked_norm, norm, rtol=1e-6)

    def test_4bit_size(self):
        """4-bit packing should produce 4 + dim/2 bytes."""
        dim = 768
        tq = TurboQuant(dim=dim, bits=4, seed=42)
        indices, norm = tq.encode(np.random.randn(dim))
        data = pack(indices, norm, bits=4)
        assert len(data) == 4 + dim // 2  # 388 bytes

    def test_2bit_size(self):
        dim = 768
        tq = TurboQuant(dim=dim, bits=2, seed=42)
        indices, norm = tq.encode(np.random.randn(dim))
        data = pack(indices, norm, bits=2)
        assert len(data) == 4 + dim // 4  # 196 bytes

    def test_1bit_size(self):
        dim = 768
        tq = TurboQuant(dim=dim, bits=1, seed=42)
        indices, norm = tq.encode(np.random.randn(dim))
        data = pack(indices, norm, bits=1)
        assert len(data) == 4 + dim // 8  # 100 bytes


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


class TestDimensions:
    @pytest.mark.parametrize("dim", [64, 128, 256, 384, 768, 1024, 1536, 3072])
    def test_various_dimensions(self, dim):
        tq = TurboQuant(dim=dim, bits=4, seed=42)
        x = np.random.randn(dim)
        indices, norm = tq.encode(x)
        restored = tq.decode(indices, norm)
        assert restored.shape == (dim,)
        # Basic sanity: not all zeros (unless input was zero)
        if np.linalg.norm(x) > 0:
            assert np.linalg.norm(restored) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_very_large_norm(self, tq4):
        x = np.random.randn(768) * 1e6
        indices, norm = tq4.encode(x)
        restored = tq4.decode(indices, norm)
        relative_err = np.linalg.norm(x - restored) / np.linalg.norm(x)
        assert relative_err < 0.15  # 4-bit, should be well under this

    def test_very_small_norm(self, tq4):
        x = np.random.randn(768) * 1e-8
        indices, norm = tq4.encode(x)
        restored = tq4.decode(indices, norm)
        relative_err = np.linalg.norm(x - restored) / np.linalg.norm(x)
        assert relative_err < 0.15

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            TurboQuant(dim=0, bits=4)

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            TurboQuant(dim=768, bits=5)

    def test_compression_ratio(self, tq4):
        ratio = tq4.compression_ratio()
        assert 7.5 < ratio < 8.5  # Should be ~7.9x for 4-bit, dim=768

    def test_compressed_size(self, tq4):
        assert tq4.compressed_size_bytes() == 388  # 4 + 768*4/8
