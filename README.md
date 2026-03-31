# turbo-quant-lite

Numpy-only vector quantization based on Google's TurboQuant algorithm. Compresses float32 vectors to 1-4 bit indices with near-optimal quality. No PyTorch, no CUDA, no model dependencies.

```python
from turbo_quant_lite import TurboQuant

tq = TurboQuant(dim=768, bits=4)

indices, norm = tq.encode(embedding)   # 3072 bytes → 388 bytes
restored = tq.decode(indices, norm)    # < 1.1% MSE distortion
```

## Why this exists

There are two existing TurboQuant implementations on PyPI:

- **[turboquant](https://pypi.org/project/turboquant/)** — PyTorch-based, focused on LLM KV cache compression. Full HuggingFace integration and GPU support. Requires PyTorch (~2 GB install).
- **[turboquant-vectors](https://pypi.org/project/turboquant-vectors/)** — Numpy-only, focused on batch vector compression and embedding privacy. Includes a `PrivateEncoder` for protecting embeddings against inversion attacks. Designed for the "compress a collection, save to disk, search the collection" workflow.

This package fills a different niche: **per-vector compression for database storage.** It's for applications that store embeddings row-by-row in PostgreSQL, SQLite, or Redis and need to compress each vector into a compact binary blob (388 bytes at 4-bit, dim=768) that can be stored in a `bytea` column or cache key.

Key differences from turboquant-vectors:
- **Per-vector binary serialization** — `pack()` / `unpack()` produce compact bytes for database row storage. No file I/O required.
- **Zero per-vector overhead** — the quantizer is shared (initialized once), compressed data is just indices + norm. No 2MB wrapper per vector.
- **Direct single-vector similarity** — `similarity(query, indices, norm)` works on raw indices without wrapping in a collection object.

Same algorithm, same quality, designed for the database storage use case.

For **embedding privacy** (protecting against inversion attacks like Vec2Text), see [turboquant-vectors PrivateEncoder](https://pypi.org/project/turboquant-vectors/). You can apply a secret rotation before compression — the two compose naturally as separate layers.

## Install

```bash
pip install turbo-quant-lite
```

Or just copy `turbo_quant_lite/core.py` into your project. It's one file.

## What is TurboQuant?

TurboQuant is a data-oblivious vector quantization algorithm from Google Research. It compresses vectors without needing training data or calibration — it works instantly on any vector from any source.

The key insight: randomly rotate a vector and each coordinate becomes approximately Gaussian with known variance. Since the distribution is known in advance, you can precompute the optimal quantization grid. This turns a hard problem (data-dependent codebook learning) into a table lookup.

**Results:**
- 4-bit: 8x compression, < 1.1% MSE distortion
- 3-bit: ~10x compression, < 4.3% MSE distortion
- 2-bit: 16x compression, < 17% MSE distortion

Quality is within 2.72x of the information-theoretic optimum (Shannon lower bound) at every bit width. This bound is provable and data-independent — it holds for any vector, not just your benchmark.

**Paper:** Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* 2025. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

**Reference implementations:**
- [turboquant](https://github.com/back2matching/turboquant) — PyTorch, KV cache focus, GPU support
- [turboquant-rs](https://github.com/aisar-labs/turboquant-rs) — Rust, research/verification focus

## Usage

### Basic encode/decode

```python
import numpy as np
from turbo_quant_lite import TurboQuant

tq = TurboQuant(dim=768, bits=4, seed=42)

# Any float array — from OpenAI, Nebius, Cohere, local model, etc.
embedding = np.random.randn(768).astype(np.float32)

# Compress
indices, norm = tq.encode(embedding)
# indices: uint8 array (768 values, each 0-15 for 4-bit)
# norm: float (the vector's L2 norm)

# Decompress
restored = tq.decode(indices, norm)

# Quality check
mse = np.mean((embedding - restored) ** 2) / np.mean(embedding ** 2)
# mse < 0.011 for 4-bit (guaranteed by theory)
```

### Batch operations

```python
embeddings = np.random.randn(1000, 768)

all_indices, all_norms = tq.encode_batch(embeddings)
restored_batch = tq.decode_batch(all_indices, all_norms)
```

### Approximate similarity (fast, no full decompression)

```python
query = np.random.randn(768)
score = tq.similarity(query, indices, norm)
# Equivalent to cosine similarity but skips the inverse rotation
```

### Binary serialization for storage

```python
from turbo_quant_lite import pack, unpack

# Pack to bytes (388 bytes for 4-bit, dim=768)
data = pack(indices, norm, bits=4)

# Unpack from bytes
indices, norm = unpack(data, dim=768, bits=4)
```

### Use with any embedding provider

```python
# OpenAI
response = openai.embeddings.create(model="text-embedding-3-small", input="hello")
embedding = np.array(response.data[0].embedding)
compressed = tq.encode(embedding)

# Nebius
embedding = await nebius_embedder.embed_text("hello")
compressed = tq.encode(np.array(embedding))

# Sentence Transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("hello")
compressed = tq.encode(embedding)
```

## When to use this

**Good fit:**
- Storing embeddings in a database (8x size reduction)
- Caching embeddings in Redis/Valkey (8x memory reduction)
- Shipping embeddings over the network (8x bandwidth reduction)
- Local vector search where you control the storage format
- Cold storage / backups of embedding collections
- Edge devices with limited memory

**Not a good fit:**
- pgvector search (pgvector needs float32/halfvec, no native 4-bit support yet)
- LLM KV cache compression (use [turboquant](https://pypi.org/project/turboquant/) with PyTorch)
- Sub-millisecond latency requirements at dim > 2048 (the rotation matmul becomes the bottleneck)

## Performance

On a typical CPU (M-series Mac, modern x86), dim=768:

| Operation | Time | Notes |
|-----------|------|-------|
| `encode` (single) | ~1.5ms | Dominated by rotation matmul |
| `decode` (single) | ~1.5ms | Same matmul |
| `encode_batch(1000)` | ~400ms | Amortized 0.4ms/vector |
| `similarity` | ~0.3ms | Skips inverse rotation |
| `pack` | ~0.1ms | Bit packing |

## Storage sizes (dim=768)

| Format | Bytes per vector | Compression |
|--------|-----------------|-------------|
| float32 | 3,072 | 1x |
| float16 | 1,536 | 2x |
| 4-bit TurboQuant | 388 | **7.9x** |
| 3-bit TurboQuant | 292 | **10.5x** |
| 2-bit TurboQuant | 196 | **15.7x** |

## Important: seed must match

The rotation matrix is generated from the seed. Encoding with `seed=42` and decoding with `seed=43` produces garbage. Use the same seed everywhere, or serialize the TurboQuant instance.

## License

MIT
