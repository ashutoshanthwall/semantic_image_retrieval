# C++-Based Semantic Image Retrieval System

A C++/DSA-oriented semantic image retrieval system using Flickr30k,
CLIP embeddings, exact brute-force retrieval, and a custom HNSW
approximate nearest-neighbor index.

## Project question

Can graph-based approximate nearest-neighbor retrieval using HNSW
reduce semantic image-search latency while maintaining high
retrieval recall compared with exact brute-force search?

## Architecture

Flickr30k
→ preprocessing
→ CLIP image embeddings
→ persistent embedding store
→ Brute Force / HNSW
→ natural-language query
→ Top-K image IDs
→ metadata lookup
→ results

## Documentation

- `PROJECT_SPEC.txt` — authoritative project architecture
- `AGENTS.md` — coding-agent rules

## Status

Development environment and implementation are being built
incrementally. No benchmark result in this repository should be
treated as valid until measured by the project benchmark system.
