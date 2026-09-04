# AGENTS.md

## Purpose

This repository is a C++ + DSA + AI/ML + Computer Vision +
Information Retrieval project.

`PROJECT_SPEC.txt` is the authoritative architectural specification.

Read it before making architectural changes.

## Operating Rules

1. Do not redesign the architecture unless explicitly instructed.
2. Do not implement the entire application in one pass.
3. Implement only the currently requested phase.
4. Inspect the repository and existing code before editing.
5. Keep each implementation step buildable and testable.
6. Do not silently modify unrelated components.
7. After implementation, build and run the relevant tests.
8. Report what changed, what was tested, and any remaining issues.
9. If the specification genuinely leaves a major architectural
   decision undefined, identify the ambiguity before making a
   consequential choice.
10. Never invent benchmark results, dataset statistics, performance
    measurements, or test results.

## Source of Truth

Use this precedence:

1. Explicit user instruction in the current task
2. `PROJECT_SPEC.txt`
3. Existing repository design and tests
4. Conservative implementation judgment

Do not replace project requirements with a generic "best practice"
architecture without explicit justification.

## Responsibility Boundary

### Python / ML side

Python may handle:
- dataset preparation
- dataset utilities
- CLIP inference
- image embedding generation
- preprocessing required for CLIP
- generation of persistent embedding files

### C++ side

C++ owns:
- embedding storage
- vector mathematics
- cosine similarity
- Top-K retrieval
- exact brute-force retrieval
- HNSW implementation
- metadata lookup
- persistence
- search orchestration
- benchmarking
- core DSA infrastructure

## Critical Project Constraints

### HNSW

HNSW is a core DSA component.

Do not replace the custom HNSW implementation with a vector database
or ANN service.

Do not use FAISS, Milvus, Qdrant, Pinecone, Chroma, Elasticsearch,
or another external vector-search engine as a substitute for the
project's C++ retrieval implementation.

Libraries may be used for supporting functionality where appropriate,
but the core retrieval algorithms must remain our implementation.

### Embeddings

Image embeddings are generated offline and persisted.

Do not regenerate all image embeddings every time the search
application starts.

Do not hard-code a CLIP embedding dimension throughout the codebase.
The dimension is model-dependent and must be treated as configuration.

Validate:
- dimension
- finite numeric values
- non-empty vectors
- valid IDs

### Dataset

`data/raw/` contains the original dataset.

Never modify raw Flickr30k files in place.

Derived information belongs in:
- `data/processed/`
- `data/embeddings/`

Report invalid files rather than silently ignoring them.

### Retrieval

Brute force is the exact reference implementation.

HNSW is approximate.

HNSW quality must be evaluated against brute-force results using
metrics such as Recall@K.

Use the same query embeddings when comparing retrieval algorithms.

### Scope

Do not add future upgrades to the core implementation unless
explicitly requested.

Future upgrades include:
- query expansion
- YOLO
- hybrid scoring
- reranking
- image-to-image search
- OCR
- relevance feedback

## Development Workflow

For each requested phase:

### Before coding

1. Read `PROJECT_SPEC.txt`.
2. Inspect the current repository.
3. Inspect relevant headers/source/tests.
4. Identify interfaces and dependencies.
5. State a short implementation plan.

### During coding

- Make small, coherent changes.
- Preserve existing interfaces unless the requested phase requires
  a change.
- Prefer clear ownership and avoid unnecessary copies.
- Use RAII and appropriate smart pointers.
- Keep model-dependent values configurable.
- Avoid premature optimization.
- Do not add unrelated dependencies.

### After coding

1. Configure/build the affected target.
2. Run relevant unit tests.
3. Run integration tests when applicable.
4. Fix compilation and obvious test failures.
5. Check for accidental unrelated changes.
6. Report:
   - files changed
   - implementation completed
   - build status
   - test status
   - known limitations

Never claim a test passed unless it was actually run.

## HNSW Development Rule

Do not begin by constructing HNSW over the full Flickr30k dataset.

First create a small deterministic synthetic test set.

Verify:
- insertion
- graph connectivity
- search
- Top-K behavior
- comparison with brute force
- edge cases

Only then use the full dataset.

## Benchmarking Rules

Never fabricate benchmark values.

When benchmarking:
- use identical query embeddings for both methods
- distinguish embedding-generation time from retrieval latency when
  possible
- record actual measured values
- calculate Recall@5 and Recall@10 from actual result sets
- report actual latency, speedup, memory, and build time
- label illustrative examples as illustrative

## Dependency Rule

Do not install or silently introduce system/Python dependencies
unless the user explicitly asks the agent to manage dependencies.

The user is intentionally managing the development environment and
dependencies manually.

The agent's primary role is implementation, testing, debugging,
and integration.

## Git Rule

Do not create commits, push to remotes, rewrite history, or change
remote configuration unless explicitly instructed.

Do not commit:
- raw Flickr30k images
- generated embeddings
- model weights
- virtual environments
- build artifacts
- secrets
- API keys

## Communication Rule

Be precise.

If something is unknown:
- say it is unknown
- inspect the repository or documentation if available
- do not hallucinate

If a requested implementation conflicts with `PROJECT_SPEC.txt`,
flag the conflict before making the change.

## Required Implementation Order

Follow the project's dependency order:

1. Dataset ingestion
2. Metadata model
3. Image preprocessing
4. CLIP embedding generation
5. Embedding persistence
6. VectorMath
7. Cosine similarity
8. Top-K
9. BruteForceIndex
10. RetrievalIndex abstraction
11. HNSWNode
12. HNSW graph structure
13. HNSW insertion
14. HNSW search
15. HNSW correctness tests
16. Real Flickr30k HNSW index
17. SearchEngine
18. Metadata integration
19. Persistence
20. BenchmarkEngine
21. Exact-vs-HNSW evaluation
22. CLI/basic interface

Do not skip ahead simply because a later component appears easy.

## Completion Standard

The core project is complete only when:

Flickr30k
→ CLIP image embeddings
→ persistent embeddings
→ C++ embedding store
→ exact brute-force retrieval
→ C++ HNSW retrieval
→ natural-language query
→ Top-K results
→ metadata lookup
→ actual image paths/results
→ benchmark
→ Recall + latency + speedup + memory

Everything beyond this is an optional upgrade.
