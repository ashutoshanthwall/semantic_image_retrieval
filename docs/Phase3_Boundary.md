# Phase 3: Image Preprocessing

## Phase 3 Responsibilities
- **Dataset image validation**: Ensures the image file is readable, not corrupted, and possesses valid dimensions (>0).
- **Corrupted/unreadable image rejection**: Identifies files that OpenCV cannot fully decode and omits them from the downstream manifest.
- **Supported-channel handling**: standardizes grayscale (1-channel) and RGBA (4-channel) into 3-channel BGR.
- **Preprocessing/validation behavior**: performs a shortest-edge bicubic resize and center-crop to exactly 224x224, then computes normalized RGB values using CLIP ImageNet constants to verify computational integrity.
- **Persistent valid-image handoff**: Generates a deterministic manifest (`data/processed/valid_ids.txt`) containing the valid `imageId` and `filename` mapping for all images that pass this pipeline, while omitting any failed images.

## Phase Boundary: Phase 3 vs Phase 4

- **Phase 3**: C++ dataset validation → persistent valid-image manifest (`data/processed/valid_ids.txt`).
  - C++ runs full pixel decoding, ensures data is safe, and records IDs that are guaranteed to open correctly.
- **Phase 4**: Python/OpenCLIP reads ONLY validated images (from the Phase 3 manifest) → native OpenCLIP preprocessing (PyTorch `transforms`) → embedding generation.
  - C++ does *not* export preprocessed pixel tensors to Python. Python simply reads the original files identified by the manifest and performs inference natively.
