#pragma once

#include "data/ImageRecord.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace sir {

// ============================================================================
// PreprocessedImage
//
// Holds the result of preprocessing one image into a CLIP-compatible
// in-memory tensor.
//
// Layout:
//   mat  : cv::Mat of type CV_32FC3, size 224x224.
//          Pixel values are normalised per channel:
//            (value/255 - mean) / stddev
//          using CLIP's ImageNet normalisation constants:
//            mean = {0.48145466, 0.4578275,  0.40821073}  (RGB order)
//            std  = {0.26862954, 0.26130258, 0.27577711}  (RGB order)
//          Channel order is RGB (OpenCV loads as BGR; this class converts).
//
//   internalId : copied from the source ImageRecord so the caller can
//                correlate the tensor with its metadata record.
//
// OpenCV does NOT perform semantic understanding; it provides the
// pixel-level operations (decode, resize, colour-convert, normalise)
// that produce CLIP-compatible input.  CLIP inference remains in
// Python (Step 4) and is not part of this component.
// ============================================================================
struct PreprocessedImage {
    int      internalId{-1};
    cv::Mat  mat;            ///< CV_32FC3, 224x224, RGB, CLIP-normalised
};

// ============================================================================
// PreprocessingReport
//
// Exact counts from a processRecords() call.  All values are measured;
// none are estimated or hard-coded.
// ============================================================================
struct PreprocessingReport {
    int totalAttempted{0};       ///< records submitted for preprocessing
    int successCount{0};         ///< images that produced a valid PreprocessedImage
    int failedLoad{0};           ///< cv::imread returned an empty Mat
    int failedZeroDimension{0};  ///< decoded Mat had rows==0 or cols==0
    int failedBadChannels{0};    ///< decoded Mat had channels not in {1, 3, 4}
};

// ============================================================================
// ImagePreprocessor
//
// Implements the image-preprocessing pipeline specified in section 5
// of PROJECT_SPEC.txt:
//
//   Image file
//       |  cv::imread (full pixel decode)
//       v
//   Loaded Mat
//       |  Validate (non-empty, non-zero dims, acceptable channels)
//       v
//   Validated Mat
//       |  Resize/crop to 224x224 (CLIP canonical input size)
//       v
//   Resized Mat
//       |  BGR->RGB colour conversion
//       |  Convert to float32, scale to [0,1]
//       |  Per-channel normalise (CLIP ImageNet constants)
//       v
//   PreprocessedImage (CV_32FC3, 224x224, RGB-normalised)
//
// Responsibilities (per spec section 5):
//   - image loading    ("OpenCV is responsible for: image loading")
//   - image validation ("image validation")
//   - image manipulation and preprocessing ("image manipulation, preprocessing")
//
// Not responsible for:
//   - CLIP inference
//   - embedding generation or persistence
//   - VectorMath / similarity
//   - EmbeddingStore
//
// Invalid images are NEVER silently treated as valid.  Each failure
// category is counted in PreprocessingReport and, when verbose=true,
// logged to std::cerr.
//
// The internalId / filePath relationship from DatasetManager is
// preserved: processRecords() does NOT reorder or compact the input
// vector.  Invalid images are excluded from the output with their
// counts tracked in PreprocessingReport.
// ============================================================================
class ImagePreprocessor {
public:
    // CLIP canonical input dimensions.
    static constexpr int kTargetWidth  = 224;
    static constexpr int kTargetHeight = 224;

    // CLIP ImageNet normalisation constants (RGB channel order).
    static constexpr float kMeanR = 0.48145466f;
    static constexpr float kMeanG = 0.45782750f;
    static constexpr float kMeanB = 0.40821073f;
    static constexpr float kStdR  = 0.26862954f;
    static constexpr float kStdG  = 0.26130258f;
    static constexpr float kStdB  = 0.27577711f;

    ImagePreprocessor()  = default;
    ~ImagePreprocessor() = default;

    // Non-copyable, movable.
    ImagePreprocessor(const ImagePreprocessor&)            = delete;
    ImagePreprocessor& operator=(const ImagePreprocessor&) = delete;
    ImagePreprocessor(ImagePreprocessor&&)                 = default;
    ImagePreprocessor& operator=(ImagePreprocessor&&)      = default;

    // -----------------------------------------------------------------------
    // preprocessOne
    //
    // Preprocess a single image identified by its file path.
    //
    // @param filePath     Absolute or project-relative path to the image.
    // @param internalId   The internalId to stamp into the result.
    // @param outImage     Populated on success.
    // @param outReason    Human-readable failure reason (empty on success).
    // @return true on success, false on any validation or decode failure.
    //
    // On failure: outImage is left in a default-constructed state;
    //             outReason describes the specific problem.
    // -----------------------------------------------------------------------
    bool preprocessOne(const std::string& filePath,
                       int                internalId,
                       PreprocessedImage&  outImage,
                       std::string&        outReason) const;

    // -----------------------------------------------------------------------
    // processRecords
    //
    // Preprocess every ImageRecord in `records`.
    //
    // For each record: calls preprocessOne(record.filePath, record.internalId).
    //   - Success  -> appended to the returned vector; successCount incremented.
    //                 If manifestPath is provided, the record's imageId and
    //                 filename are written to the manifest.
    //   - Failure  -> NOT appended; appropriate failure counter incremented;
    //                 if verbose, the failure is logged to std::cerr.
    //
    // The order of successful results in the output vector matches the order
    // of their source records in `records`.  internalIds are never changed.
    //
    // @param records      Records from DatasetManager (width/height may be 0).
    // @param outReport    Filled with exact per-category counts.
    // @param verbose      Print per-image failures to std::cerr.
    // @param manifestPath Optional path to write a persistent manifest of valid IDs.
    // @return             PreprocessedImages for every successfully processed record.
    // @throws std::runtime_error if manifestPath is non-empty but cannot be written.
    // -----------------------------------------------------------------------
    std::vector<PreprocessedImage> processRecords(
        const std::vector<ImageRecord>& records,
        PreprocessingReport&            outReport,
        bool                            verbose = false,
        const std::string&              manifestPath = "") const;
};

}  // namespace sir
