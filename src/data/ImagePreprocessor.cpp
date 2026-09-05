#include "data/ImagePreprocessor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace sir {

// ---------------------------------------------------------------------------
// Internal helper: perform the full spec §5 pipeline on one decoded Mat.
//
// Pipeline:
//   1. Validate (non-empty, non-zero dims, acceptable channels)
//   2. Normalise to 3-channel BGR (grayscale -> replicate; BGRA -> drop alpha)
//   3. Resize shortest edge to 224 with INTER_CUBIC (bicubic), preserving
//      aspect ratio.  Matches open_clip Resize(size=224, bicubic).
//   4. Center-crop to exactly 224x224.
//      Matches open_clip CenterCrop(224, 224).
//   5. BGR -> RGB reorder
//   6. Convert to CV_32F and scale to [0, 1]  (divide by 255)
//   7. Per-channel CLIP/ImageNet normalisation:
//        pixel_norm = (pixel - mean) / std
//      using RGB-order constants:
//        mean = (0.48145466, 0.45782750, 0.40821073)
//        std  = (0.26862954, 0.26130258, 0.27577711)
//
// Returns false and populates outReason on any failure.
// ---------------------------------------------------------------------------

static bool applyPipeline(cv::Mat&           mat,
                           std::string&        outReason)
{
    // --- Validation --------------------------------------------------------
    if (mat.empty()) {
        outReason = "decoded Mat is empty";
        return false;
    }
    if (mat.rows == 0 || mat.cols == 0) {
        outReason = "decoded Mat has zero dimension";
        return false;
    }
    const int ch = mat.channels();
    if (ch != 1 && ch != 3 && ch != 4) {
        outReason = "unsupported channel count: " + std::to_string(ch);
        return false;
    }

    // --- Normalise channel count to 3-channel BGR --------------------------
    if (ch == 1) {
        // Grayscale -> BGR (replicate single channel)
        cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
    } else if (ch == 4) {
        // BGRA -> BGR (drop alpha)
        cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
    }
    // ch == 3: already BGR, nothing to do

    // --- Step 1: resize shortest edge to 224 (bicubic, aspect-ratio preserved)
    // Mirrors: torchvision Resize(size=224, interpolation=BICUBIC)
    // When size is a scalar, torchvision resizes the shorter side to that value
    // while scaling the longer side proportionally.
    {
        const int H = mat.rows;
        const int W = mat.cols;
        if (H != ImagePreprocessor::kTargetHeight ||
            W != ImagePreprocessor::kTargetWidth)
        {
            int newH, newW;
            if (H <= W) {
                // Height is the shorter (or equal) edge.
                newH = ImagePreprocessor::kTargetHeight;
                newW = static_cast<int>(
                    std::round(static_cast<double>(W) *
                               ImagePreprocessor::kTargetHeight / H));
            } else {
                // Width is the shorter edge.
                newW = ImagePreprocessor::kTargetWidth;
                newH = static_cast<int>(
                    std::round(static_cast<double>(H) *
                               ImagePreprocessor::kTargetWidth / W));
            }
            cv::resize(mat, mat, cv::Size(newW, newH), 0, 0, cv::INTER_CUBIC);
        }
    }

    // --- Step 2: center-crop to exactly 224x224 ----------------------------
    // Mirrors: torchvision CenterCrop(224, 224)
    {
        const int H  = mat.rows;
        const int W  = mat.cols;
        const int y0 = (H - ImagePreprocessor::kTargetHeight) / 2;
        const int x0 = (W - ImagePreprocessor::kTargetWidth)  / 2;
        // Use .clone() so the cropped Mat owns its memory (not a subview).
        mat = mat(cv::Rect(x0, y0,
                           ImagePreprocessor::kTargetWidth,
                           ImagePreprocessor::kTargetHeight)).clone();
    }

    // --- BGR -> RGB --------------------------------------------------------
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    // --- float32, scale to [0,1] -------------------------------------------
    mat.convertTo(mat, CV_32F, 1.0 / 255.0);

    // --- Per-channel CLIP/ImageNet normalisation ----------------------------
    // mat is now CV_32FC3 in RGB order.
    // Split, normalise each channel, then merge.
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);

    // channels[0]=R, channels[1]=G, channels[2]=B  (after BGR->RGB above)
    channels[0] = (channels[0] - ImagePreprocessor::kMeanR)
                  / ImagePreprocessor::kStdR;
    channels[1] = (channels[1] - ImagePreprocessor::kMeanG)
                  / ImagePreprocessor::kStdG;
    channels[2] = (channels[2] - ImagePreprocessor::kMeanB)
                  / ImagePreprocessor::kStdB;

    cv::merge(channels, mat);

    return true;
}

// ---------------------------------------------------------------------------
// ImagePreprocessor::preprocessOne
// ---------------------------------------------------------------------------

bool ImagePreprocessor::preprocessOne(const std::string& filePath,
                                       int                internalId,
                                       PreprocessedImage&  outImage,
                                       std::string&        outReason) const
{
    outImage = PreprocessedImage{};   // reset to default state
    outReason.clear();

    // Full pixel decode via OpenCV (CLIP uses colour images; request BGR).
    cv::Mat raw = cv::imread(filePath, cv::IMREAD_UNCHANGED);

    if (raw.empty()) {
        outReason = "cv::imread returned empty Mat for: " + filePath;
        return false;
    }

    // Run the preprocessing pipeline on the decoded Mat.
    if (!applyPipeline(raw, outReason)) {
        return false;
    }

    outImage.internalId = internalId;
    outImage.mat        = std::move(raw);
    return true;
}

// ---------------------------------------------------------------------------
// ImagePreprocessor::processRecords
// ---------------------------------------------------------------------------

std::vector<PreprocessedImage>
ImagePreprocessor::processRecords(const std::vector<ImageRecord>& records,
                                   PreprocessingReport&            outReport,
                                   bool                            verbose,
                                   const std::string&              manifestPath) const
{
    outReport = PreprocessingReport{};
    std::vector<PreprocessedImage> results;
    results.reserve(records.size());

    // Optional manifest output file.
    std::ofstream manifestFile;
    if (!manifestPath.empty()) {
        manifestFile.open(manifestPath);
        if (!manifestFile.is_open()) {
            throw std::runtime_error("Cannot open manifest path for writing: " + manifestPath);
        }
        // Write a simple header
        manifestFile << "imageId\tfilename\n";
    }

    for (const ImageRecord& rec : records) {
        ++outReport.totalAttempted;

        PreprocessedImage img;
        std::string reason;

        if (preprocessOne(rec.filePath, rec.internalId, img, reason)) {
            ++outReport.successCount;
            results.push_back(std::move(img));

            // Write to manifest if enabled. We extract the filename to keep it portable.
            if (manifestFile.is_open()) {
                std::filesystem::path p(rec.filePath);
                manifestFile << rec.imageId << '\t' << p.filename().string() << '\n';
            }
        } else {
            // Determine failure category from the reason string / Mat state.
            // applyPipeline sets the reason; imread failure is caught in
            // preprocessOne before applyPipeline is called.
            if (reason.find("cv::imread") != std::string::npos) {
                ++outReport.failedLoad;
            } else if (reason.find("zero dimension") != std::string::npos) {
                ++outReport.failedZeroDimension;
            } else if (reason.find("unsupported channel") != std::string::npos) {
                ++outReport.failedBadChannels;
            } else {
                // Catch-all: treat as load failure so the count is never lost.
                ++outReport.failedLoad;
            }

            if (verbose) {
                std::cerr << "[ImagePreprocessor] FAIL (internalId="
                          << rec.internalId << " path=" << rec.filePath
                          << "): " << reason << '\n';
            }
        }
    }

    if (manifestFile.is_open()) {
        manifestFile.close();
    }

    return results;
}

}  // namespace sir
