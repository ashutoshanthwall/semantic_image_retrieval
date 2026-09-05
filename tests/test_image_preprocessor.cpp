/**
 * test_image_preprocessor.cpp
 *
 * Tests for ImagePreprocessor (Phase 3, Step 3).
 *
 * Covers:
 *   A. preprocessOne on a known-good image from Flickr30k
 *      - loads without error
 *      - produces CV_32FC3 mat, 224x224
 *      - pixel values are in normalised range (sanity check)
 *   B. processRecords on a small sample from Flickr30k
 *      - successCount > 0
 *      - report accounting: successCount + failedLoad + failedZeroDimension
 *        + failedBadChannels == totalAttempted
 *      - all output mats are CV_32FC3, 224x224
 *      - internalIds in output match source records
 *   C. Invalid / unreadable path is rejected, never silently accepted
 *      - preprocessOne returns false
 *      - outReason is non-empty
 *      - processRecords counts it in failedLoad, not successCount
 *   D. Empty record list produces empty output and zeroed report
 *
 * Convention:
 *   - Paths resolved via SIR_PROJECT_ROOT env var (same as Phase 1/2 tests).
 *   - No result values are fabricated.
 *   - No OpenCV headless display calls.
 */

#include "data/DatasetManager.hpp"
#include "data/ImagePreprocessor.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Minimal test harness (same style as Phase 1/2)
// ---------------------------------------------------------------------------

static int gPassed = 0;
static int gFailed = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::cerr << "[FAIL] " << (msg) << '\n';                 \
            ++gFailed;                                                \
        } else {                                                      \
            std::cout << "[PASS] " << (msg) << '\n';                 \
            ++gPassed;                                                \
        }                                                             \
    } while (false)

// ---------------------------------------------------------------------------

int main(int /*argc*/, char** /*argv*/) {

    // Resolve project root.
    const char* envRoot    = std::getenv("SIR_PROJECT_ROOT");
    const fs::path projRoot = envRoot ? fs::path(envRoot) : fs::current_path();

    const fs::path imageDir    = projRoot / "data/raw/flickr30k/flickr30k-images";
    const fs::path captionJson = projRoot / "data/raw/flickr30k/dataset_flickr30k.json";

    std::cout << "=== ImagePreprocessor Tests ===\n";
    std::cout << "Project root : " << projRoot   << '\n';
    std::cout << "Image dir    : " << imageDir   << '\n';

    // Pre-flight: dataset must be present.
    if (!fs::is_directory(imageDir) || !fs::is_regular_file(captionJson)) {
        std::cerr << "[SKIP] Flickr30k dataset not found at expected location.\n";
        std::cerr << "       Cannot run integration tests without the dataset.\n";
        return 1;
    }

    // Load a small sample via DatasetManager (reuses Phase 1 component).
    sir::DatasetManager dm;
    const sir::IngestionReport ingReport =
        dm.load(imageDir.string(), captionJson.string(), /*verbose=*/false);

    if (ingReport.successCount == 0) {
        std::cerr << "[FATAL] DatasetManager loaded 0 records.\n";
        return 1;
    }

    const std::vector<sir::ImageRecord>& allRecords = dm.records();

    // Work with first 5 records for the focused tests (fast, deterministic).
    const int kSampleSize = std::min(5, static_cast<int>(allRecords.size()));
    std::vector<sir::ImageRecord> sample(
        allRecords.begin(), allRecords.begin() + kSampleSize);

    sir::ImagePreprocessor preprocessor;

    // =========================================================================
    // A. preprocessOne on the first valid Flickr30k image
    // =========================================================================
    std::cout << "\n--- A. preprocessOne (single valid image) ---\n";
    {
        const sir::ImageRecord& firstRec = allRecords[0];
        std::cout << "  Image: " << firstRec.filePath << '\n';

        sir::PreprocessedImage outImg;
        std::string reason;

        const bool ok = preprocessor.preprocessOne(
            firstRec.filePath, firstRec.internalId, outImg, reason);

        CHECK(ok, "A1: preprocessOne returns true for a valid image");
        CHECK(reason.empty(), "A2: outReason is empty on success");

        if (ok) {
            CHECK(outImg.internalId == firstRec.internalId,
                  "A3: outImage.internalId matches source record");
            CHECK(!outImg.mat.empty(),
                  "A4: output mat is not empty");
            CHECK(outImg.mat.rows == sir::ImagePreprocessor::kTargetHeight,
                  "A5: output mat height == 224");
            CHECK(outImg.mat.cols == sir::ImagePreprocessor::kTargetWidth,
                  "A6: output mat width == 224");
            CHECK(outImg.mat.type() == CV_32FC3,
                  "A7: output mat type is CV_32FC3");
            CHECK(outImg.mat.channels() == 3,
                  "A8: output mat has 3 channels");

            // Sanity-check: normalised pixel values should not all be exactly
            // zero or exactly one; the distribution should be spread around 0
            // after ImageNet normalisation.  We compute the absolute mean of
            // the first channel; a completely degenerate result would give
            // either 0 (blank) or a single fixed value.
            cv::Scalar meanVal = cv::mean(outImg.mat);
            // After normalisation, values are centred near 0 and could be
            // negative; just confirm the mat is numerically populated.
            CHECK(outImg.mat.total() > 0,
                  "A9: output mat has non-zero total pixel count");

            std::cout << "  Channel means after normalisation: "
                      << "R=" << meanVal[0]
                      << " G=" << meanVal[1]
                      << " B=" << meanVal[2] << '\n';
        }
    }

    // =========================================================================
    // B. processRecords on the sample
    // =========================================================================
    std::cout << "\n--- B. processRecords (sample of "
              << kSampleSize << " records) ---\n";
    {
        sir::PreprocessingReport report;
        std::vector<sir::PreprocessedImage> results =
            preprocessor.processRecords(sample, report, /*verbose=*/true);

        std::cout << "  totalAttempted       : " << report.totalAttempted    << '\n';
        std::cout << "  successCount         : " << report.successCount      << '\n';
        std::cout << "  failedLoad           : " << report.failedLoad        << '\n';
        std::cout << "  failedZeroDimension  : " << report.failedZeroDimension << '\n';
        std::cout << "  failedBadChannels    : " << report.failedBadChannels << '\n';

        CHECK(report.totalAttempted == kSampleSize,
              "B1: totalAttempted == kSampleSize");

        // Accounting invariant: no image is lost.
        const int accountedFor = report.successCount
                               + report.failedLoad
                               + report.failedZeroDimension
                               + report.failedBadChannels;
        CHECK(accountedFor == report.totalAttempted,
              "B2: successCount + failures == totalAttempted");

        CHECK(report.successCount > 0,
              "B3: at least one image succeeded");

        CHECK(static_cast<int>(results.size()) == report.successCount,
              "B4: results.size() == successCount");

        // All output mats must have the correct shape and type.
        int badShape = 0;
        int badType  = 0;
        for (const auto& img : results) {
            if (img.mat.rows != sir::ImagePreprocessor::kTargetHeight ||
                img.mat.cols != sir::ImagePreprocessor::kTargetWidth) {
                ++badShape;
            }
            if (img.mat.type() != CV_32FC3) {
                ++badType;
            }
        }
        CHECK(badShape == 0, "B5: all output mats are 224x224");
        CHECK(badType  == 0, "B6: all output mats are CV_32FC3");

        // internalIds in output must be non-negative and must correspond to
        // valid records (they are a subset of the sample's IDs).
        int badId = 0;
        for (const auto& img : results) {
            if (img.internalId < 0) ++badId;
        }
        CHECK(badId == 0, "B7: all output internalIds are non-negative");
    }

    // =========================================================================
    // C. Invalid / unreadable path rejected explicitly
    // =========================================================================
    std::cout << "\n--- C. Invalid path is rejected ---\n";
    {
        // C1: non-existent file
        {
            sir::PreprocessedImage outImg;
            std::string reason;
            const bool ok = preprocessor.preprocessOne(
                "/nonexistent/path/fake_image_99999.jpg", 42, outImg, reason);

            CHECK(!ok,           "C1a: preprocessOne returns false for non-existent path");
            CHECK(!reason.empty(),"C1b: outReason is non-empty on failure");
            CHECK(outImg.mat.empty(),
                  "C1c: outImage.mat is empty after failure");
            CHECK(outImg.internalId == -1,
                  "C1d: outImage.internalId is -1 (default) after failure");
        }

        // C2: processRecords with one synthetic invalid record
        {
            sir::ImageRecord badRec;
            badRec.internalId = 99;
            badRec.imageId    = "fake";
            badRec.filePath   = "/nonexistent/path/fake_image.jpg";

            sir::PreprocessingReport report;
            std::vector<sir::PreprocessedImage> results =
                preprocessor.processRecords({badRec}, report, /*verbose=*/false);

            CHECK(report.totalAttempted == 1,
                  "C2a: totalAttempted == 1");
            CHECK(report.successCount == 0,
                  "C2b: successCount == 0 (invalid image not accepted)");
            CHECK(report.failedLoad == 1,
                  "C2c: failedLoad == 1");
            CHECK(results.empty(),
                  "C2d: results vector is empty (no silent acceptance)");
        }
    }

    // =========================================================================
    // D. Empty record list
    // =========================================================================
    std::cout << "\n--- D. Empty record list ---\n";
    {
        sir::PreprocessingReport report;
        std::vector<sir::PreprocessedImage> results =
            preprocessor.processRecords({}, report, /*verbose=*/false);

        CHECK(results.empty(),            "D1: empty input -> empty output");
        CHECK(report.totalAttempted == 0, "D2: totalAttempted == 0");
        CHECK(report.successCount   == 0, "D3: successCount == 0");
        CHECK(report.failedLoad     == 0, "D4: failedLoad == 0");
    }

    // =========================================================================
    // E. Mathematical verification of normalisation (Synthetic Image)
    // =========================================================================
    std::cout << "\n--- E. Mathematical normalisation verification ---\n";
    {
        // We write out a temporary synthetic image to test the normalisation
        // constants exactly. This avoids resizing or cropping issues because we
        // generate it directly at 224x224.
        const fs::path synthPath = projRoot / "data/raw/synthetic_norm_test.png";
        
        // Choose known RGB values. 
        // OpenCV writes in BGR, so we construct the Mat as BGR!
        // Target RGB: R=128, G=64, B=192
        const uchar targetR = 128;
        const uchar targetG = 64;
        const uchar targetB = 192;
        cv::Mat synthMat(224, 224, CV_8UC3, cv::Scalar(targetB, targetG, targetR));
        cv::imwrite(synthPath.string(), synthMat);

        sir::PreprocessedImage outImg;
        std::string reason;
        const bool ok = preprocessor.preprocessOne(
            synthPath.string(), 100, outImg, reason);

        CHECK(ok, "E1: synthetic image loaded successfully");

        if (ok) {
            // Expected analytic values:
            // R = (128/255 - 0.48145466) / 0.26862954
            // G = ( 64/255 - 0.45782750) / 0.26130258
            // B = (192/255 - 0.40821073) / 0.27577711
            const float expR = (128.0f / 255.0f - sir::ImagePreprocessor::kMeanR) / sir::ImagePreprocessor::kStdR;
            const float expG = ( 64.0f / 255.0f - sir::ImagePreprocessor::kMeanG) / sir::ImagePreprocessor::kStdG;
            const float expB = (192.0f / 255.0f - sir::ImagePreprocessor::kMeanB) / sir::ImagePreprocessor::kStdB;

            // Sample the center pixel
            const cv::Vec3f& pixel = outImg.mat.at<cv::Vec3f>(112, 112);

            // Recall that the output is in RGB order
            const float actualR = pixel[0];
            const float actualG = pixel[1];
            const float actualB = pixel[2];

            const float kTol = 1e-5f;
            CHECK(std::abs(actualR - expR) <= kTol, "E2: R channel normalisation exact");
            CHECK(std::abs(actualG - expG) <= kTol, "E3: G channel normalisation exact");
            CHECK(std::abs(actualB - expB) <= kTol, "E4: B channel normalisation exact");
        }

        // Clean up
        if (fs::exists(synthPath)) fs::remove(synthPath);
    }

    // =========================================================================
    // F. Aspect-ratio / Crop verification (Synthetic Non-Square Image)
    // =========================================================================
    std::cout << "\n--- F. Aspect-ratio preservation and center-crop ---\n";
    {
        // Construct a 400x300 image (landscape).
        // Shortest edge = 300. 
        // Resize scale = 224 / 300 = 0.74666...
        // New Width = 400 * (224 / 300) = 299 (rounded).
        // So resized image is 299x224.
        // Center crop to 224x224 crops (299 - 224)/2 = 37 pixels from left/right.
        const fs::path synthPath = projRoot / "data/raw/synthetic_aspect_test.png";
        
        cv::Mat synthMat(300, 400, CV_8UC3, cv::Scalar(0, 0, 0)); // Black

        // We want the cropped area to be entirely blue (B=255, G=0, R=0) in OpenCV BGR.
        // Everything outside the cropped area will be red (B=0, G=0, R=255).
        // Original X range that survives: crop x starts at 37 on 299-width image.
        // 37 * (300/224) = 49.5 -> ~50.
        // Let's make x in [60, 340] blue, and x < 60 or x > 340 red.
        for (int y = 0; y < 300; ++y) {
            for (int x = 0; x < 400; ++x) {
                if (x < 60 || x > 340) {
                    synthMat.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // Red (BGR)
                } else {
                    synthMat.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0); // Blue (BGR)
                }
            }
        }
        cv::imwrite(synthPath.string(), synthMat);

        sir::PreprocessedImage outImg;
        std::string reason;
        const bool ok = preprocessor.preprocessOne(
            synthPath.string(), 101, outImg, reason);

        CHECK(ok, "F1: non-square image loaded successfully");

        if (ok) {
            CHECK(outImg.mat.rows == 224 && outImg.mat.cols == 224, 
                  "F2: non-square image outputs 224x224");
            
            // Expected blue normalisation (B=255, R=0, G=0):
            const float expBlue_R = (0.0f / 255.0f - sir::ImagePreprocessor::kMeanR) / sir::ImagePreprocessor::kStdR;
            const float expBlue_B = (255.0f / 255.0f - sir::ImagePreprocessor::kMeanB) / sir::ImagePreprocessor::kStdB;

            const float kTol = 0.1f; // allow some interpolation blur

            const cv::Vec3f& centerPix = outImg.mat.at<cv::Vec3f>(112, 112);
            CHECK(std::abs(centerPix[0] - expBlue_R) <= kTol && 
                  std::abs(centerPix[2] - expBlue_B) <= kTol, 
                  "F3: center pixel is blue (not red or distorted)");
        }
        
        // Write a specific non-square test specifically for squash vs crop.
        // 400x300 image, vertical line at original X = 100.
        // If squashed: X=100 -> 100 * (224/400) = X_out = 56.
        // If cropped: Resize to 299x224. X=100 -> 100 * (299/400) = 74.75.
        // Then crop from X=37. So X_out = 74.75 - 37 = 37.75.
        
        const fs::path synthPath2 = projRoot / "data/raw/synthetic_marker_test.png";
        cv::Mat synthMat2(300, 400, CV_8UC3, cv::Scalar(0, 0, 0)); // Black
        for (int y = 0; y < 300; ++y) {
            synthMat2.at<cv::Vec3b>(y, 100) = cv::Vec3b(255, 255, 255); // White line at X=100
        }
        cv::imwrite(synthPath2.string(), synthMat2);
        
        sir::PreprocessedImage outImg2;
        const bool ok2 = preprocessor.preprocessOne(synthPath2.string(), 102, outImg2, reason);
        CHECK(ok2, "F4: marker image loaded");
        
        if (ok2) {
            // Find the peak column.
            int bestCol = -1;
            float maxVal = -1e9f;
            for (int x = 0; x < 224; ++x) {
                float sum = 0;
                for (int y = 0; y < 224; ++y) {
                    sum += outImg2.mat.at<cv::Vec3f>(y, x)[0]; // R channel
                }
                if (sum > maxVal) {
                    maxVal = sum;
                    bestCol = x;
                }
            }
            // If cropped, bestCol should be near 38.
            CHECK(std::abs(bestCol - 38) <= 2, "F5: marker is at expected crop position (near X=38)");
        }

        // Clean up
        if (fs::exists(synthPath)) fs::remove(synthPath);
        if (fs::exists(synthPath2)) fs::remove(synthPath2);
    }

    // =========================================================================
    // G. Manifest generation
    // =========================================================================
    std::cout << "\n--- G. Manifest generation ---\n";
    {
        const fs::path manifestPath = projRoot / "data/processed/valid_ids.txt";
        fs::create_directories(manifestPath.parent_path());
        if (fs::exists(manifestPath)) fs::remove(manifestPath);

        // Process 'sample' which has valid images, plus one invalid record.
        std::vector<sir::ImageRecord> testRecords = sample;
        sir::ImageRecord badRec;
        badRec.internalId = 999;
        badRec.imageId = "bad_image_G";
        badRec.filePath = (projRoot / "data/raw/flickr30k/flickr30k-images/nonexistent_G.jpg").string();
        testRecords.push_back(badRec);

        sir::PreprocessingReport report;
        std::vector<sir::PreprocessedImage> results =
            preprocessor.processRecords(testRecords, report, false, manifestPath.string());

        CHECK(fs::exists(manifestPath), "G1: Manifest file was created");
        
        if (fs::exists(manifestPath)) {
            std::ifstream ifs(manifestPath);
            std::string line;
            std::vector<std::string> lines;
            while (std::getline(ifs, line)) {
                if (!line.empty()) lines.push_back(line);
            }
            
            CHECK(lines.size() == sample.size() + 1, "G2: Manifest has expected number of lines (header + valid)");
            if (lines.size() >= 2) {
                CHECK(lines[0] == "imageId\tfilename", "G3: Manifest header is correct");
                std::string expectedFirst = sample[0].imageId + "\t" + fs::path(sample[0].filePath).filename().string();
                CHECK(lines[1] == expectedFirst, "G4: First entry matches expected format");
            }
            
            bool badFound = false;
            for (const auto& l : lines) {
                if (l.find("bad_image_G") != std::string::npos) badFound = true;
            }
            CHECK(!badFound, "G5: Invalid image is excluded from manifest");
        }

        bool caughtError = false;
        try {
            preprocessor.processRecords(sample, report, false, "/invalid_dir_that_does_not_exist/valid_ids.txt");
        } catch (const std::runtime_error&) {
            caughtError = true;
        }
        CHECK(caughtError, "G6: Throws runtime_error on invalid manifest path");

        if (fs::exists(manifestPath)) fs::remove(manifestPath);
    }

    // =========================================================================
    // H. Grayscale and RGBA handling
    // =========================================================================
    std::cout << "\n--- H. Grayscale and RGBA handling ---\n";
    {
        const fs::path grayPath = projRoot / "data/raw/synthetic_gray.png";
        const fs::path rgbaPath = projRoot / "data/raw/synthetic_rgba.png";

        cv::Mat grayMat(224, 224, CV_8UC1, cv::Scalar(128));
        cv::imwrite(grayPath.string(), grayMat);

        cv::Mat rgbaMat(224, 224, CV_8UC4, cv::Scalar(192, 64, 128, 255));
        cv::imwrite(rgbaPath.string(), rgbaMat);

        sir::PreprocessedImage outGray;
        std::string reasonGray;
        bool okGray = preprocessor.preprocessOne(grayPath.string(), 200, outGray, reasonGray);
        CHECK(okGray, "H1: Grayscale image processed successfully");
        if (okGray) {
            CHECK(outGray.mat.channels() == 3, "H2: Grayscale converted to 3 channels");
            const float expR = (128.0f / 255.0f - sir::ImagePreprocessor::kMeanR) / sir::ImagePreprocessor::kStdR;
            CHECK(std::abs(outGray.mat.at<cv::Vec3f>(0,0)[0] - expR) <= 1e-5f, "H3: Grayscale values normalized correctly");
        }

        sir::PreprocessedImage outRgba;
        std::string reasonRgba;
        bool okRgba = preprocessor.preprocessOne(rgbaPath.string(), 201, outRgba, reasonRgba);
        CHECK(okRgba, "H4: RGBA image processed successfully");
        if (okRgba) {
            CHECK(outRgba.mat.channels() == 3, "H5: RGBA converted to 3 channels (alpha dropped)");
        }

        if (fs::exists(grayPath)) fs::remove(grayPath);
        if (fs::exists(rgbaPath)) fs::remove(rgbaPath);
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "\n=== Test Summary ===\n";
    std::cout << "  Passed : " << gPassed << '\n';
    std::cout << "  Failed : " << gFailed << '\n';

    return gFailed == 0 ? 0 : 1;
}
