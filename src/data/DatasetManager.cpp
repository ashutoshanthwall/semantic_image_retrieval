#include "data/DatasetManager.hpp"

#include <third_party/nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace sir {

// ---------------------------------------------------------------------------
// Supported extensions
// ---------------------------------------------------------------------------

const std::vector<std::string> DatasetManager::kSupportedExtensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/** Convert string to lower-case in-place. */
std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

/** Return true when the extension (lower-cased) is in the supported set. */
bool isSupportedExtension(const std::string& ext,
                           const std::vector<std::string>& supported)
{
    const std::string low = toLower(ext);
    return std::find(supported.begin(), supported.end(), low) != supported.end();
}

/**
 * Derive imageId from a filename.
 *
 * Flickr30k filenames have the form "<numeric-id>.jpg".
 * We strip the extension and require that the stem is non-empty and
 * consists only of alphanumeric characters plus underscores/hyphens
 * (covers the actual Flickr30k naming convention and common variants).
 *
 * Returns an empty string when the stem is invalid.
 */
std::string stemToImageId(const std::string& stem) {
    if (stem.empty()) return {};
    for (char c : stem) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
            return {};
        }
    }
    return stem;
}

/**
 * Lightweight file validation.
 *
 * Confirms:
 * - the path refers to a regular file (not a directory/symlink to dir),
 * - the file size is greater than zero,
 * - the file can be opened for reading.
 *
 * Full pixel-level validation (corrupt JPEG, truncated data) requires
 * OpenCV decoding and belongs to the preprocessing phase.
 */
bool isFileReadable(const fs::path& p, std::string& outReason) {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) {
        outReason = "not a regular file";
        return false;
    }
    const uintmax_t size = fs::file_size(p, ec);
    if (ec || size == 0) {
        outReason = ec ? ec.message() : "file is empty (0 bytes)";
        return false;
    }
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs.is_open()) {
        outReason = "cannot open file for reading";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Caption index built from the Karpathy JSON format
// ---------------------------------------------------------------------------

/**
 * Caption index keyed by filename (e.g. "1000092795.jpg").
 * Each value is a vector of raw sentence strings.
 */
using CaptionIndex = std::unordered_map<std::string, std::vector<std::string>>;

/**
 * Parse the Karpathy-format dataset_flickr30k.json and build a CaptionIndex.
 *
 * Expected top-level structure:
 * {
 *   "images": [
 *     {
 *       "filename": "1000092795.jpg",
 *       "sentences": [
 *         { "raw": "Two young guys ...", ... },
 *         ...
 *       ],
 *       ...
 *     },
 *     ...
 *   ]
 * }
 *
 * The parser is intentionally strict: missing "filename" or "sentences"
 * fields are flagged as malformed entries (counted but not fatal).
 *
 * @param jsonPath        Path to the JSON file.
 * @param outIndex        Populated on success.
 * @param outMalformed    Number of image entries that could not be parsed.
 * @throws std::runtime_error if the file cannot be opened or is not valid JSON.
 */
void buildCaptionIndex(const std::string& jsonPath,
                       CaptionIndex&      outIndex,
                       int&               outMalformed)
{
    outMalformed = 0;

    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        throw std::runtime_error(
            "Cannot open caption JSON file: " + jsonPath);
    }

    json root;
    try {
        ifs >> root;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(
            std::string("JSON parse error in ") + jsonPath + ": " + e.what());
    }

    if (!root.contains("images") || !root["images"].is_array()) {
        throw std::runtime_error(
            "Caption JSON does not contain a top-level 'images' array: "
            + jsonPath);
    }

    for (const auto& imgEntry : root["images"]) {
        if (!imgEntry.contains("filename") || !imgEntry["filename"].is_string()) {
            ++outMalformed;
            continue;
        }
        if (!imgEntry.contains("sentences") || !imgEntry["sentences"].is_array()) {
            ++outMalformed;
            continue;
        }

        const std::string filename = imgEntry["filename"].get<std::string>();
        std::vector<std::string> caps;
        caps.reserve(imgEntry["sentences"].size());

        bool entryOk = true;
        for (const auto& sent : imgEntry["sentences"]) {
            if (!sent.contains("raw") || !sent["raw"].is_string()) {
                // Individual malformed sentence — skip this entry entirely.
                entryOk = false;
                break;
            }
            caps.push_back(sent["raw"].get<std::string>());
        }

        if (!entryOk) {
            ++outMalformed;
            continue;
        }

        if (caps.empty()) {
            ++outMalformed;
            continue;
        }

        outIndex[filename] = std::move(caps);
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// DatasetManager::load
// ---------------------------------------------------------------------------

IngestionReport DatasetManager::load(const std::string& imageDir,
                                     const std::string& captionJson,
                                     bool verbose)
{
    // Reset state.
    records_.clear();
    idMap_.clear();

    IngestionReport report{};

    // -----------------------------------------------------------------------
    // Step 1: Parse caption JSON
    // -----------------------------------------------------------------------
    CaptionIndex captionIndex;
    int jsonMalformed = 0;
    try {
        buildCaptionIndex(captionJson, captionIndex, jsonMalformed);
    } catch (const std::exception& e) {
        std::cerr << "[DatasetManager] FATAL: " << e.what() << '\n';
        return report;
    }

    if (verbose && jsonMalformed > 0) {
        std::cerr << "[DatasetManager] WARNING: " << jsonMalformed
                  << " malformed entries in caption JSON were skipped.\n";
    }

    // -----------------------------------------------------------------------
    // Step 2: Traverse the image directory
    // -----------------------------------------------------------------------
    const fs::path dirPath(imageDir);
    {
        std::error_code ec;
        if (!fs::is_directory(dirPath, ec)) {
            std::cerr << "[DatasetManager] FATAL: image directory does not "
                         "exist or is not a directory: " << imageDir << '\n';
            return report;
        }
    }

    // Collect directory entries (non-recursive; Flickr30k is flat).
    std::vector<fs::directory_entry> entries;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        entries.push_back(entry);
    }

    // Sort for deterministic internalId assignment across runs.
    std::sort(entries.begin(), entries.end(),
              [](const fs::directory_entry& a, const fs::directory_entry& b) {
                  return a.path().filename() < b.path().filename();
              });

    // Track seen imageIds to detect duplicates.
    std::unordered_set<std::string> seenImageIds;

    for (const auto& entry : entries) {
        ++report.totalFilesFound;

        const fs::path& p        = entry.path();
        const std::string fname  = p.filename().string();
        const std::string ext    = p.extension().string();
        const std::string stem   = p.stem().string();

        // --- Extension check ------------------------------------------------
        if (!isSupportedExtension(ext, kSupportedExtensions)) {
            ++report.skippedUnsupported;
            if (verbose) {
                std::cerr << "[DatasetManager] SKIP (unsupported extension '"
                          << ext << "'): " << fname << '\n';
            }
            continue;
        }

        // --- Filename / imageId extraction ----------------------------------
        const std::string imageId = stemToImageId(stem);
        if (imageId.empty()) {
            ++report.skippedInvalidFilename;
            if (verbose) {
                std::cerr << "[DatasetManager] SKIP (invalid filename stem): "
                          << fname << '\n';
            }
            continue;
        }

        // --- Duplicate imageId check ----------------------------------------
        if (seenImageIds.count(imageId)) {
            ++report.skippedDuplicateId;
            if (verbose) {
                std::cerr << "[DatasetManager] SKIP (duplicate imageId '"
                          << imageId << "'): " << fname << '\n';
            }
            continue;
        }

        // --- Lightweight file validation ------------------------------------
        std::string readReason;
        if (!isFileReadable(p, readReason)) {
            ++report.skippedUnreadable;
            if (verbose) {
                std::cerr << "[DatasetManager] SKIP (unreadable — "
                          << readReason << "): " << fname << '\n';
            }
            continue;
        }

        // --- Caption lookup -------------------------------------------------
        auto capIt = captionIndex.find(fname);
        if (capIt == captionIndex.end()) {
            ++report.skippedMissingCaption;
            if (verbose) {
                std::cerr << "[DatasetManager] SKIP (no captions in JSON): "
                          << fname << '\n';
            }
            continue;
        }

        // --- Build ImageRecord ----------------------------------------------
        ImageRecord rec;
        rec.internalId = static_cast<int>(records_.size());
        rec.imageId    = imageId;
        rec.filePath   = p.string();
        rec.width      = 0;   // deferred to OpenCV preprocessing phase
        rec.height     = 0;
        rec.captions   = capIt->second;

        seenImageIds.insert(imageId);
        idMap_[imageId] = rec.internalId;
        records_.push_back(std::move(rec));
        ++report.successCount;
    }

    // -----------------------------------------------------------------------
    // Step 3: Report JSON entries whose image file is absent from disk
    // -----------------------------------------------------------------------
    for (const auto& [captionFname, caps] : captionIndex) {
        // captionFname is the filename as stored in JSON (e.g. "1234.jpg").
        const std::string capStem = fs::path(captionFname).stem().string();
        if (!seenImageIds.count(capStem)) {
            ++report.captionsMissingImage;
            if (verbose) {
                std::cerr << "[DatasetManager] WARNING (caption JSON references "
                             "missing/skipped image): " << captionFname << '\n';
            }
        }
    }

    return report;
}

// ---------------------------------------------------------------------------
// Lookup helpers
// ---------------------------------------------------------------------------

const ImageRecord* DatasetManager::findById(const std::string& imageId) const {
    auto it = idMap_.find(imageId);
    if (it == idMap_.end()) return nullptr;
    return &records_[static_cast<std::size_t>(it->second)];
}

const ImageRecord* DatasetManager::findByInternalId(int internalId) const {
    if (internalId < 0 || internalId >= static_cast<int>(records_.size())) {
        return nullptr;
    }
    return &records_[static_cast<std::size_t>(internalId)];
}

}  // namespace sir
