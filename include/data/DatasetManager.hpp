#pragma once

#include "data/ImageRecord.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace sir {

/**
 * Ingestion summary returned by DatasetManager::load().
 *
 * All counts are exact values measured from the actual filesystem traversal
 * and JSON parse; none are estimated or hard-coded.
 */
struct IngestionReport {
    int totalFilesFound{0};        ///< files encountered during traversal
    int successCount{0};           ///< ImageRecords successfully built
    int skippedUnsupported{0};     ///< files with an unsupported extension
    int skippedUnreadable{0};      ///< files that could not be opened/read
    int skippedDuplicateId{0};     ///< files whose imageId was already seen
    int skippedMissingCaption{0};  ///< files present on disk but absent from JSON
    int skippedInvalidFilename{0}; ///< filenames that don't yield a valid imageId
    int captionsMissingImage{0};   ///< JSON entries whose file is absent from disk
};

/**
 * Manages dataset discovery, validation, and ImageRecord construction.
 *
 * DatasetManager traverses an image directory, validates each file
 * (existence, readability, supported extension, non-duplicate imageId),
 * cross-references captions from a Karpathy-format JSON annotation file,
 * and assembles a vector of ImageRecord objects with sequential internalIds.
 *
 * Detected problems are reported via IngestionReport and, optionally,
 * written to std::cerr.  Invalid files are never silently ignored.
 *
 * Phase 1 note:
 *   File validation in this phase is "lightweight": we confirm the file
 *   is readable and has a non-zero size.  Pixel-level validation (corrupt
 *   JPEG data, truncated images) requires full OpenCV decoding, which
 *   belongs to the preprocessing phase.
 *
 * Structures used:
 *   std::vector<ImageRecord>            — ordered collection of valid records
 *   std::unordered_map<std::string,int> — imageId → internalId (O(1) lookup)
 *   std::unordered_map<std::string,…>   — JSON caption index keyed by filename
 */
class DatasetManager {
public:
    /**
     * Supported image extensions (lower-case, including the dot).
     * Only files whose extension matches this set are processed.
     */
    static const std::vector<std::string> kSupportedExtensions;

    DatasetManager() = default;
    ~DatasetManager() = default;

    // Non-copyable, movable.
    DatasetManager(const DatasetManager&) = delete;
    DatasetManager& operator=(const DatasetManager&) = delete;
    DatasetManager(DatasetManager&&) = default;
    DatasetManager& operator=(DatasetManager&&) = default;

    /**
     * Load the dataset.
     *
     * @param imageDir   Path to the directory containing image files.
     *                   Traversal is non-recursive (Flickr30k stores all
     *                   images in a single flat directory).
     * @param captionJson Path to the Karpathy-format JSON annotation file
     *                    (dataset_flickr30k.json).
     * @param verbose    When true, each individual problem is printed to
     *                   std::cerr in addition to being counted in the report.
     * @return IngestionReport with exact counts of what was found and skipped.
     *
     * After a successful call, use records() and idMap() to access results.
     * Calling load() again clears and replaces any previous state.
     */
    IngestionReport load(const std::string& imageDir,
                         const std::string& captionJson,
                         bool verbose = false);

    /** All successfully built ImageRecord objects, in internalId order. */
    const std::vector<ImageRecord>& records() const { return records_; }

    /**
     * Map from Flickr imageId string to internalId integer.
     * Average-case O(1) lookup via std::unordered_map.
     */
    const std::unordered_map<std::string, int>& idMap() const { return idMap_; }

    /**
     * Look up a single record by Flickr imageId.
     * Returns nullptr if the imageId is not present.
     */
    const ImageRecord* findById(const std::string& imageId) const;

    /**
     * Look up a single record by internalId.
     * Returns nullptr if internalId is out of range.
     */
    const ImageRecord* findByInternalId(int internalId) const;

private:
    std::vector<ImageRecord>            records_;
    std::unordered_map<std::string,int> idMap_;  ///< imageId → internalId
};

}  // namespace sir
