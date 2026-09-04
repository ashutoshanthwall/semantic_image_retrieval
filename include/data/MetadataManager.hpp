#pragma once

#include "data/ImageRecord.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace sir {

/**
 * Result returned by MetadataManager::load().
 *
 * Records the outcome of the validation pass over an incoming vector of
 * ImageRecords.  All counts are exact; none are estimated or hard-coded.
 */
struct MetadataLoadResult {
    int accepted{0};               ///< records successfully stored
    int rejectedNegativeId{0};     ///< records with internalId < 0
    int rejectedIdMismatch{0};     ///< internalId != expected vector position
    int rejectedEmptyImageId{0};   ///< imageId is an empty string
    int rejectedDuplicateImageId{0}; ///< imageId already accepted in this batch
};

/**
 * Owns the authoritative ImageRecord collection and provides efficient lookup.
 *
 * MetadataManager is the metadata owner for the retrieval system.
 * After DatasetManager produces a validated vector<ImageRecord>, callers
 * transfer ownership to MetadataManager via load().  Downstream components
 * (retrieval indexes, SearchEngine, …) resolve integer internalIds back to
 * ImageRecord data through MetadataManager without coupling to DatasetManager
 * or any other Phase-1 ingestion detail.
 *
 * Design invariant enforced on every stored record:
 *   records_[i].internalId == static_cast<int>(i)   for all valid i
 *
 * Structures:
 *   std::vector<ImageRecord>            — ordered record store; O(1) index access
 *   std::unordered_map<std::string,int> — imageId → internalId; average O(1) lookup
 *
 * Not implemented here (out of scope for Phase 2):
 *   OpenCV, CLIP, embeddings, VectorMath, similarity, BruteForceIndex,
 *   HNSW, SearchEngine, persistence, benchmarking, CLI.
 */
class MetadataManager {
public:
    MetadataManager()  = default;
    ~MetadataManager() = default;

    // Non-copyable: the record collection can be large; require explicit move.
    MetadataManager(const MetadataManager&)            = delete;
    MetadataManager& operator=(const MetadataManager&) = delete;

    // Movable.
    MetadataManager(MetadataManager&&)            = default;
    MetadataManager& operator=(MetadataManager&&) = default;

    // -----------------------------------------------------------------------
    // Loading
    // -----------------------------------------------------------------------

    /**
     * Load a vector of ImageRecords into MetadataManager with transactional
     * and strict-validation semantics.
     *
     * Two-pass algorithm:
     *   Pass 1 — validate every record in the incoming vector WITHOUT
     *             modifying any MetadataManager state.
     *   Pass 2 — only if Pass 1 finds zero violations:
     *             clear the old state, move the entire vector in,
     *             and build the imageId → internalId map.
     *
     * If ANY record fails validation:
     *   - the MetadataLoadResult counts indicate which records failed and why
     *   - result.accepted == 0
     *   - NO records from this batch are inserted
     *   - NO internalId is modified or renumbered
     *   - the MetadataManager retains exactly the state it had before the call
     *   - the caller's vector is NOT partially moved (Pass 1 is read-only)
     *
     * If ALL records pass validation:
     *   - the incoming vector is moved into MetadataManager (O(1))
     *   - every ImageRecord field is preserved exactly as received
     *   - internalId is NEVER modified; the caller's IDs are the stable identity
     *   - the invariant records_[i].internalId == i holds because the caller
     *     must supply a correctly sequenced vector (validated by check 2)
     *   - result.accepted == (int)incoming.size()
     *
     * Validation rules applied per record:
     *   1. internalId >= 0
     *   2. internalId == its 0-based position in the incoming vector
     *   3. imageId is non-empty
     *   4. imageId is unique within this batch
     *
     * width == 0 and height == 0 are explicitly NOT errors: Phase 1
     * intentionally leaves dimensions unset; full decoding belongs to the
     * OpenCV preprocessing phase.
     *
     * @param records  Source vector; moved in on success; untouched on failure.
     * @param verbose  When true, each per-record rejection is printed to
     *                 std::cerr along with a batch-level summary.
     * @return MetadataLoadResult with exact per-category counts.
     */
    MetadataLoadResult load(std::vector<ImageRecord> records,
                            bool verbose = false);

    // -----------------------------------------------------------------------
    // Observation
    // -----------------------------------------------------------------------

    /** Number of successfully stored records. */
    int size() const { return static_cast<int>(records_.size()); }

    /** True when no records have been loaded. */
    bool empty() const { return records_.empty(); }

    /**
     * Read-only access to the entire record collection.
     * Iterating over this vector yields records in internalId order.
     */
    const std::vector<ImageRecord>& records() const { return records_; }

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /**
     * Look up a record by internalId.
     *
     * Runs in O(1) time via vector index.
     * Returns nullptr for negative IDs or IDs >= size().
     * Does not copy the record.
     */
    const ImageRecord* findByInternalId(int internalId) const;

    /**
     * Look up a record by Flickr imageId string.
     *
     * Runs in average O(1) time via std::unordered_map.
     * Returns nullptr when the imageId is not present.
     * Does not copy the record.
     */
    const ImageRecord* findByImageId(const std::string& imageId) const;

private:
    std::vector<ImageRecord>            records_;
    std::unordered_map<std::string,int> imageIdMap_; ///< imageId → internalId
};

}  // namespace sir
