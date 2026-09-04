#include "data/MetadataManager.hpp"

#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sir {

// ---------------------------------------------------------------------------
// MetadataManager::load
//
// Transactional semantics:
//   Pass 1 — validate every record in `incoming` WITHOUT modifying any
//             MetadataManager state or moving records.
//   Pass 2 — only if Pass 1 found zero violations:
//             clear the old state, move `incoming` into records_, build map.
//
// If any record fails validation the entire batch is refused: no records are
// inserted, no IDs are renumbered, and the MetadataManager retains whatever
// state it held before the call.
// ---------------------------------------------------------------------------

MetadataLoadResult MetadataManager::load(std::vector<ImageRecord> incoming,
                                         bool verbose)
{
    MetadataLoadResult result{};

    // -----------------------------------------------------------------------
    // Pass 1: validate — do NOT touch records_ or imageIdMap_
    // -----------------------------------------------------------------------
    std::unordered_set<std::string> seenImageIds;
    seenImageIds.reserve(incoming.size());

    for (std::size_t idx = 0; idx < incoming.size(); ++idx) {
        const ImageRecord& rec = incoming[idx];

        // Validation 1: internalId must be non-negative.
        if (rec.internalId < 0) {
            ++result.rejectedNegativeId;
            if (verbose) {
                std::cerr << "[MetadataManager] REJECT (negative internalId "
                          << rec.internalId << " at position " << idx
                          << "): imageId='" << rec.imageId << "'\n";
            }
            continue;
        }

        // Validation 2: internalId must equal its position in the vector.
        // internalId is a stable identity set by DatasetManager; MetadataManager
        // must not renumber it.  A mismatch signals a malformed batch.
        if (rec.internalId != static_cast<int>(idx)) {
            ++result.rejectedIdMismatch;
            if (verbose) {
                std::cerr << "[MetadataManager] REJECT (internalId "
                          << rec.internalId
                          << " != position " << idx
                          << "): imageId='" << rec.imageId << "'\n";
            }
            continue;
        }

        // Validation 3: imageId must not be empty.
        if (rec.imageId.empty()) {
            ++result.rejectedEmptyImageId;
            if (verbose) {
                std::cerr << "[MetadataManager] REJECT (empty imageId)"
                          << " at position " << idx << '\n';
            }
            continue;
        }

        // Validation 4: imageId must be unique within this batch.
        if (seenImageIds.count(rec.imageId)) {
            ++result.rejectedDuplicateImageId;
            if (verbose) {
                std::cerr << "[MetadataManager] REJECT (duplicate imageId '"
                          << rec.imageId << "') at position " << idx << '\n';
            }
            continue;
        }

        seenImageIds.insert(rec.imageId);
    }

    // -----------------------------------------------------------------------
    // Decision: if any violation was found, refuse the entire batch.
    // The MetadataManager state is left exactly as it was.
    // -----------------------------------------------------------------------
    const int totalRejected = result.rejectedNegativeId
                            + result.rejectedIdMismatch
                            + result.rejectedEmptyImageId
                            + result.rejectedDuplicateImageId;

    if (totalRejected > 0) {
        // result already holds the per-category counts; accepted stays 0.
        if (verbose) {
            std::cerr << "[MetadataManager] Batch rejected: "
                      << totalRejected << " invalid record(s). "
                      << "MetadataManager state is unchanged.\n";
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Pass 2: all records are valid — commit.
    // Clear old state, then move the incoming vector in its entirety.
    // No internalId is modified; fields are preserved exactly.
    // -----------------------------------------------------------------------
    records_.clear();
    imageIdMap_.clear();

    records_ = std::move(incoming);

    imageIdMap_.reserve(records_.size());
    for (const ImageRecord& rec : records_) {
        imageIdMap_[rec.imageId] = rec.internalId;
    }

    result.accepted = static_cast<int>(records_.size());
    return result;
}

// ---------------------------------------------------------------------------
// MetadataManager::findByInternalId
// ---------------------------------------------------------------------------

const ImageRecord* MetadataManager::findByInternalId(int internalId) const {
    if (internalId < 0 || internalId >= static_cast<int>(records_.size())) {
        return nullptr;
    }
    return &records_[static_cast<std::size_t>(internalId)];
}

// ---------------------------------------------------------------------------
// MetadataManager::findByImageId
// ---------------------------------------------------------------------------

const ImageRecord* MetadataManager::findByImageId(const std::string& imageId) const {
    const auto it = imageIdMap_.find(imageId);
    if (it == imageIdMap_.end()) {
        return nullptr;
    }
    return &records_[static_cast<std::size_t>(it->second)];
}

}  // namespace sir
