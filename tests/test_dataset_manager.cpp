/**
 * test_dataset_manager.cpp
 *
 * Integration test for DatasetManager against the actual Flickr30k dataset.
 *
 * Paths are expressed relative to the project root, resolved at runtime
 * from the working directory.  Run this executable from the project root
 * (or supply SIR_PROJECT_ROOT env var) so the relative paths resolve.
 *
 * What is tested:
 *  1. load() returns without throwing.
 *  2. successCount > 0.
 *  3. records() size == successCount.
 *  4. Every record has a non-empty imageId, filePath, and ≥1 caption.
 *  5. Every record's internalId equals its position in the vector.
 *  6. idMap() round-trips: imageId → internalId → record.imageId is consistent.
 *  7. findById() and findByInternalId() return correct results.
 *  8. findById() with an unknown ID returns nullptr.
 *  9. findByInternalId() out-of-range returns nullptr.
 * 10. IngestionReport is printed so actual numbers can be verified by the user.
 *
 * No result values are hard-coded or fabricated.
 */

#include "data/DatasetManager.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Minimal test harness (no external framework dependency)
// ---------------------------------------------------------------------------

static int gTestsPassed = 0;
static int gTestsFailed = 0;

#define CHECK(cond, msg)                                             \
    do {                                                             \
        if (!(cond)) {                                               \
            std::cerr << "[FAIL] " << (msg) << '\n';                \
            ++gTestsFailed;                                          \
        } else {                                                     \
            std::cout << "[PASS] " << (msg) << '\n';                \
            ++gTestsPassed;                                          \
        }                                                            \
    } while (false)

// ---------------------------------------------------------------------------

int main(int /*argc*/, char** /*argv*/) {

    // Resolve project root: prefer SIR_PROJECT_ROOT env var, else use cwd.
    const char* envRoot = std::getenv("SIR_PROJECT_ROOT");
    const fs::path projectRoot = envRoot ? fs::path(envRoot) : fs::current_path();

    const fs::path imageDir    = projectRoot
        / "data/raw/flickr30k/flickr30k-images";
    const fs::path captionJson = projectRoot
        / "data/raw/flickr30k/dataset_flickr30k.json";

    std::cout << "=== DatasetManager Integration Test ===\n";
    std::cout << "Project root : " << projectRoot << '\n';
    std::cout << "Image dir    : " << imageDir    << '\n';
    std::cout << "Caption JSON : " << captionJson << "\n\n";

    // ------------------------------------------------------------------
    // T1: Paths exist
    // ------------------------------------------------------------------
    CHECK(fs::is_directory(imageDir),
          "T1a: image directory exists");
    CHECK(fs::is_regular_file(captionJson),
          "T1b: caption JSON file exists");

    if (gTestsFailed > 0) {
        std::cerr << "\nPre-flight check failed — cannot proceed.\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // T2: load() does not throw and returns a report
    // ------------------------------------------------------------------
    sir::DatasetManager dm;
    sir::IngestionReport report{};
    bool loadThrew = false;

    try {
        // verbose=false to keep test output readable; problems still counted
        report = dm.load(imageDir.string(), captionJson.string(),
                         /*verbose=*/false);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] load() threw: " << e.what() << '\n';
        loadThrew = true;
    }

    CHECK(!loadThrew, "T2: load() does not throw");

    if (loadThrew) {
        std::cerr << "Cannot continue after fatal load() failure.\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // T3: Ingestion report — print actual numbers, check basic sanity
    // ------------------------------------------------------------------
    std::cout << "\n--- Ingestion Report ---\n";
    std::cout << "  totalFilesFound       : " << report.totalFilesFound       << '\n';
    std::cout << "  successCount          : " << report.successCount          << '\n';
    std::cout << "  skippedUnsupported    : " << report.skippedUnsupported    << '\n';
    std::cout << "  skippedUnreadable     : " << report.skippedUnreadable     << '\n';
    std::cout << "  skippedDuplicateId    : " << report.skippedDuplicateId    << '\n';
    std::cout << "  skippedMissingCaption : " << report.skippedMissingCaption << '\n';
    std::cout << "  skippedInvalidFilename: " << report.skippedInvalidFilename<< '\n';
    std::cout << "  captionsMissingImage  : " << report.captionsMissingImage  << '\n';
    std::cout << "---\n\n";

    CHECK(report.successCount > 0,
          "T3a: at least one record was loaded");
    CHECK(report.skippedDuplicateId == 0,
          "T3b: no duplicate imageIds detected");

    // ------------------------------------------------------------------
    // T4: records() size matches successCount
    // ------------------------------------------------------------------
    const auto& recs = dm.records();
    CHECK(static_cast<int>(recs.size()) == report.successCount,
          "T4: records().size() == successCount");

    // ------------------------------------------------------------------
    // T5: Each record has valid internalId, imageId, filePath, captions
    // ------------------------------------------------------------------
    int badInternalId   = 0;
    int emptyImageId    = 0;
    int emptyFilePath   = 0;
    int noCaptions      = 0;
    int emptyCaption    = 0;
    int fileNotOnDisk   = 0;

    for (std::size_t i = 0; i < recs.size(); ++i) {
        const auto& r = recs[i];
        if (static_cast<std::size_t>(r.internalId) != i) ++badInternalId;
        if (r.imageId.empty())           ++emptyImageId;
        if (r.filePath.empty())          ++emptyFilePath;
        if (r.captions.empty())          ++noCaptions;
        for (const auto& c : r.captions) {
            if (c.empty()) ++emptyCaption;
        }
        if (!fs::is_regular_file(r.filePath)) ++fileNotOnDisk;
    }

    CHECK(badInternalId == 0,
          "T5a: all internalIds equal their position in records()");
    CHECK(emptyImageId == 0,
          "T5b: no record has an empty imageId");
    CHECK(emptyFilePath == 0,
          "T5c: no record has an empty filePath");
    CHECK(noCaptions == 0,
          "T5d: every record has at least one caption");
    CHECK(emptyCaption == 0,
          "T5e: no caption string is empty");
    CHECK(fileNotOnDisk == 0,
          "T5f: every record's filePath exists on disk");

    // ------------------------------------------------------------------
    // T6: idMap() round-trip consistency
    // ------------------------------------------------------------------
    const auto& idMap = dm.idMap();
    CHECK(static_cast<int>(idMap.size()) == report.successCount,
          "T6a: idMap size == successCount");

    int idMapMismatch = 0;
    for (const auto& [imgId, internalId] : idMap) {
        if (internalId < 0 || internalId >= static_cast<int>(recs.size())) {
            ++idMapMismatch;
            continue;
        }
        if (recs[static_cast<std::size_t>(internalId)].imageId != imgId) ++idMapMismatch;
    }
    CHECK(idMapMismatch == 0,
          "T6b: idMap values point to matching records");

    // ------------------------------------------------------------------
    // T7: findById() — spot-check first and last record
    // ------------------------------------------------------------------
    if (!recs.empty()) {
        const std::string firstId = recs.front().imageId;
        const std::string lastId  = recs.back().imageId;

        const sir::ImageRecord* pFirst = dm.findById(firstId);
        CHECK(pFirst != nullptr && pFirst->imageId == firstId,
              "T7a: findById(first) returns correct record");

        const sir::ImageRecord* pLast = dm.findById(lastId);
        CHECK(pLast != nullptr && pLast->imageId == lastId,
              "T7b: findById(last) returns correct record");
    }

    // ------------------------------------------------------------------
    // T8: findById() with unknown ID returns nullptr
    // ------------------------------------------------------------------
    CHECK(dm.findById("__nonexistent_id__") == nullptr,
          "T8: findById(unknown) returns nullptr");

    // ------------------------------------------------------------------
    // T9: findByInternalId() — spot-check and out-of-range
    // ------------------------------------------------------------------
    if (!recs.empty()) {
        const sir::ImageRecord* p = dm.findByInternalId(0);
        CHECK(p != nullptr && p->internalId == 0,
              "T9a: findByInternalId(0) returns first record");

        const int last = static_cast<int>(recs.size()) - 1;
        const sir::ImageRecord* pL = dm.findByInternalId(last);
        CHECK(pL != nullptr && pL->internalId == last,
              "T9b: findByInternalId(last) returns last record");
    }
    CHECK(dm.findByInternalId(-1) == nullptr,
          "T9c: findByInternalId(-1) returns nullptr");
    CHECK(dm.findByInternalId(static_cast<int>(recs.size())) == nullptr,
          "T9d: findByInternalId(size) returns nullptr");

    // ------------------------------------------------------------------
    // T10: Sample a few records and print for manual inspection
    // ------------------------------------------------------------------
    std::cout << "\n--- Sample Records (first 3) ---\n";
    for (std::size_t i = 0; i < std::min(std::size_t{3}, recs.size()); ++i) {
        const auto& r = recs[i];
        std::cout << "  [" << r.internalId << "] imageId=" << r.imageId
                  << "  captions=" << r.captions.size()
                  << "  path=" << r.filePath << '\n';
        if (!r.captions.empty()) {
            std::cout << "    caption[0]: " << r.captions[0] << '\n';
        }
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    std::cout << "\n=== Test Summary ===\n";
    std::cout << "  Passed : " << gTestsPassed << '\n';
    std::cout << "  Failed : " << gTestsFailed << '\n';

    return gTestsFailed == 0 ? 0 : 1;
}
