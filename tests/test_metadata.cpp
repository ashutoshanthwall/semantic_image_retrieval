/**
 * test_metadata.cpp
 *
 * Tests for MetadataManager (Phase 2).
 *
 * Covers:
 *   A. Empty MetadataManager
 *   B. Synthetic valid metadata
 *   C. Invalid metadata — strict transactional rejection
 *   D. Integration with Phase 1 (actual Flickr30k via DatasetManager)
 *
 * Key invariants under test:
 *   - load() is all-or-nothing: a batch with ANY invalid record is entirely
 *     refused; no partial insertion occurs
 *   - MetadataManager state is unchanged after a failed load()
 *   - internalIds are NEVER renumbered or modified by MetadataManager
 *   - records_[i].internalId == i holds for every stored record (because
 *     the incoming vector must be correctly sequenced — validated by load())
 *
 * No OpenCV, CLIP, embeddings, or network calls.
 * No hard-coded dataset counts (all counts are compared dynamically).
 * No result values are fabricated.
 */

#include "data/DatasetManager.hpp"
#include "data/ImageRecord.hpp"
#include "data/MetadataManager.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Minimal test harness
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
// Helper: build a synthetic ImageRecord
// ---------------------------------------------------------------------------
static sir::ImageRecord makeRecord(int id,
                                   const std::string& imageId,
                                   const std::string& path = "/fake/path",
                                   int w = 0, int h = 0,
                                   std::vector<std::string> caps = {"cap"})
{
    sir::ImageRecord r;
    r.internalId = id;
    r.imageId    = imageId;
    r.filePath   = path;
    r.width      = w;
    r.height     = h;
    r.captions   = std::move(caps);
    return r;
}

// ===========================================================================
// A. Empty MetadataManager
// ===========================================================================
static void testEmptyManager() {
    std::cout << "\n--- A. Empty MetadataManager ---\n";

    sir::MetadataManager mm;

    CHECK(mm.empty(),           "A1: empty() is true on default-constructed manager");
    CHECK(mm.size() == 0,       "A2: size() == 0 on default-constructed manager");
    CHECK(mm.records().empty(), "A3: records() is empty on default-constructed manager");

    // Lookups must not crash and must return nullptr.
    CHECK(mm.findByInternalId(-1)  == nullptr, "A4: findByInternalId(-1) returns nullptr");
    CHECK(mm.findByInternalId(0)   == nullptr, "A5: findByInternalId(0) returns nullptr (empty)");
    CHECK(mm.findByInternalId(100) == nullptr, "A6: findByInternalId(100) returns nullptr (empty)");
    CHECK(mm.findByImageId("")     == nullptr, "A7: findByImageId(\"\") returns nullptr");
    CHECK(mm.findByImageId("999")  == nullptr, "A8: findByImageId(\"999\") returns nullptr");
}

// ===========================================================================
// B. Synthetic valid metadata
// ===========================================================================
static void testValidMetadata() {
    std::cout << "\n--- B. Synthetic valid metadata ---\n";

    std::vector<sir::ImageRecord> recs;
    recs.push_back(makeRecord(0, "111", "/img/a.jpg", 0, 0,
                               {"First caption A", "Second caption A"}));
    recs.push_back(makeRecord(1, "222", "/img/b.jpg", 640, 480,
                               {"Caption B"}));
    recs.push_back(makeRecord(2, "333", "/img/c.jpg", 1920, 1080,
                               {"Caption C1", "Caption C2", "Caption C3"}));

    sir::MetadataManager mm;
    sir::MetadataLoadResult res = mm.load(std::move(recs));

    // Result counts
    CHECK(res.accepted == 3,                  "B1: load() accepted 3 records");
    CHECK(res.rejectedNegativeId == 0,        "B2: no negative-id rejections");
    CHECK(res.rejectedIdMismatch == 0,        "B3: no id-mismatch rejections");
    CHECK(res.rejectedEmptyImageId == 0,      "B4: no empty-imageId rejections");
    CHECK(res.rejectedDuplicateImageId == 0,  "B5: no duplicate-imageId rejections");
    CHECK(mm.size() == 3,                     "B6: size() == 3");
    CHECK(!mm.empty(),                        "B7: empty() is false");

    // Invariant: records_[i].internalId == i
    const auto& stored = mm.records();
    bool invariantOk = true;
    for (int i = 0; i < mm.size(); ++i) {
        if (stored[static_cast<std::size_t>(i)].internalId != i) {
            invariantOk = false;
        }
    }
    CHECK(invariantOk, "B8: invariant records_[i].internalId == i holds for all i");

    // findByInternalId spot-checks
    const sir::ImageRecord* p0 = mm.findByInternalId(0);
    CHECK(p0 != nullptr,                        "B9a: findByInternalId(0) is not null");
    CHECK(p0->imageId == "111",                 "B9b: findByInternalId(0) -> imageId");
    CHECK(p0->filePath == "/img/a.jpg",         "B9c: filePath preserved exactly");
    CHECK(p0->width == 0,                       "B9d: width==0 preserved (Phase 1 boundary)");
    CHECK(p0->height == 0,                      "B9e: height==0 preserved (Phase 1 boundary)");
    CHECK(p0->captions.size() == 2,             "B9f: captions.size()==2 preserved");
    CHECK(p0->captions[0] == "First caption A", "B9g: captions[0] preserved");
    CHECK(p0->captions[1] == "Second caption A","B9h: captions[1] preserved");
    CHECK(p0->internalId == 0,                  "B9i: internalId not renumbered (stays 0)");

    const sir::ImageRecord* p1 = mm.findByInternalId(1);
    CHECK(p1 != nullptr,         "B10a: findByInternalId(1) is not null");
    CHECK(p1->imageId == "222",  "B10b: findByInternalId(1) -> imageId");
    CHECK(p1->width == 640,      "B10c: width preserved");
    CHECK(p1->height == 480,     "B10d: height preserved");
    CHECK(p1->internalId == 1,   "B10e: internalId not renumbered (stays 1)");

    const sir::ImageRecord* p2 = mm.findByInternalId(2);
    CHECK(p2 != nullptr,                   "B11a: findByInternalId(2) is not null");
    CHECK(p2->captions.size() == 3,        "B11b: 3 captions preserved");
    CHECK(p2->captions[2] == "Caption C3", "B11c: captions[2] preserved");
    CHECK(p2->internalId == 2,             "B11d: internalId not renumbered (stays 2)");

    // Out-of-range
    CHECK(mm.findByInternalId(-1) == nullptr, "B12: findByInternalId(-1) nullptr");
    CHECK(mm.findByInternalId(3)  == nullptr, "B13: findByInternalId(size) nullptr");

    // findByImageId
    const sir::ImageRecord* q0 = mm.findByImageId("111");
    CHECK(q0 != nullptr,       "B14a: findByImageId(\"111\") not null");
    CHECK(q0->internalId == 0, "B14b: findByImageId(\"111\") -> internalId==0");

    const sir::ImageRecord* q2 = mm.findByImageId("333");
    CHECK(q2 != nullptr,       "B15a: findByImageId(\"333\") not null");
    CHECK(q2->internalId == 2, "B15b: findByImageId(\"333\") -> internalId==2");

    // Unknown imageId
    CHECK(mm.findByImageId("__unknown__") == nullptr,
          "B16: findByImageId(unknown) returns nullptr");

    // Records collection size
    CHECK(static_cast<int>(mm.records().size()) == mm.size(),
          "B17: records().size() == size()");

    // Returned pointer aliases stored record (no copy made)
    CHECK(p0 == &mm.records()[0], "B18: findByInternalId returns reference into records_");

    // Confirm valid second load() replaces state
    std::vector<sir::ImageRecord> recs2;
    recs2.push_back(makeRecord(0, "alpha", "/img/alpha.jpg"));
    const sir::MetadataLoadResult res2 = mm.load(std::move(recs2));
    CHECK(res2.accepted == 1,                    "B19a: second valid load() accepted 1 record");
    CHECK(mm.size() == 1,                        "B19b: size() == 1 after second load");
    CHECK(mm.findByImageId("111") == nullptr,    "B19c: old imageId gone after reload");
    CHECK(mm.findByImageId("alpha") != nullptr,  "B19d: new imageId present after reload");
}

// ===========================================================================
// C. Invalid metadata — strict transactional rejection
//
// Key property under test: a batch with ANY invalid record is entirely
// refused.  No records are inserted, and no IDs are renumbered.
// The MetadataManager's pre-existing state is preserved unchanged.
// ===========================================================================
static void testInvalidMetadata() {
    std::cout << "\n--- C. Invalid metadata (transactional rejection) ---\n";

    // -----------------------------------------------------------------------
    // C1: single negative internalId — entire batch refused, manager stays empty
    // -----------------------------------------------------------------------
    {
        std::vector<sir::ImageRecord> recs;
        recs.push_back(makeRecord(-1, "aaa"));

        sir::MetadataManager mm;
        sir::MetadataLoadResult res = mm.load(std::move(recs), /*verbose=*/false);

        CHECK(res.rejectedNegativeId == 1, "C1a: negative internalId counted");
        CHECK(res.accepted == 0,           "C1b: accepted == 0 (whole batch refused)");
        CHECK(mm.size() == 0,              "C1c: manager remains empty");
        CHECK(mm.empty(),                  "C1d: empty() is true");
    }

    // -----------------------------------------------------------------------
    // C2: internalId != position in vector — entire batch refused
    //     No re-stamping: the valid records retain their original IDs.
    // -----------------------------------------------------------------------
    {
        // Batch: pos-0 id=0 OK, pos-1 id=5 MISMATCH, pos-2 id=2 OK
        // The mismatch invalidates the whole batch.
        std::vector<sir::ImageRecord> recs;
        recs.push_back(makeRecord(0, "aaa"));   // OK
        recs.push_back(makeRecord(5, "bbb"));   // MISMATCH: id=5 != pos=1
        recs.push_back(makeRecord(2, "ccc"));   // OK

        sir::MetadataManager mm;
        sir::MetadataLoadResult res = mm.load(std::move(recs), /*verbose=*/false);

        CHECK(res.rejectedIdMismatch == 1, "C2a: one id-mismatch counted");
        CHECK(res.accepted == 0,           "C2b: accepted == 0 (whole batch refused)");
        CHECK(mm.size() == 0,              "C2c: manager is empty — no partial insertion");
        CHECK(mm.findByImageId("aaa") == nullptr,
              "C2d: 'aaa' not inserted (whole batch refused)");
        CHECK(mm.findByImageId("ccc") == nullptr,
              "C2e: 'ccc' not inserted (whole batch refused)");
    }

    // -----------------------------------------------------------------------
    // C3: duplicate imageId — entire batch refused
    // -----------------------------------------------------------------------
    {
        std::vector<sir::ImageRecord> recs;
        recs.push_back(makeRecord(0, "dup"));
        recs.push_back(makeRecord(1, "dup"));   // duplicate
        recs.push_back(makeRecord(2, "uniq"));

        sir::MetadataManager mm;
        sir::MetadataLoadResult res = mm.load(std::move(recs), /*verbose=*/false);

        CHECK(res.rejectedDuplicateImageId == 1, "C3a: one duplicate counted");
        CHECK(res.accepted == 0,                 "C3b: accepted == 0 (whole batch refused)");
        CHECK(mm.size() == 0,                    "C3c: manager is empty");
        CHECK(mm.findByImageId("dup")  == nullptr,"C3d: 'dup' not inserted");
        CHECK(mm.findByImageId("uniq") == nullptr,"C3e: 'uniq' not inserted");
    }

    // -----------------------------------------------------------------------
    // C4: empty imageId — entire batch refused
    // -----------------------------------------------------------------------
    {
        std::vector<sir::ImageRecord> recs;
        recs.push_back(makeRecord(0, "good"));
        recs.push_back(makeRecord(1, ""));      // empty imageId

        sir::MetadataManager mm;
        sir::MetadataLoadResult res = mm.load(std::move(recs), /*verbose=*/false);

        CHECK(res.rejectedEmptyImageId == 1, "C4a: empty imageId counted");
        CHECK(res.accepted == 0,             "C4b: accepted == 0 (whole batch refused)");
        CHECK(mm.size() == 0,                "C4c: manager is empty");
        CHECK(mm.findByImageId("good") == nullptr, "C4d: 'good' not inserted");
    }

    // -----------------------------------------------------------------------
    // C5: mixed invalid batch — all four violation types present
    //     Entire batch is refused; accepted == 0.
    //     internalIds of "valid" records within the batch must NOT be
    //     renumbered by MetadataManager.
    // -----------------------------------------------------------------------
    {
        std::vector<sir::ImageRecord> recs;
        recs.push_back(makeRecord( 0, "good1"));    // pos 0 — valid
        recs.push_back(makeRecord(-1, "bad_neg"));  // pos 1 — negative id
        recs.push_back(makeRecord( 7, "bad_mis"));  // pos 2 — mismatch (7 != 2)
        recs.push_back(makeRecord( 3, ""));         // pos 3 — empty imageId
        recs.push_back(makeRecord( 4, "good1"));    // pos 4 — duplicate of pos 0
        recs.push_back(makeRecord( 5, "good2"));    // pos 5 — valid

        sir::MetadataManager mm;
        sir::MetadataLoadResult res = mm.load(std::move(recs), /*verbose=*/false);

        CHECK(res.rejectedNegativeId == 1,        "C5a: 1 negative-id");
        CHECK(res.rejectedIdMismatch == 1,         "C5b: 1 id-mismatch");
        CHECK(res.rejectedEmptyImageId == 1,       "C5c: 1 empty-imageId");
        CHECK(res.rejectedDuplicateImageId == 1,   "C5d: 1 duplicate-imageId");
        CHECK(res.accepted == 0,                   "C5e: accepted == 0 (entire batch refused)");
        CHECK(mm.size() == 0,                      "C5f: manager is empty — no partial insertion");
        CHECK(mm.findByImageId("good1") == nullptr,"C5g: 'good1' not inserted");
        CHECK(mm.findByImageId("good2") == nullptr,"C5h: 'good2' not inserted");
    }

    // -----------------------------------------------------------------------
    // C6: old MetadataManager state is PRESERVED after a failed load()
    //     This is the core transactional guarantee.
    // -----------------------------------------------------------------------
    {
        // First: load a valid batch so mm has some state.
        std::vector<sir::ImageRecord> goodBatch;
        goodBatch.push_back(makeRecord(0, "alpha"));
        goodBatch.push_back(makeRecord(1, "beta"));

        sir::MetadataManager mm;
        sir::MetadataLoadResult r1 = mm.load(std::move(goodBatch));
        CHECK(r1.accepted == 2,    "C6a: first (valid) load accepted 2 records");
        CHECK(mm.size() == 2,      "C6b: mm has 2 records after valid load");

        // Now attempt to load an invalid batch (mismatch at position 1).
        std::vector<sir::ImageRecord> badBatch;
        badBatch.push_back(makeRecord(0, "gamma"));
        badBatch.push_back(makeRecord(9, "delta")); // id=9 != pos=1

        sir::MetadataLoadResult r2 = mm.load(std::move(badBatch), /*verbose=*/false);

        CHECK(r2.rejectedIdMismatch == 1, "C6c: failed load() counted 1 id-mismatch");
        CHECK(r2.accepted == 0,           "C6d: failed load() accepted 0");

        // mm must still contain the original 2 records exactly.
        CHECK(mm.size() == 2,                    "C6e: mm still has 2 records (state preserved)");
        CHECK(mm.findByImageId("alpha") != nullptr,"C6f: 'alpha' still present after failed load");
        CHECK(mm.findByImageId("beta")  != nullptr,"C6g: 'beta' still present after failed load");
        CHECK(mm.findByImageId("gamma") == nullptr,"C6h: 'gamma' not inserted (load failed)");
        CHECK(mm.findByImageId("delta") == nullptr,"C6i: 'delta' not inserted (load failed)");

        // The original records' internalIds must be unchanged.
        const sir::ImageRecord* pa = mm.findByImageId("alpha");
        CHECK(pa != nullptr && pa->internalId == 0, "C6j: alpha->internalId still 0");
        const sir::ImageRecord* pb = mm.findByImageId("beta");
        CHECK(pb != nullptr && pb->internalId == 1, "C6k: beta->internalId still 1");
    }

    // -----------------------------------------------------------------------
    // C7: successful load() after a failed load() replaces the valid state
    //     (confirms that a prior failure doesn't corrupt future loads)
    // -----------------------------------------------------------------------
    {
        sir::MetadataManager mm;

        // Load 1: valid
        std::vector<sir::ImageRecord> v1;
        v1.push_back(makeRecord(0, "aaa"));
        mm.load(std::move(v1));
        CHECK(mm.size() == 1, "C7a: first valid load -> 1 record");

        // Load 2: invalid (triggers transactional rollback)
        std::vector<sir::ImageRecord> v2;
        v2.push_back(makeRecord(0, "bbb"));
        v2.push_back(makeRecord(9, "ccc")); // mismatch
        mm.load(std::move(v2), /*verbose=*/false);
        CHECK(mm.size() == 1, "C7b: after failed load -> still 1 record");

        // Load 3: valid — replaces state cleanly
        std::vector<sir::ImageRecord> v3;
        v3.push_back(makeRecord(0, "ddd"));
        v3.push_back(makeRecord(1, "eee"));
        v3.push_back(makeRecord(2, "fff"));
        const sir::MetadataLoadResult r3 = mm.load(std::move(v3));
        CHECK(r3.accepted == 3,                   "C7c: third valid load accepted 3");
        CHECK(mm.size() == 3,                     "C7d: size() == 3");
        CHECK(mm.findByImageId("aaa") == nullptr, "C7e: 'aaa' gone");
        CHECK(mm.findByImageId("ddd") != nullptr, "C7f: 'ddd' present");
        CHECK(mm.findByImageId("fff") != nullptr, "C7g: 'fff' present");

        // internalIds must be exactly those from v3 — NOT renumbered.
        const sir::ImageRecord* pd = mm.findByImageId("ddd");
        CHECK(pd != nullptr && pd->internalId == 0, "C7h: ddd->internalId == 0 (original)");
        const sir::ImageRecord* pf = mm.findByImageId("fff");
        CHECK(pf != nullptr && pf->internalId == 2, "C7i: fff->internalId == 2 (original)");
    }

    // -----------------------------------------------------------------------
    // C8: a valid record that follows an invalid candidate in the vector
    //     must NOT be inserted and must NOT be renumbered.
    //     This directly tests that no filter-and-compact occurs.
    // -----------------------------------------------------------------------
    {
        // pos-0 id=0 VALID, pos-1 id=1 VALID, pos-2 id=99 MISMATCH,
        // pos-3 id=3 VALID.  Because pos-2 is bad, the ENTIRE batch is refused.
        std::vector<sir::ImageRecord> recs;
        recs.push_back(makeRecord(0, "x0"));
        recs.push_back(makeRecord(1, "x1"));
        recs.push_back(makeRecord(99,"x2")); // id=99 != pos=2
        recs.push_back(makeRecord(3, "x3"));

        sir::MetadataManager mm;
        sir::MetadataLoadResult res = mm.load(std::move(recs), /*verbose=*/false);

        CHECK(res.rejectedIdMismatch == 1, "C8a: one id-mismatch counted");
        CHECK(res.accepted == 0,           "C8b: accepted == 0 (whole batch refused)");
        CHECK(mm.size() == 0,              "C8c: no records inserted");
        CHECK(mm.findByImageId("x0") == nullptr, "C8d: x0 not inserted");
        CHECK(mm.findByImageId("x1") == nullptr, "C8e: x1 not inserted");
        CHECK(mm.findByImageId("x3") == nullptr, "C8f: x3 not inserted");
    }
}

// ===========================================================================
// D. Integration with Phase 1 (actual Flickr30k via DatasetManager)
// ===========================================================================
static void testIntegrationWithDatasetManager(const std::string& projectRoot) {
    std::cout << "\n--- D. Integration with Phase 1 (Flickr30k) ---\n";

    const std::string imageDir = projectRoot
        + "/data/raw/flickr30k/flickr30k-images";
    const std::string captionJson = projectRoot
        + "/data/raw/flickr30k/dataset_flickr30k.json";

    // Pre-flight: bail gracefully if dataset is absent.
    if (!fs::is_directory(imageDir)) {
        std::cerr << "[SKIP] D: image directory not found: " << imageDir << '\n';
        return;
    }
    if (!fs::is_regular_file(captionJson)) {
        std::cerr << "[SKIP] D: caption JSON not found: " << captionJson << '\n';
        return;
    }

    // Phase 1: ingest
    sir::DatasetManager dm;
    const sir::IngestionReport ingReport =
        dm.load(imageDir, captionJson, /*verbose=*/false);

    CHECK(ingReport.successCount > 0, "D1: DatasetManager loaded > 0 records");

    // Copy the records (DatasetManager retains ownership; we pass a copy to MM)
    std::vector<sir::ImageRecord> copyOfRecords = dm.records();
    const int expectedCount = static_cast<int>(copyOfRecords.size());

    // Phase 2: load into MetadataManager
    sir::MetadataManager mm;
    sir::MetadataLoadResult mmResult =
        mm.load(std::move(copyOfRecords), /*verbose=*/false);

    std::cout << "  DatasetManager successCount  : " << ingReport.successCount  << '\n';
    std::cout << "  MetadataManager accepted     : " << mmResult.accepted       << '\n';
    std::cout << "  MetadataManager rejections   : "
              << (mmResult.rejectedNegativeId + mmResult.rejectedIdMismatch
                  + mmResult.rejectedEmptyImageId + mmResult.rejectedDuplicateImageId)
              << '\n';

    // All records from DatasetManager are already valid, so all must be accepted.
    CHECK(mmResult.accepted == expectedCount,
          "D2: MetadataManager accepted all records from DatasetManager");
    CHECK(mmResult.rejectedNegativeId == 0,       "D3: no negative-id rejections");
    CHECK(mmResult.rejectedIdMismatch == 0,        "D4: no id-mismatch rejections");
    CHECK(mmResult.rejectedEmptyImageId == 0,      "D5: no empty-imageId rejections");
    CHECK(mmResult.rejectedDuplicateImageId == 0,  "D6: no duplicate-imageId rejections");
    CHECK(mm.size() == expectedCount,              "D7: mm.size() == expectedCount");

    // Invariant: records_[i].internalId == i for all i
    bool invariantOk = true;
    for (int i = 0; i < mm.size(); ++i) {
        if (mm.records()[static_cast<std::size_t>(i)].internalId != i) {
            invariantOk = false;
            break;
        }
    }
    CHECK(invariantOk, "D8: invariant records_[i].internalId==i holds for all Flickr30k records");

    // internalIds must be the original ones from DatasetManager — not renumbered.
    // DatasetManager assigns 0-based sequential IDs, so mm.records()[i].internalId
    // must equal i exactly (which the invariant above already covers; this makes
    // the intent explicit for the first and last record).
    const sir::ImageRecord* first = mm.findByInternalId(0);
    CHECK(first != nullptr,           "D9a: findByInternalId(0) not null");
    CHECK(first->internalId == 0,     "D9b: first record's internalId is 0 (not renumbered)");
    CHECK(!first->imageId.empty(),    "D9c: first record has non-empty imageId");
    CHECK(!first->filePath.empty(),   "D9d: first record has non-empty filePath");
    CHECK(!first->captions.empty(),   "D9e: first record has captions");

    const sir::ImageRecord* last = mm.findByInternalId(mm.size() - 1);
    CHECK(last != nullptr,                    "D10a: findByInternalId(last) not null");
    CHECK(last->internalId == mm.size() - 1,  "D10b: last record's internalId not renumbered");
    CHECK(!last->imageId.empty(),             "D10c: last record has non-empty imageId");
    CHECK(!last->captions.empty(),            "D10d: last record has captions");

    // Cross-reference: findByImageId must return the same pointer as findByInternalId
    if (first != nullptr) {
        const sir::ImageRecord* byId = mm.findByImageId(first->imageId);
        CHECK(byId != nullptr, "D11a: findByImageId(first->imageId) not null");
        CHECK(byId == first,   "D11b: findByImageId returns same pointer as findByInternalId");
    }

    if (last != nullptr) {
        const sir::ImageRecord* byId = mm.findByImageId(last->imageId);
        CHECK(byId != nullptr, "D12a: findByImageId(last->imageId) not null");
        CHECK(byId == last,    "D12b: findByImageId returns same pointer as findByInternalId");
    }

    // Unknown imageId and out-of-range internalId
    CHECK(mm.findByImageId("__does_not_exist__") == nullptr,
          "D13: findByImageId(unknown) returns nullptr");
    CHECK(mm.findByInternalId(-1) == nullptr,
          "D14: findByInternalId(-1) returns nullptr");
    CHECK(mm.findByInternalId(mm.size()) == nullptr,
          "D15: findByInternalId(size) returns nullptr");

    // FilePath and captions are preserved
    if (first != nullptr) {
        CHECK(fs::is_regular_file(first->filePath),
              "D16: first record's filePath exists on disk");
        for (const auto& cap : first->captions) {
            CHECK(!cap.empty(), "D17: first caption string is non-empty");
            break;  // check just the first to keep output concise
        }
    }

    // D18: a failed load() after a successful one preserves the full Flickr30k state.
    //      Use a tiny invalid batch (mismatch) to trigger the rollback.
    {
        std::vector<sir::ImageRecord> badBatch;
        badBatch.push_back(makeRecord(0, "fake_img"));
        badBatch.push_back(makeRecord(9, "fake_img2")); // id=9 != pos=1

        sir::MetadataLoadResult rb = mm.load(std::move(badBatch), /*verbose=*/false);

        CHECK(rb.accepted == 0,         "D18a: invalid batch after Flickr30k load accepted 0");
        CHECK(mm.size() == expectedCount,
              "D18b: Flickr30k records preserved after failed load()");
        CHECK(mm.findByImageId("fake_img") == nullptr,
              "D18c: fake_img not inserted");
        // The first real record must still be accessible.
        const sir::ImageRecord* stillFirst = mm.findByInternalId(0);
        CHECK(stillFirst != nullptr,  "D18d: first Flickr30k record still accessible");
        if (first != nullptr && stillFirst != nullptr) {
            CHECK(stillFirst->imageId == first->imageId,
                  "D18e: first record imageId unchanged");
        }
    }
}

// ===========================================================================
// main
// ===========================================================================
int main(int /*argc*/, char** /*argv*/) {
    const char* envRoot = std::getenv("SIR_PROJECT_ROOT");
    const std::string projectRoot =
        envRoot ? std::string(envRoot) : fs::current_path().string();

    std::cout << "=== MetadataManager Tests ===\n";
    std::cout << "Project root: " << projectRoot << '\n';

    testEmptyManager();
    testValidMetadata();
    testInvalidMetadata();
    testIntegrationWithDatasetManager(projectRoot);

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "  Passed : " << gPassed << '\n';
    std::cout << "  Failed : " << gFailed << '\n';

    return gFailed == 0 ? 0 : 1;
}
