#pragma once

#include <string>
#include <vector>

namespace sir {

/**
 * Represents a single image entry in the dataset.
 *
 * internalId : sequential integer assigned by DatasetManager (0-based).
 *              Used as the primary key for vector-search indexes, which
 *              operate on integers rather than string paths.
 *
 * imageId    : the Flickr image identifier derived from the filename
 *              (e.g. "1000092795" for "1000092795.jpg").
 *              Kept as a string because Flickr IDs can be large and are
 *              treated as opaque identifiers, not arithmetic values.
 *
 * filePath   : absolute or project-relative path to the image file on disk.
 *
 * width / height : pixel dimensions populated when image headers are readable;
 *                  set to 0 when dimensions cannot be determined without
 *                  full OpenCV decoding (Phase 1 does lightweight validation
 *                  only — full decode belongs to the preprocessing phase).
 *
 * captions   : one or more natural-language descriptions sourced from
 *              dataset_flickr30k.json.  Typically 5 sentences per image.
 */
struct ImageRecord {
    int                      internalId{-1};
    std::string              imageId;
    std::string              filePath;
    int                      width{0};
    int                      height{0};
    std::vector<std::string> captions;
};

}  // namespace sir
