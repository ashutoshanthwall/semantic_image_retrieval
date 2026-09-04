cmake_minimum_required(VERSION 3.16)

project(SemanticImageRetrieval
    VERSION 0.1.0
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Global C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
    )
endif()

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
# Project headers: #include "data/DatasetManager.hpp"
include_directories(${CMAKE_SOURCE_DIR}/include)

# Third-party headers: #include <third_party/nlohmann/json.hpp>
# Placed under a separate prefix so third-party includes are visually distinct.
include_directories(${CMAKE_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Data library: Phase 1 (ingestion) + Phase 2 (metadata)
# ---------------------------------------------------------------------------
add_library(sir_data
    src/data/DatasetManager.cpp
    src/data/MetadataManager.cpp
)
target_include_directories(sir_data PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}          # for third_party/
)

# ---------------------------------------------------------------------------
# Phase 1 test executable
# ---------------------------------------------------------------------------
add_executable(test_dataset_manager
    tests/test_dataset_manager.cpp
)
target_link_libraries(test_dataset_manager PRIVATE sir_data)

# ---------------------------------------------------------------------------
# Phase 2 test executable
# ---------------------------------------------------------------------------
add_executable(test_metadata
    tests/test_metadata.cpp
)
target_link_libraries(test_metadata PRIVATE sir_data)

# ---------------------------------------------------------------------------
# CTest integration (run via: ctest --output-on-failure)
# ---------------------------------------------------------------------------
enable_testing()

add_test(
    NAME DatasetManagerIntegration
    COMMAND test_dataset_manager
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(DatasetManagerIntegration PROPERTIES
    ENVIRONMENT "SIR_PROJECT_ROOT=${CMAKE_SOURCE_DIR}"
)

add_test(
    NAME MetadataManagerTests
    COMMAND test_metadata
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(MetadataManagerTests PROPERTIES
    ENVIRONMENT "SIR_PROJECT_ROOT=${CMAKE_SOURCE_DIR}"
)
