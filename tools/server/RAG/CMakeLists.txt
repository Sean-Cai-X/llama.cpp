set(LLAMA_SERVER_RAG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(LLAMA_SERVER_RAG_SRC_ROOT ${LLAMA_SERVER_RAG_ROOT}/src)

option(LLAMA_SERVER_RAG_ENABLE "Build the isolated RAG module under tools/server/RAG" ON)
option(LLAMA_SERVER_RAG_ENABLE_CLIPS "Build the local CLIPS core bridge for the isolated RAG module" OFF)
option(LLAMA_SERVER_RAG_ENABLE_CXPARSER_BRIDGE "Build the cxparser-facing explain bridge inside tools/server/RAG" OFF)
option(LLAMA_SERVER_RAG_ENABLE_ROCKSDB "Mirror RAG storage write batches into a RocksDB backend" OFF)

set(LLAMA_SERVER_RAG_CXPARSER_TYPES_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../cxparser_ext/pipeline/parser_task_types.h")
set(LLAMA_SERVER_RAG_CXPARSER_ENTRY_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../parser_unified_entry.h")

if (NOT LLAMA_SERVER_RAG_ENABLE)
    return()
endif()

set(LLAMA_SERVER_RAG_FAISS_ROOT "" CACHE PATH "Root directory of the FAISS source tree for tools/server/RAG")
set(LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR "" CACHE PATH "Directory containing FAISS headers for tools/server/RAG")
set(LLAMA_SERVER_RAG_FAISS_LIBRARY "" CACHE FILEPATH "Path to the FAISS library for tools/server/RAG")
set(LLAMA_SERVER_RAG_OPENBLAS_LIBRARY "" CACHE FILEPATH "Path to the OpenBLAS library used by FAISS for tools/server/RAG")
set(LLAMA_SERVER_RAG_ROCKSDB_ROOT "" CACHE PATH "Root directory of the RocksDB source or install tree for tools/server/RAG")
set(LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR "" CACHE PATH "Directory containing RocksDB headers for tools/server/RAG")
set(LLAMA_SERVER_RAG_ROCKSDB_LIBRARY "" CACHE FILEPATH "Path to the RocksDB library for tools/server/RAG")

if (LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR AND
    NOT EXISTS "${LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR}/faiss/IndexFlat.h")
    unset(LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR CACHE)
endif()

if (LLAMA_SERVER_RAG_FAISS_ROOT AND
    NOT EXISTS "${LLAMA_SERVER_RAG_FAISS_ROOT}/faiss/IndexFlat.h")
    unset(LLAMA_SERVER_RAG_FAISS_ROOT CACHE)
endif()

set(_rag_faiss_root_candidates
    "${CMAKE_SOURCE_DIR}/analysis_workspace/faiss-main"
    "${CMAKE_SOURCE_DIR}/../analysis_workspace/faiss-main"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../analysis_workspace/faiss-main"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../analysis_workspace/faiss-main"
)

if (NOT LLAMA_SERVER_RAG_FAISS_ROOT)
    foreach(_candidate IN LISTS _rag_faiss_root_candidates)
        if (EXISTS "${_candidate}/faiss/IndexFlat.h")
            get_filename_component(_candidate_abs "${_candidate}" ABSOLUTE)
            set(LLAMA_SERVER_RAG_FAISS_ROOT "${_candidate_abs}" CACHE PATH "Root directory of the FAISS source tree for tools/server/RAG" FORCE)
            break()
        endif()
    endforeach()
endif()

if (NOT LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR AND LLAMA_SERVER_RAG_FAISS_ROOT)
    set(_rag_faiss_include_candidates
        "${LLAMA_SERVER_RAG_FAISS_ROOT}"
        "${LLAMA_SERVER_RAG_FAISS_ROOT}/include"
    )
    foreach(_candidate IN LISTS _rag_faiss_include_candidates)
        if (EXISTS "${_candidate}/faiss/IndexFlat.h")
            set(LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR "${_candidate}" CACHE PATH "Directory containing FAISS headers for tools/server/RAG" FORCE)
            break()
        endif()
    endforeach()
endif()

if (NOT LLAMA_SERVER_RAG_FAISS_LIBRARY AND LLAMA_SERVER_RAG_FAISS_ROOT)
    find_library(LLAMA_SERVER_RAG_FAISS_LIBRARY
        NAMES faiss
        HINTS
            "${LLAMA_SERVER_RAG_FAISS_ROOT}/build-openblas-check2/faiss/Release"
            "${LLAMA_SERVER_RAG_FAISS_ROOT}/build-openblas-check2/faiss"
            "${LLAMA_SERVER_RAG_FAISS_ROOT}/build-mkl-check2/faiss/Release"
            "${LLAMA_SERVER_RAG_FAISS_ROOT}/build-mkl-check2/faiss"
            "${LLAMA_SERVER_RAG_FAISS_ROOT}/build/faiss/Release"
            "${LLAMA_SERVER_RAG_FAISS_ROOT}/build/faiss"
    )
endif()

if (NOT LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR)
    find_path(LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR
        NAMES faiss/IndexFlat.h
        HINTS ${LLAMA_SERVER_RAG_FAISS_ROOT}
    )
endif()

if (NOT LLAMA_SERVER_RAG_FAISS_LIBRARY)
    find_library(LLAMA_SERVER_RAG_FAISS_LIBRARY
        NAMES faiss
    )
endif()

if (NOT LLAMA_SERVER_RAG_OPENBLAS_LIBRARY)
    find_library(LLAMA_SERVER_RAG_OPENBLAS_LIBRARY
        NAMES openblas libopenblas
        HINTS
            "D:/OpenBLAS/lib"
    )
endif()

if (LLAMA_SERVER_RAG_ENABLE_ROCKSDB)
    if (LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR AND
        NOT EXISTS "${LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR}/rocksdb/db.h")
        unset(LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR CACHE)
    endif()

    if (LLAMA_SERVER_RAG_ROCKSDB_ROOT AND
        NOT EXISTS "${LLAMA_SERVER_RAG_ROCKSDB_ROOT}/include/rocksdb/db.h" AND
        NOT EXISTS "${LLAMA_SERVER_RAG_ROCKSDB_ROOT}/rocksdb/db.h")
        unset(LLAMA_SERVER_RAG_ROCKSDB_ROOT CACHE)
    endif()

    set(_rag_rocksdb_root_candidates
        "${CMAKE_SOURCE_DIR}/analysis_workspace/rocksdb-11.0.4"
        "${CMAKE_SOURCE_DIR}/../analysis_workspace/rocksdb-11.0.4"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../analysis_workspace/rocksdb-11.0.4"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../analysis_workspace/rocksdb-11.0.4"
    )

    if (NOT LLAMA_SERVER_RAG_ROCKSDB_ROOT)
        foreach(_candidate IN LISTS _rag_rocksdb_root_candidates)
            if (EXISTS "${_candidate}/include/rocksdb/db.h")
                get_filename_component(_candidate_abs "${_candidate}" ABSOLUTE)
                set(LLAMA_SERVER_RAG_ROCKSDB_ROOT "${_candidate_abs}" CACHE PATH "Root directory of the RocksDB source or install tree for tools/server/RAG" FORCE)
                break()
            endif()
        endforeach()
    endif()

    if (NOT LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR AND LLAMA_SERVER_RAG_ROCKSDB_ROOT)
        set(_rag_rocksdb_include_candidates
            "${LLAMA_SERVER_RAG_ROCKSDB_ROOT}/include"
            "${LLAMA_SERVER_RAG_ROCKSDB_ROOT}"
        )
        foreach(_candidate IN LISTS _rag_rocksdb_include_candidates)
            if (EXISTS "${_candidate}/rocksdb/db.h")
                set(LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR "${_candidate}" CACHE PATH "Directory containing RocksDB headers for tools/server/RAG" FORCE)
                break()
            endif()
        endforeach()
    endif()

    if (NOT LLAMA_SERVER_RAG_ROCKSDB_LIBRARY AND LLAMA_SERVER_RAG_ROCKSDB_ROOT)
        find_library(LLAMA_SERVER_RAG_ROCKSDB_LIBRARY
            NAMES rocksdb librocksdb rocksdb_static
            HINTS
                "${LLAMA_SERVER_RAG_ROCKSDB_ROOT}/build/Release"
                "${LLAMA_SERVER_RAG_ROCKSDB_ROOT}/build"
                "${LLAMA_SERVER_RAG_ROCKSDB_ROOT}/lib"
                "${CMAKE_SOURCE_DIR}/rocksdb-build/Release"
                "${CMAKE_SOURCE_DIR}/../rocksdb-build/Release"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../../rocksdb-build/Release"
        )
    endif()

    if (NOT LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR)
        find_path(LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR
            NAMES rocksdb/db.h
            HINTS ${LLAMA_SERVER_RAG_ROCKSDB_ROOT}
        )
    endif()

    if (NOT LLAMA_SERVER_RAG_ROCKSDB_LIBRARY)
        find_library(LLAMA_SERVER_RAG_ROCKSDB_LIBRARY
            NAMES rocksdb librocksdb rocksdb_static
            HINTS
                "${CMAKE_SOURCE_DIR}/rocksdb-build/Release"
                "${CMAKE_SOURCE_DIR}/../rocksdb-build/Release"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../../rocksdb-build/Release"
        )
    endif()

    if (NOT LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR OR NOT LLAMA_SERVER_RAG_ROCKSDB_LIBRARY)
        message(FATAL_ERROR "tools/server/RAG RocksDB backend requested. Set LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR and LLAMA_SERVER_RAG_ROCKSDB_LIBRARY.")
    endif()
endif()

if (NOT LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR OR NOT LLAMA_SERVER_RAG_FAISS_LIBRARY)
    message(FATAL_ERROR "tools/server/RAG requires FAISS. Set LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR and LLAMA_SERVER_RAG_FAISS_LIBRARY.")
endif()

if (NOT LLAMA_SERVER_RAG_OPENBLAS_LIBRARY)
    message(FATAL_ERROR "tools/server/RAG requires OpenBLAS. Set LLAMA_SERVER_RAG_OPENBLAS_LIBRARY.")
endif()

set(LLAMA_SERVER_RAG_CORE_SRCS
    ${LLAMA_SERVER_RAG_SRC_ROOT}/BM25Index.cpp
    ${LLAMA_SERVER_RAG_SRC_ROOT}/rag.cpp
    ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_metadata.cpp
    ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_storage.cpp
    ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_clips_meta.cpp
    ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_clips_runner.cpp
    ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_integration_bridge.cpp
    ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_server_runtime.cpp
    ${LLAMA_SERVER_RAG_SRC_ROOT}/repo_scanner.cpp
)

if (LLAMA_SERVER_RAG_ENABLE_CXPARSER_BRIDGE AND
    EXISTS "${LLAMA_SERVER_RAG_CXPARSER_TYPES_HEADER}" AND
    EXISTS "${LLAMA_SERVER_RAG_CXPARSER_ENTRY_HEADER}")
    list(APPEND LLAMA_SERVER_RAG_CORE_SRCS
        ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_explain.cpp
        ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_cxparser_bridge.cpp
        ${LLAMA_SERVER_RAG_SRC_ROOT}/rag_chat_minimal.cpp
    )
elseif (LLAMA_SERVER_RAG_ENABLE_CXPARSER_BRIDGE)
    message(WARNING "tools/server/RAG cxparser bridge requested but parser headers are missing; bridge sources will be skipped.")
endif()

add_library(server-rag-core STATIC
    ${LLAMA_SERVER_RAG_CORE_SRCS}
)

target_include_directories(server-rag-core PUBLIC
    ${LLAMA_SERVER_RAG_SRC_ROOT}
    ${LLAMA_SERVER_RAG_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_SOURCE_DIR}
    ${LLAMA_SERVER_RAG_FAISS_INCLUDE_DIR}
)

target_link_libraries(server-rag-core PUBLIC
    llama-common
    mtmd
    ${LLAMA_SERVER_RAG_FAISS_LIBRARY}
    ${LLAMA_SERVER_RAG_OPENBLAS_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_definitions(server-rag-core PUBLIC
    LLAMA_SERVER_RAG_MODULE=1
)

if (LLAMA_SERVER_RAG_ENABLE_ROCKSDB)
    target_compile_features(server-rag-core PUBLIC cxx_std_20)
    target_include_directories(server-rag-core PUBLIC
        ${LLAMA_SERVER_RAG_ROCKSDB_INCLUDE_DIR}
    )
    target_link_libraries(server-rag-core PUBLIC
        ${LLAMA_SERVER_RAG_ROCKSDB_LIBRARY}
    )
    if (WIN32)
        target_link_libraries(server-rag-core PUBLIC
            shlwapi
            rpcrt4
        )
    endif()
    target_compile_definitions(server-rag-core PUBLIC
        LLAMA_SERVER_RAG_ROCKSDB_BACKEND=1
    )
endif()

if (LLAMA_SERVER_RAG_ENABLE_CXPARSER_BRIDGE AND
    EXISTS "${LLAMA_SERVER_RAG_CXPARSER_TYPES_HEADER}" AND
    EXISTS "${LLAMA_SERVER_RAG_CXPARSER_ENTRY_HEADER}")
    target_include_directories(server-rag-core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    )
    target_compile_definitions(server-rag-core PUBLIC
        LLAMA_SERVER_RAG_CXPARSER_BRIDGE=1
    )
endif()

if (EXISTS "${LLAMA_SERVER_RAG_ROOT}/clips_core/clips.h")
    target_compile_definitions(server-rag-core PUBLIC
        LLAMA_SERVER_RAG_CLIPS_SOURCE_PRESENT=1
    )
    target_include_directories(server-rag-core PUBLIC
        ${LLAMA_SERVER_RAG_ROOT}/clips_core
    )
endif()

if (LLAMA_SERVER_RAG_ENABLE_CLIPS)
    file(GLOB LLAMA_SERVER_RAG_CLIPS_SRCS
        "${LLAMA_SERVER_RAG_ROOT}/clips_core/*.c"
    )

    list(FILTER LLAMA_SERVER_RAG_CLIPS_SRCS EXCLUDE REGEX ".*/main\\.c$")

    add_library(server-rag-clips-core STATIC
        ${LLAMA_SERVER_RAG_CLIPS_SRCS}
    )

    target_include_directories(server-rag-clips-core PUBLIC
        ${LLAMA_SERVER_RAG_ROOT}/clips_core
    )

    target_compile_definitions(server-rag-clips-core PUBLIC
        LLAMA_SERVER_RAG_CLIPS_CORE_LINKED=1
    )

    target_link_libraries(server-rag-core PUBLIC
        server-rag-clips-core
    )
endif()
