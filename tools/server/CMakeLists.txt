include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(RAG)

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-context-support.cpp
    server-context-support.h
    server-memory-slice-contract.cpp
    server-memory-slice-contract.h
    server-remote-session.cpp
    server-remote-session.h
    server-tools.cpp
    server-tools.h
    server-goal.cpp
    server-goal.h
    server-supervision.cpp
    server-supervision.h
    server-tool-envelope.cpp
    server-tool-envelope.h
    server-audit-event.cpp
    server-audit-event.h
    server-clips-goal-router.cpp
    server-clips-goal-router.h
    server-goal-executor.cpp
    server-goal-executor.h

        # PR-1: JSON utils + remote-session helpers
    server-json-utils.cpp
    server-json-utils.h
    server-trace-registry.cpp
    server-trace-registry.h
    server-remote-session-turn.cpp
    server-remote-session-turn.h

    # PR-2: RAG routes
    server-response-generator.h
    server-rag-routes.cpp
    server-rag-routes.h

    # PR-3: embeddings / rerank routes
    server-embedding-routes.cpp
    server-embedding-routes.h

    # PR-4: slot save / restore / erase routes
    server-slot-action-routes.cpp
    server-slot-action-routes.h

    # PR-5: server_slot
    server-slot.cpp
    server-slot.h

    # PR-6: server_metrics
    server-metrics.cpp
    server-metrics.h

    # PR-7: LoRA manager + routes
    server-lora-manager.cpp
    server-lora-manager.h
    server-lora-routes.cpp
    server-lora-routes.h

    # PR-8: token response processor
    server-token-processor.cpp
    server-token-processor.h

    # PR-9: task dispatcher
    server-task-dispatcher.cpp
    server-task-dispatcher.h

    # PR-10: update_slots helper
    server-slot-updater.cpp
    server-slot-updater.h

    # PR-11: speculative decoder
    server-speculative-decoder.cpp
    server-speculative-decoder.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/RAG/src)
target_link_libraries(${TARGET} PUBLIC llama-common mtmd server-rag-core ${CMAKE_THREAD_LIBS_INIT})


# llama-server executable

set(TARGET llama-server)

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)

option(LLAMA_BUILD_WEBUI "Build the embedded Web UI" ON)

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        codex-mcp-runtime.js
        bundle.js
        bundle.css
        loading.html
        bundle.parts/manifest.json
        bundle.parts/bundle.part.000.js
        bundle.parts/bundle.part.001.js
        bundle.parts/bundle.part.002.js
        bundle.parts/bundle.part.003.js
        bundle.parts/bundle.part.004.js
        bundle.parts/bundle.part.005.js
        bundle.parts/bundle.part.006.js
        bundle.parts/bundle.part.007.js
        bundle.parts/bundle.part.008.js
        bundle.parts/bundle.part.009.js
        bundle.parts/bundle.part.010.js
        bundle.parts/bundle.part.011.js
        bundle.parts/bundle.part.012.js
        bundle.parts/bundle.part.013.js
    )

    foreach(asset ${PUBLIC_ASSETS})
        set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
        get_filename_component(output_dir "${output}" DIRECTORY)
        file(MAKE_DIRECTORY "${output_dir}")
        list(APPEND TARGET_SRCS ${output})
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
        )
        set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
    endforeach()
    add_definitions(-DLLAMA_BUILD_WEBUI)
else()
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)
