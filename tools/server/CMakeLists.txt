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
    server-remote-session.cpp
    server-remote-session.h
    server-tools.cpp
    server-tools.h
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
