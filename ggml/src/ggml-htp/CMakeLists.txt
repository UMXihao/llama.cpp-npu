message(STATUS "Using My Hexagon backend")

set(GGML_SOURCES_HTP ggml-htp.cc)

ggml_add_backend_library(ggml-htp ${GGML_SOURCES_HTP})

# TODO: target_link_libraries(ggml-htp PRIVATE ${HTP_LIBRARIES})
