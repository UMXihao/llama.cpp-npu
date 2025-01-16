// This is a C++ header
#pragma once

#include "ggml.h"
#include "rpcmem_mapper.h"

// singleton HTP backend context
struct ggml_backend_htp_context {
    // stuff in struct ggml_backend_cpu_context, see ggml-cpu.cpp
    size_t    work_size = 0;
    uint8_t * work_data = nullptr;

    int                      n_threads  = 0;
    struct ggml_threadpool * threadpool = nullptr;

    // TODO(hzx): add abort_callback & abort_callback_data

    // shared rpcmem mapper
    RpcMemMapper mapper;

    ggml_backend_htp_context();
    ~ggml_backend_htp_context();

    static ggml_backend_htp_context * instance();
};

extern "C" {

enum ggml_status ggml_graph_compute_htp_hybrid(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

}
