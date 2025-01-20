#include "rpcmem_mapper.h"

#include <vector>

#include "dsprpc_interface.h"
#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml-impl.h"

void RpcMemMapper::validate(const ggml_tensor * dst) {
    std::vector<ggml_backend_buffer *> buffers;

    auto add_buffer = [&](ggml_backend_buffer * buf) {
        if (ggml_backend_buft_is_rpcmem(buf->buft)) {
            buffers.push_back(buf);
        }
    };

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src) {
            add_buffer(src->buffer);
        }
    }
    add_buffer(dst->buffer);

    size_t required_size = 0;
    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        if (buf_mapping.count(buf_base)) {
            // buffer already mapped, we put it at the front of the LRU access list
            accessed_bufs.erase(buf_iters.at(buf_base));
            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base] = accessed_bufs.begin();
        } else {
            required_size += buf->size;
        }
    }

    GGML_ASSERT(required_size <= max_active_map_size);
    while (active_map_size + required_size > max_active_map_size) {
        // remove least recent used mapping
        void * buf_base     = accessed_bufs.back();
        auto [fd, buf_size] = buf_mapping.at(buf_base);

        // fprintf(stderr, "rpcmem_mapper: removing memory mapping for rpcmem buffer %p, size %.2f MiB, fd %d\n", buf_base,
        //          buf_size / 1048576.0, fd);
        int err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf_base, buf_size);
        if (err) {
            fprintf(stderr, "fastrpc_munmap failed with return code: %x\n", err);
        }

        accessed_bufs.pop_back();
        buf_iters.erase(buf_base);
        buf_mapping.erase(buf_base);
        active_map_size -= buf_size;
    }

    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        size_t buf_size = buf->size;
        if (!buf_mapping.count(buf_base)) {
            int fd = rpcmem_to_fd(buf_base);
            if (fd < 0) {
                GGML_ABORT("rpcmem_to_fd returns %d\n", fd);
            }

            int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf_base, 0, buf_size, FASTRPC_MAP_FD);
            if (err) {
                GGML_ABORT("fastrpc_mmap failed with return code: %x\n", err);
            }
            // fprintf(stderr, "rpcmem_mapper: creating memory mapping for rpcmem buffer %p, size %.2f MiB, fd %d\n", buf_base,
            //          buf_size / 1048576.0, fd);

            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base]   = accessed_bufs.begin();
            buf_mapping[buf_base] = { fd, buf_size };
            active_map_size += buf_size;
        }
    }
}

std::pair<int, ssize_t> RpcMemMapper::get_tensor_mapping(const ggml_tensor * tensor) const {
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(tensor->buffer->buft));

    void * buf_base = ggml_backend_buffer_get_base(tensor->buffer);
    auto [fd, _]    = buf_mapping.at(buf_base);
    auto offset     = (intptr_t) tensor->data - (intptr_t) buf_base;
    return { fd, offset };
}

extern "C" {

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst) {
    ggml_backend_htp_context::instance()->mapper.validate(dst);
    return 0;
}
}
