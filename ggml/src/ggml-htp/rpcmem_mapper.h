#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#    include <list>
#    include <unordered_map>

struct RpcMemMapper {
    RpcMemMapper() : max_active_map_size{ (size_t) -1 }, active_map_size{ 0 } {}

    RpcMemMapper(size_t max_size) : max_active_map_size{ max_size }, active_map_size{ 0 } {}

    void                    validate(const struct ggml_tensor * dst);
    std::pair<int, ssize_t> get_tensor_mapping(const struct ggml_tensor *) const;  // returns <mapping fd, offset>

  private:
    size_t max_active_map_size;
    size_t active_map_size;

    std::unordered_map<void *, std::pair<int, size_t>>      buf_mapping;
    std::unordered_map<void *, std::list<void *>::iterator> buf_iters;      // for LRU replacement
    std::list<void *>                                       accessed_bufs;  // for LRU replacement
};
#endif
