#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// rpcmem
void   rpcmem_init(void);
void   rpcmem_deinit(void);
void * rpcmem_alloc(int heap_id, uint32_t flags, int size);
void   rpcmem_free(void * p);
int    rpcmem_to_fd(void * p);

#define RPCMEM_FLAG_UNCACHED 0
#define RPCMEM_FLAG_CACHED   1  // Allocate memory with the same properties as the ION_FLAG_CACHED flag

enum rpc_heap_ids {
    RPCMEM_HEAP_ID_SECURE = 9,
    RPCMEM_HEAP_ID_CONTIG = 22,
    RPCMEM_HEAP_ID_SYSTEM = 25,
};

#ifdef __cplusplus
}
#endif
