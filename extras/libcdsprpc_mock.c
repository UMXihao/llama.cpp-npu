// mock libcsprpc.so content
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// rpcmem
void   rpcmem_init(void);
void   rpcmem_deinit(void);
void * rpcmem_alloc(int heap_id, uint32_t flags, int size);  // NOTE: `size` is indeed an int here (< 2GB)
// NOTE(hzx): we don't mock rpcmem_alloc2 since it is not implemented in actual libcdsprpc.so (at least this is true on my phone)
void   rpcmem_free(void * p);
int    rpcmem_to_fd(void * p);

// rpcmem impl
void rpcmem_init(void) {}

void rpcmem_deinit(void) {}

// returns pointer to the buffer on success; NULL on failure
void * rpcmem_alloc(int heap_id, uint32_t flags, int size) {
    void * p = NULL;
    posix_memalign(&p, 128, size);
    return p;

    (void) heap_id;
    (void) flags;
}

// free a buffer and ignore invalid buffers
void rpcmem_free(void * p) {
    free(p);
}

// returns an associated file descriptor
int rpcmem_to_fd(void * p) {
    return -1;

    (void) p;
}
