#include "ggml-htp.h"

#include <stdlib.h>
#include <string.h>

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

struct ggml_backend_htp_context {
    // TODO
};

// HTP backend buffer type (shared rpcmem)

// TODO(hzx): real impl

static void * ggml_backend_htp_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void ggml_backend_htp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void ggml_backend_htp_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_htp_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_htp_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_htp_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_htp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_htp_buffer_i = {
    /* .free_buffer     = */ ggml_backend_htp_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_htp_buffer_get_base,
    /* .init_tensor     = */ nullptr,  // no initialization required?
    /* .memset_tensor   = */ ggml_backend_htp_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_htp_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_htp_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_htp_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_htp_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * ggml_backend_htp_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "RPCMEM";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_htp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // TODO(hzx): allocate real rpcmem, use posix_memalign for emulation now

    void * data = nullptr;
    posix_memalign(&data, 128, size);
    GGML_ASSERT(data);

    printf("RPCMEM alloc size = %.5f MiB\n", size / 1024.0 / 1024.0);

    return ggml_backend_buffer_init(buft, ggml_backend_htp_buffer_i, data, size);
}

static size_t ggml_backend_htp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static bool ggml_backend_htp_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_htp_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    // TODO(hzx): change max size
    return 128 * size_t(1024 * 1024);

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_htp_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_htp_buffer_type = {
        /* .iface   = */ {
                          /* .get_name         = */ ggml_backend_htp_buffer_type_get_name,
                          /* .alloc_buffer     = */ ggml_backend_htp_buffer_type_alloc_buffer,
                          /* .get_alignment    = */ ggml_backend_htp_buffer_type_get_alignment,
                          /* .get_max_size     = */ ggml_backend_htp_buffer_type_get_max_size,
                          /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_htp_buffer_type_is_host,
                          },
        /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_htp_reg(), 0),
        /* .context = */ nullptr,
    };

    return &ggml_backend_htp_buffer_type;
}

// backend interface

static const char * ggml_backend_htp_get_name(ggml_backend_t backend) {
    return "MyHTP";

    GGML_UNUSED(backend);
}

static void ggml_backend_htp_free(ggml_backend_t backend) {
    ggml_backend_htp_context * ctx = (ggml_backend_htp_context *) backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_htp_mul_mat(ggml_backend_htp_context * ctx, struct ggml_tensor * dst) {
    // TODO
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static enum ggml_status ggml_backend_htp_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    static ggml_backend_t my_cpu_backend = nullptr;
    if (!my_cpu_backend) {
        my_cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        GGML_ASSERT(my_cpu_backend);
    }

    return ggml_backend_graph_compute(my_cpu_backend, cgraph);

    GGML_UNUSED(backend);
}

static struct ggml_backend_i htp_backend_i = {
    /* .get_name                = */ ggml_backend_htp_get_name,
    /* .free                    = */ ggml_backend_htp_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_htp_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
};

static ggml_guid_t ggml_backend_htp_guid(void) {
    static ggml_guid guid = { 0x6b, 0x22, 0x31, 0xb2, 0xfb, 0x66, 0x46, 0xf6,
                              0x87, 0x3b, 0x7d, 0x8a, 0x13, 0xe5, 0x78, 0x13 };
    return &guid;
}

static ggml_backend_t ggml_backend_htp_init(void) {
    ggml_backend_htp_context * ctx = new ggml_backend_htp_context;

    ggml_backend_t backend = new ggml_backend{
        /* .guid      = */ ggml_backend_htp_guid(),
        /* .interface = */ htp_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_htp_reg(), 0),
        /* .context   = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_htp(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_htp_guid());
}

static const char * ggml_backend_htp_device_get_name(ggml_backend_dev_t dev) {
    return "HTP";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_htp_device_get_description(ggml_backend_dev_t dev) {
    return "Unknown Hexagon Processor";

    GGML_UNUSED(dev);
}

static void ggml_backend_htp_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_htp_device_get_type(ggml_backend_dev_t dev) {
    // TODO(hzx): use GGML_BACKEND_DEVICE_TYPE_GPU or GGML_BACKEND_DEVICE_TYPE_ACCEL?
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_htp_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_htp_device_get_name(dev);
    props->description = ggml_backend_htp_device_get_description(dev);
    props->type        = ggml_backend_htp_device_get_type(dev);
    ggml_backend_htp_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_htp_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_htp_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_htp_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_htp_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_htp_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto * cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    return ggml_backend_dev_supports_op(cpu_dev, op);

    GGML_UNUSED(dev);
}

static bool ggml_backend_htp_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_htp_buffer_type_get_name;

    GGML_UNUSED(dev);
}

static bool ggml_backend_htp_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto * cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    return ggml_backend_dev_supports_op(cpu_dev, op);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_htp_device_i = {
    /* .get_name             = */ ggml_backend_htp_device_get_name,
    /* .get_description      = */ ggml_backend_htp_device_get_description,
    /* .get_memory           = */ ggml_backend_htp_device_get_memory,
    /* .get_type             = */ ggml_backend_htp_device_get_type,
    /* .get_props            = */ ggml_backend_htp_device_get_props,
    /* .init_backend         = */ ggml_backend_htp_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_htp_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_htp_device_supports_op,
    /* .supports_buft        = */ ggml_backend_htp_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// backend reg interface

static const char * ggml_backend_htp_reg_get_name(ggml_backend_reg_t reg) {
    return "MyHTP";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_htp_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_htp_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    // TODO

    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_htp_device = {
        /* .iface   = */ ggml_backend_htp_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &ggml_backend_htp_device;
}

static const struct ggml_backend_reg_i ggml_backend_htp_reg_i = {
    /* .get_name         = */ ggml_backend_htp_reg_get_name,
    /* .get_device_count = */ ggml_backend_htp_reg_get_device_count,
    /* .get_device       = */ ggml_backend_htp_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_htp_reg(void) {
    static struct ggml_backend_reg ggml_backend_htp_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_htp_reg_i,
        /* .context     = */ nullptr,
    };
    return &ggml_backend_htp_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_htp_reg)
