#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"


void nvbit_at_init() {
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
    const char* name, void* params, CUresult* pStatus) {
}

void nvbit_at_ctx_init(CUcontext ctx) {
}

void nvbit_at_ctx_term(CUcontext ctx) {
}
