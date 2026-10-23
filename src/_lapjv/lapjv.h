// Copyright (c) 2026 Ratha SIV | MIT License
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int laprust_run(const void*, int, const void*, int, size_t, size_t,
                size_t, double, size_t, int, int, int, int64_t*, size_t*,
                double*, size_t*);
#ifdef __cplusplus
}
#endif
