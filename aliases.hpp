#pragma once

#include "backend_cpu.hpp"
#include "tensor.hpp"
#include <cstddef>

using CPUTensorF32 = Tensor<CpuBackend<float>, float>;
using CPUTensorF64 = Tensor<CpuBackend<double>, double>;
using CPUTensorI32 = Tensor<CpuBackend<int32_t>, int32_t>;
using CPUTensorI64 = Tensor<CpuBackend<int64_t>, int64_t>;
