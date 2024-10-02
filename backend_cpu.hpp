#pragma once
#include "tensor.hpp"

#include <algorithm>
#include <random>

// Wrapper struct that contains methods for CPU backend operations
template <typename T>
struct CpuBackend {

    using TensorType = Tensor<CpuBackend, T>;
    using ValueType = T;

    static void fill(TensorType& tensor, T value)
    {
        std::fill(tensor.data.get(), tensor.data.get() + tensor.size, value);
    }

    static void rand(TensorType& tensor)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0, 1);
        std::generate(tensor.data.get(), tensor.data.get() + tensor.size, [&]() { return dis(gen); });
    }
};