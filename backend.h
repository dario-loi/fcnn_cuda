#pragma once

#include "operation.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <random>
#include <string_view>
#include <type_traits>

template <typename T>
  requires std::is_arithmetic_v<T>
struct Tensor;

template <typename T>
  requires std::is_arithmetic_v<T>
class Backend {

public:
  virtual void fill(Tensor<T> &tensor, T value) = 0;
  virtual void rand(Tensor<T> &tensor, T from = static_cast<T>(0),
                    T to = static_cast<T>(1)) = 0;
  virtual void randn(Tensor<T> &tensor, T mean = static_cast<T>(0),
                     T std = static_cast<T>(1)) = 0;

  virtual Tensor<T> add(Tensor<T> const &a, Tensor<T> const &b) = 0;
  virtual Tensor<T> mul(Tensor<T> const &a, Tensor<T> const &b) = 0;

  virtual const std::string_view get_name() const = 0;
};

template <typename T>
  requires std::is_arithmetic_v<T>
class CpuBackend : public Backend<T> {

  const std::string_view name{"cpu"};

public:
  void fill(Tensor<T> &tensor, T value) override {
    std::fill(tensor.data.get(), tensor.data.get() + tensor.size, value);
  }

  void rand(Tensor<T> &tensor, T from = static_cast<T>(0),
            T to = static_cast<T>(1)) override {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(from, to);
    std::generate(tensor.data.get(), tensor.data.get() + tensor.size,
                  [&]() { return dis(gen); });
  }

  void randn(Tensor<T> &tensor, T mean = static_cast<T>(0),
             T std = static_cast<T>(1)) override {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, std);
    std::generate(tensor.data.get(), tensor.data.get() + tensor.size,
                  [&]() { return dis(gen); });
  }

  Tensor<T> add(Tensor<T> const &a, Tensor<T> const &b) override {
    Tensor<T> result(a.shape, std::make_shared<AddOperation<T>>(a, b));
#pragma omp simd
    for (size_t i = 0; i < a.size; ++i) {
      result.data[i] = a.data[i] + b.data[i];
    }
    return result;
  }

  Tensor<T> mul(Tensor<T> const &a, Tensor<T> const &b) {
    Tensor<T> result(a.shape, std::make_shared<MulOperation<T>>(a, b));
#pragma omp simd
    for (size_t i = 0; i < a.size; ++i) {
      result.data[i] = a.data[i] * b.data[i];
    }
    return result;
  }

  const std::string_view get_name() const { return name; }
};