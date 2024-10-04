#pragma once

#include "backend.h"
#include "operation.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>
#include <cstring>

enum class Device { CPU, GPU };

template <typename T>
  requires std::is_arithmetic_v<T>
struct Tensor {

  std::shared_ptr<T[]> data;
  std::shared_ptr<T[]> grad;
  std::vector<size_t> shape;
  size_t size;
  uint32_t ndim;
  bool requires_grad = true;
  std::shared_ptr<Operation<T>> creator_op;
  std::shared_ptr<Backend<T>> backend;

  Tensor<T>(std::vector<size_t> const &m_shape, bool requires_grad = true)
      : shape(m_shape), size(std::accumulate(shape.begin(), shape.end(), 1,
                                             std::multiplies<size_t>())),
        ndim(shape.size()), requires_grad(requires_grad),
        creator_op(std::make_shared<CreateOperation<T>>()) {
    data = std::make_shared<T[]>(size);
    if (requires_grad) {
      grad = std::make_shared_for_overwrite<T[]>(size);
    } else {
      grad = nullptr;
    }
    to_device(Device::CPU);
  }

  Tensor<T>(std::vector<size_t> const &m_shape,
            std::shared_ptr<Operation<T>> op, bool requires_grad = true)
      : shape(m_shape), size(std::accumulate(shape.begin(), shape.end(), 1,
                                             std::multiplies<size_t>())),
        ndim(shape.size()), requires_grad(requires_grad), creator_op(op) {
    data = std::make_shared<T[]>(size);
    if (requires_grad) {
      grad = std::make_shared_for_overwrite<T[]>(size);
    } else {
      grad = nullptr;
    }
    to_device(Device::CPU);
  }

  template <typename... Indices> T &operator()(Indices... indices) {
    size_t index = 0;
    size_t indices_array[] = {static_cast<size_t>(indices)...};
    for (uint32_t i = 0; i < ndim; ++i) {
      index +=
          indices_array[i] * std::accumulate(shape.begin() + i + 1, shape.end(),
                                             1, std::multiplies<size_t>());
    }
    return data[index];
  }

  constexpr std::span<T> as_span() const {
    return std::span<T>(data.get(), size);
  }

  void pretty_print() { print_recursive(std::span(shape)); }
  void print() {
    for (size_t i = 0; i < size - 1; ++i) {
      std::cout << data[i] << ", ";
    }
    std::cout << data[size - 1] << std::endl;
  }

  void print_grad() {
    assert(
        requires_grad &&
        "Tensor with requires_grad = false cannot have its gradient printed");
    for (size_t i = 0; i < size - 1; ++i) {
      std::cout << grad[i] << ", ";
    }
    std::cout << grad[size - 1] << std::endl;
  }

  void fill(T value) { backend->fill(*this, value); }
  void rand(T from = static_cast<T>(0), T to = static_cast<T>(1)) {
    backend->rand(*this, from, to);
  }
  void randn(T mean = static_cast<T>(0), T std = static_cast<T>(1)) {
    backend->randn(*this, mean, std);
  }

  void to_device(Device device) {
    static std::shared_ptr<Backend<T>> cpu_backend =
        std::make_shared<CpuBackend<T>>();
    if (device == Device::CPU) {
      backend = cpu_backend;
    } else {
      throw std::runtime_error("Device not supported");
    }
  }

  void print_info() const {
    std::cout << "Tensor of shape (";
    for (uint32_t i = 0; i < ndim - 1; ++i) {
      std::cout << shape[i] << ", ";
    }
    std::cout << shape[ndim - 1] << ") and size " << size << std::endl;
    std::cout << "Creator operation: " << creator_op->get_name() << std::endl;
    std::cout << "Backend: " << backend->get_name() << std::endl;
  }

  void backward(const Tensor<T> &upstream) {
    assert(
        upstream.shape == shape &&
        "Error in backward pass, upstream shape does not match tensor shape");
    assert(upstream.backend->get_name() == backend->get_name() &&
           "Tensors must be on the same device");
    assert(requires_grad &&
           "Tensor with requires_grad = false cannot be used in backward pass");

#ifdef DEBUG
    if (grad == nullptr) {
      grad = std::make_shared_for_overwrite<T[]>(size);
      std::cout << "Warning: gradient tensor was not initialized, creating one "
                   "with zeros"
                << __FILE__ << ":" << __LINE__ << std::endl;
    }
#endif // DEBUG

    std::copy(upstream.data.get(), upstream.data.get() + size, grad.get());
    creator_op->backward(upstream);
  }

  void backward() {
    auto grad_tensor = Tensor<T>(shape, false);
    grad_tensor.fill(1);
    std::fill(grad.get(), grad.get() + size, 1);
    creator_op->backward(grad_tensor);
  }

  void zero_grad() {
    assert(requires_grad &&
           "Tensor with requires_grad = false cannot have its gradient zeroed");
    std::fill(grad.get(), grad.get() + size, 0);
  }

  void detach() {
    requires_grad = false;
    this->grad = nullptr;
  }

private:
  void print_recursive(std::span<size_t> strides, size_t offset = 0) const {
    if (strides.size() == 1) {
      for (size_t i = 0; i < strides[0] - 1; ++i) {
        std::cout << data[i + offset] << ", ";
      }
      std::cout << data[strides[0] - 1 + offset];
    } else {
      for (size_t i = 0; i < strides[0]; ++i) {
        if (strides.size() == 2) {
          std::cout << "[";
        } else {
          std::cout << "\n";
        }
        print_recursive(strides.subspan(1), offset + i * strides[1]);
        if (strides.size() == 2) {
          std::cout << "]\n";
        }
      }
    }
  }
};

template <typename T>
Tensor<T> operator+(Tensor<T> const &a, Tensor<T> const &b) {
  assert(a.ndim == b.ndim && a.shape == b.shape &&
         "Tensors must have the same shape");
  assert(a.backend->get_name() == b.backend->get_name() &&
         "Tensors must be on the same device");

  return a.backend->add(a, b);
}

template <typename T>
Tensor<T> operator*(Tensor<T> const &a, Tensor<T> const &b) {
  assert(a.ndim == b.ndim && a.shape == b.shape &&
         "Tensors must have the same shape");
  assert(a.backend->get_name() == b.backend->get_name() &&
         "Tensors must be on the same device");

  return a.backend->mul(a, b);
}



using TensorF32 = Tensor<float>;
using TensorF64 = Tensor<double>;
using TensorI8 = Tensor<int8_t>;
using TensorI16 = Tensor<int16_t>;
using TensorI32 = Tensor<int32_t>;
using TensorI64 = Tensor<int64_t>;
using TensorU8 = Tensor<uint8_t>;
using TensorU16 = Tensor<uint16_t>;
using TensorU32 = Tensor<uint32_t>;
using TensorU64 = Tensor<uint64_t>;