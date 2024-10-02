#pragma once

#include <memory>
#include <type_traits>
#include <numeric>
#include <span>
#include <vector>
#include <ranges>
#include <iostream>
#include "backend.hpp"

template <typename Backend, typename T>
  requires std::is_arithmetic_v<T>
struct Tensor {

  Backend [[no_unique_address]] backend;
  std::unique_ptr<T[]> data;
  std::vector<size_t> shape;
  size_t size;
  size_t ndim;

  Tensor(std::vector<size_t> &&m_shape)
      : backend(backend), shape(std::move(m_shape)),
        size(std::accumulate(shape.begin(), shape.end(), 1,
                             std::multiplies<size_t>())),
        ndim(shape.size()) {
    data = std::make_unique<T[]>(size);
  }

  template <typename... Indices> T &operator()(Indices... indices) {

    size_t index = 0;
    size_t indices_array[] = {static_cast<size_t>(indices)...};
    for (size_t i = 0; i < ndim; ++i) {
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
  void fill(T value) { backend.fill(*this, value); }
  void rand() { backend.rand(*this); }

private:
  void print_recursive(std::span<size_t> strides, size_t offset = 0) {

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
