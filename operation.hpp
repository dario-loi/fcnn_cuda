#pragma once

#include <iostream>
#include <memory>
#include <string_view>
#include <type_traits>

// Forward declaration of Tensor class
template <typename T>
  requires std::is_arithmetic_v<T>
struct Tensor;

template <typename T>
  requires std::is_arithmetic_v<T>
class Operation {

  using TensorType = Tensor<T>;
  using ValueType = T;

public:
  virtual ~Operation() = default;

  virtual void backward(const Tensor<T> &) = 0;
  virtual std::string_view get_name() const = 0;
  virtual void print_info() const = 0;
};

template <typename T>
  requires std::is_arithmetic_v<T>
class CreateOperation : public Operation<T> {
public:
  const std::string_view name = "leaf";
  void backward(const Tensor<T> &) override {}
  std::string_view get_name() const { return name; }

  void print_info() const override { std::cout << "Leaf tensor" << std::endl; }
}; // leaf tensor

template <typename T>
  requires std::is_arithmetic_v<T>
class AddOperation : public Operation<T> {
public:
  AddOperation(Tensor<T> const &a, Tensor<T> const &b) : a(a), b(b) {}

  ~AddOperation() = default;

  const std::string_view name = "add";
  void backward(const Tensor<T> &upstream) override {
    a.backward(upstream);
    b.backward(upstream);
  }
  std::string_view get_name() const { return name; }

  void print_info() const override {
    std::cout << "Add operation between tensors of shape (";
    for (size_t i = 0; i < a.ndim - 1; ++i) {
      std::cout << a.shape[i] << ", ";
    }
    std::cout << a.shape[a.ndim - 1] << ") and (";
    for (size_t i = 0; i < b.ndim - 1; ++i) {
      std::cout << b.shape[i] << ", ";
    }
    std::cout << b.shape[b.ndim - 1] << ")" << std::endl;
  }

private:
  Tensor<T> a;
  Tensor<T> b;
}; // add operation

template <typename T>
  requires std::is_arithmetic_v<T>
class MulOperation : public Operation<T> {
public:
  MulOperation(Tensor<T> const &a, Tensor<T> const &b) : a(a), b(b) {}

  ~MulOperation() = default;

  const std::string_view name = "mul";
  void backward(const Tensor<T> &upstream) override {
    a.backward(upstream * b);
    b.backward(upstream * a);
  }
  std::string_view get_name() const { return name; }

  void print_info() const override {
    std::cout << "Mul operation between tensors of shape (";
    for (size_t i = 0; i < a.ndim - 1; ++i) {
      std::cout << a.shape[i] << ", ";
    }
    std::cout << a.shape[a.ndim - 1] << ") and (";
    for (size_t i = 0; i < b.ndim - 1; ++i) {
      std::cout << b.shape[i] << ", ";
    }
    std::cout << b.shape[b.ndim - 1] << ")" << std::endl;
  }

private:
  Tensor<T> a;
  Tensor<T> b;
}; // mul operation