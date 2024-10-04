
#include "tensor.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <numeric>
#include <span>

int main()
{

    auto T = TensorF32({ 1, 2, 2 * 2 });
    auto T2 = TensorF32({ 1, 2, 2 * 2 });

    T.randn();
    T2.randn(0, 5);

    auto Z = (T * T2);
    auto Z2 = (T + T2);
    auto Z3 = (Z * Z2);

    std::cout << sizeof(TensorF32) << std::endl;
    std::cout << sizeof(TensorF64) << std::endl;
    Z3.backward();

    // Z.print();
    Z.print_info();
    Z.creator_op->print_info();
    Z.print_grad();
    T.print_grad();
    T2.print_grad();

    T.print();
    T2.print();

    return 0;
}
