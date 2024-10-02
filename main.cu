#include "aliases.hpp"
#include "backend_cpu.hpp"
#include "tensor.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <numeric>
#include <span>
int main()
{
    CPUTensorF32 T = CPUTensorF32 { { 2, 2, 2, 4 } };

    printf("Size of T: %d\n", T.size);
    printf("Rank of T: %d\n", T.ndim);
    T.fill(1.0f);

    T.rand();
    T.pretty_print();
    T.print();
    return 0;
}
