cmake_minimum_required(VERSION 3.11)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)

project(fcnn LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fcnn main.cu)