cmake_minimum_required(VERSION 3.0)
project(cost_model)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 14)

add_executable(cost_model main.cpp IsPrimeNumber.h GetPrimeFactor.h ChooseWidth.h GetWidth.h PrintTreeStructure.h CostModel.h timer.h)
target_link_libraries(cost_model ${MPI_CXX_LIBRARIES} pthread)