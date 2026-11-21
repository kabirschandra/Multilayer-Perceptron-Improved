cmake_minimum_required(VERSION 3.16)
project(MLP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MLP_BUILD_TESTS "Build test executables" ON)
option(MLP_NATIVE "Optimise for the host CPU" ON)
option(MLP_FAST_MATH "Enable aggressive floating-point optimisations" OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen/Eigen/Dense")
    message(FATAL_ERROR "Missing external/eigen (Eigen headers)")
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/cereal-1.3.2/include/cereal/cereal.hpp")
    message(FATAL_ERROR "Missing external/cereal-1.3.2/include (cereal headers)")
endif()

add_library(mlp_headers INTERFACE)
target_include_directories(mlp_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cereal-1.3.2/include
)

include(CheckIPOSupported)
check_ipo_supported(RESULT MLP_IPO_SUPPORTED OUTPUT MLP_IPO_ERROR LANGUAGES CXX)
if(NOT MLP_IPO_SUPPORTED)
    message(STATUS "LTO unavailable: ${MLP_IPO_ERROR}")
endif()

function(mlp_options target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-DNDEBUG>
        )
        if(MLP_NATIVE)
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-march=native>)
        endif()
        if(MLP_FAST_MATH)
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-ffast-math>)
        endif()
    endif()
    if(MLP_IPO_SUPPORTED)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endfunction()

add_executable(main src/main.c++)
target_link_libraries(main PRIVATE mlp_headers)
mlp_options(main)

if(MLP_BUILD_TESTS)
    enable_testing()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_target(download_datasets
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testing/download_data.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Download and validate Iris/MNIST datasets")
    add_test(NAME datasets COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testing/download_data.py)
    set_tests_properties(datasets PROPERTIES FIXTURES_SETUP real_datasets TIMEOUT 300)
    foreach(name IN ITEMS xor save_load iris mnist)
        if(name STREQUAL "xor")
            set(source testing/XOR/xor_test.c++)
        elseif(name STREQUAL "save_load")
            set(source testing/Save_load/save_load_test.c++)
        elseif(name STREQUAL "iris")
            set(source testing/IRIS/test_iris.c++)
        else()
            set(source testing/MNIST/test_mnist.c++)
        endif()
        add_executable(test_${name} ${source})
        target_link_libraries(test_${name} PRIVATE mlp_headers)
        target_compile_definitions(test_${name} PRIVATE MLP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testing")
        target_include_directories(test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/testing)
        mlp_options(test_${name})
        add_test(NAME ${name} COMMAND test_${name})
        if(name STREQUAL "iris" OR name STREQUAL "mnist")
            set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED real_datasets TIMEOUT 300)
        endif()
    endforeach()
endif()
