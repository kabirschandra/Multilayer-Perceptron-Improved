cmake_minimum_required(VERSION 3.16)

project(MLP LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main src/main.c++)

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cereal-1.3.2/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/

)














