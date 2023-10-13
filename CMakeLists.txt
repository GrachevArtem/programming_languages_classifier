cmake_minimum_required(VERSION 3.16)

project(programming_lang_classifier VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(WIN32)
    add_compile_definitions(WINDOWS=1)
endif ()

if (UNIX AND NOT APPLE)
    add_compile_definitions(LINUX=1)
endif ()

if (APPLE)
    add_compile_definitions(APPLE=1)
endif ()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    add_compile_definitions(_DEBUG=1)
else()
    add_compile_definitions(NDEBUG=1)
endif(${CMAKE_BUILD_TYPE} MATCHES Debug)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /Zc:__cplusplus")
endif()


add_executable(train src/train.cpp)

find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)

target_compile_options(train PRIVATE -fopenmp)
target_link_libraries(train PRIVATE OpenMP::OpenMP_CXX ${ARMADILLO_LIBRARIES})

add_executable(test src/test.cpp)

find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)

target_compile_options(test PRIVATE -fopenmp)
target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX ${ARMADILLO_LIBRARIES})

add_executable(analysis src/analysis.cpp)

find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)

target_compile_options(analysis PRIVATE -fopenmp)
target_link_libraries(analysis PRIVATE OpenMP::OpenMP_CXX ${ARMADILLO_LIBRARIES})
