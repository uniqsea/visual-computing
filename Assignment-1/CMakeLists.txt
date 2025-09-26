cmake_minimum_required(VERSION 3.10)
project(Assignment-1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES
    main.cpp
    src/FeatureDetector.cpp
    src/FeatureMatcher.cpp
    src/HomographyEstimator.cpp
    src/PanoramaStitcher.cpp
    src/Evaluator.cpp
    src/ImageWarper.cpp
)

set(HEADERS
    src/include/FeatureDetector.hpp
    src/include/FeatureMatcher.hpp
    src/include/HomographyEstimator.hpp
    src/include/PanoramaStitcher.hpp
    src/include/Evaluator.hpp
    src/include/ImageWarper.hpp
)

add_executable(Assignment-1 ${SOURCES} ${HEADERS})

target_link_libraries(Assignment-1 ${OpenCV_LIBS})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(Assignment-1 PRIVATE -Wall -Wextra -O2)
endif()

target_include_directories(Assignment-1 PRIVATE src/include/ src/)
