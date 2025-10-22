cmake_minimum_required(VERSION 3.15)
project(RealTimeVideoProcessing VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add Homebrew paths for macOS
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
endif()

# Find required packages
find_package(OpenGL REQUIRED)

# Try to find GLFW3
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    # Fallback to pkg-config
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW REQUIRED glfw3)
    set(GLFW_LIBRARIES ${GLFW_LINK_LIBRARIES})
    include_directories(${GLFW_INCLUDE_DIRS})
    link_directories(${GLFW_LIBRARY_DIRS})
endif()

find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends
    ${OpenCV_INCLUDE_DIRS}
)

# GLAD source
set(GLAD_SOURCES
    external/glad/gl.c
)

# Dear ImGui sources
set(IMGUI_SOURCES
    external/imgui/imgui.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

# Application sources
set(APP_SOURCES
    src/main.cpp
    src/Application.cpp
    src/VideoCapture.cpp
    src/Renderer.cpp
    src/ShaderProgram.cpp
    src/Texture.cpp
    src/GUI.cpp
    src/filters/FilterManager.cpp
    src/filters/PixelationFilter.cpp
    src/filters/CartoonFilter.cpp
    src/transforms/AffineTransform.cpp
    src/utils/Timer.cpp
    src/utils/PerformanceLogger.cpp
)

# Create executable
add_executable(${PROJECT_NAME}
    ${APP_SOURCES}
    ${GLAD_SOURCES}
    ${IMGUI_SOURCES}
)

# Link libraries
if(glfw3_FOUND)
    target_link_libraries(${PROJECT_NAME}
        OpenGL::GL
        glfw
        ${OpenCV_LIBS}
    )
else()
    target_link_libraries(${PROJECT_NAME}
        OpenGL::GL
        ${GLFW_LIBRARIES}
        ${OpenCV_LIBS}
    )
endif()

# Platform-specific settings
if(APPLE)
    target_link_libraries(${PROJECT_NAME}
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()

# Compiler flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_BUILD)
    message(STATUS "Building in Debug mode")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_definitions(${PROJECT_NAME} PRIVATE RELEASE_BUILD)
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /O2)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -O3)
    endif()
    message(STATUS "Building in Release mode")
endif()

# Copy shaders to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders
     DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Print configuration
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

