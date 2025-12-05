cmake_minimum_required(VERSION 3.17)
project(Sketch3DBackend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_OPENGL "Enable OpenGL renderer" ON)
option(ENABLE_OPENCV "Enable OpenCV processing" ON)

if(ENABLE_OPENCV)
  find_package(OpenCV QUIET)
  if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found; SketchProcessor will operate in demo mode.")
  endif()
endif()

find_package(Threads REQUIRED)

if(ENABLE_OPENGL)
  find_package(OpenGL QUIET)
  find_package(glfw3 QUIET)
  if(NOT OPENGL_FOUND OR NOT glfw3_FOUND)
    message(WARNING "OpenGL/GLFW not found; OffscreenRenderer will emit placeholder files.")
  endif()
endif()

add_library(sketch3d_backend STATIC
  src/api/HttpServer.cpp
  src/core/PipelineController.cpp
  src/sketch/SketchProcessor.cpp
  src/geometry/Mesh.cpp
  src/geometry/ExtrusionGenerator.cpp
  src/geometry/RevolutionGenerator.cpp
  src/geometry/HeightMapGenerator.cpp
  src/rendering/OffscreenRenderer.cpp
  src/rendering/MeshSerializer.cpp)

target_include_directories(sketch3d_backend PUBLIC include)

if(OpenCV_FOUND)
  target_link_libraries(sketch3d_backend PUBLIC ${OpenCV_LIBS})
endif()
if(OPENGL_FOUND)
  target_link_libraries(sketch3d_backend PUBLIC OpenGL::GL)
endif()
if(glfw3_FOUND)
  target_link_libraries(sketch3d_backend PUBLIC glfw)
endif()
target_link_libraries(sketch3d_backend PUBLIC Threads::Threads)

add_executable(sketch3d_service src/main_entry.cpp)
target_link_libraries(sketch3d_service PRIVATE sketch3d_backend)
