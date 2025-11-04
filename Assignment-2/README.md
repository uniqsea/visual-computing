# Real-time Video Processing (Assignment 2)

## Dependencies
- C++17 compiler
- CMake 3.15+
- OpenGL 3.3, GLFW, GLM
- OpenCV (for camera capture and CPU filters)

## Build
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```
For Debug builds:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j
```

## Run
From the build:
```bash
./Assignment-2
```
