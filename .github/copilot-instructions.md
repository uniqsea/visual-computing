# Copilot instructions for this repo (visual-computing)

This is a small C++14 OpenCV assignment project. The active code lives in `Assignment-1/` with a single CMake target. Use these conventions and workflows when editing or adding code.

## Repo layout and targets
- `Assignment-1/CMakeLists.txt`: single-target CMake project using OpenCV.
  - Current target: `Assignment-1` built from `main.cpp`.
  - OpenCV integration: `find_package(OpenCV REQUIRED)` and `target_link_libraries(… ${OpenCV_LIBS})`.
- `Assignment-1/main.cpp`: program entrypoint (add more sources here by listing them in `add_executable`).
- There may be a previously used target name `panorama` in older snippets; keep the target name consistent across `add_executable` and `target_link_libraries`.

## Build and run (macOS, zsh)
Dependencies: OpenCV 4. Install via Homebrew if missing.

```zsh
# Install OpenCV (if needed)
brew install opencv

# Configure and build out-of-source from the assignment folder
cd Assignment-1
cmake -S . -B build
cmake --build build -j

# Run the binary
./build/Assignment-1
```

If CMake cannot locate OpenCV, set `OpenCV_DIR` to the Homebrew package config:
```zsh
export OpenCV_DIR="$(brew --prefix)/opt/opencv/lib/cmake/opencv4"
cmake -S . -B build
```
(Use `opencv@4` in the path if that’s your installed formula name.)

## VS Code usage
- Use the CMake Tools extension. Set the source directory to the assignment folder:
  - Settings: `cmake.sourceDirectory = ${workspaceFolder}/Assignment-1`.
- Configure/Build from the CMake Tools status bar.
- Debug: choose the `Assignment-1` target and start debugging (LLDB on macOS).

## Adding files and linking
- Prefer explicitly listing sources in `add_executable`:
  ```cmake
  add_executable(Assignment-1
      main.cpp
      stitcher.cpp
      utils/io.cpp)
  ```
- Headers are picked up via `include_directories(${OpenCV_INCLUDE_DIRS})`. Add any project includes via `target_include_directories` if you create subfolders.

## Common pitfalls (and quick fixes)
- CMake error "Cannot find source file: main.cpp" or "No SOURCES given to target":
  - Cause: configuring from the wrong source directory (e.g., repo root) or stale build cache.
  - Fix:
    ```zsh
    # From repo root
    rm -rf build
    cd Assignment-1 && rm -rf build && cmake -S . -B build && cmake --build build -j
    ```
  - Ensure your CMake source dir is `Assignment-1/` (see VS Code setting above).
- OpenCV not found:
  - Ensure Homebrew OpenCV is installed; export `OpenCV_DIR` as above; or pass `-DOpenCV_DIR=…` to `cmake -S . -B build`.

## Project conventions
- Language standard: C++14 (`set(CMAKE_CXX_STANDARD 14)`).
- No test framework or CI configured; keep examples and small driver code in `main.cpp`.
- Keep target name and link directives in sync; if you rename the target, update both `add_executable` and `target_link_libraries`.

## When extending the project
- Add new `.cpp` files to `add_executable` and group them under folders as needed.
- If you introduce libraries, prefer `target_link_libraries(Assignment-1 PRIVATE <lib>)` and `target_include_directories` rather than global `include_directories`.
- Document minimal run instructions in `README.md` if you add required CLI args or data files.
