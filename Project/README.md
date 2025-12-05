# Sketch-Based 3D Shape Generation (Reimagined)

This repository is now split into a **C++ backend** (OpenCV/OpenGL pipeline) and a **React frontend** (styled-components + optional three.js). The backend ingests sketches, generates meshes through extrusion/revolution/height-map modules, and performs server-side rendering. The frontend uploads sketches, polls rendered outputs, and can visualize meshes interactively.

## Repository Layout
- `backend/`: CMake-based C++ service with modules for sketch processing, mesh generation, and off-screen rendering.
- `frontend/`: Vite + React + styled-components client showing rendered images and an optional three.js viewer.
- `report/`, `docs/`: Project documentation and LaTeX report.

## Backend Quickstart
```bash
cd backend
rm -rf build && cmake -S . -B build && cmake --build build && ./build/sketch3d_service
```
If you restart frequently, make sure no old instance still listens on port 8080 (check with `lsof -i :8080` and `kill <PID>`). OpenCV/OpenGL detection is optional; if missing, the stubs still run in demo mode.

## Frontend Quickstart
```bash
cd frontend
npm install
npm run dev
```
The dev server proxies `/api` requests to `http://localhost:8080` by default.

## Next Steps
- Implement real HTTP endpoints inside `backend/src/api/HttpServer.cpp` (currently a stub invoking the pipeline directly).
- Replace mock renderer output with actual OpenGL FBO screenshots.
- Wire `frontend/src/services/api.ts` to backend endpoints to fetch real render data.
