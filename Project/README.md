# Sketch-Based 3D Shape Generation

## Backend
```bash
cd backend
rm -rf build
cmake -S . -B build
cmake --build build
./build/sketch3d_service
```

## Frontend
```bash
cd frontend
npm install
npm run dev
