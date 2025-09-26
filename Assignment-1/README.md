Build
1) mkdir build && cd build
2) cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j

CLI quick reference
- Mode
  - Experiment: pairwise sweep over detector × blending × RANSAC thresholds (simple mode disabled)

- Options
  - --experiment                run experiment mode (default)
  - --detector SIFT|ORB|AKAZE   feature detector/descriptor
  - --matcher BF|ANN            matcher (ANN = approximate kNN; ORB/AKAZE use LSH, BF uses Hamming)
  - --blending OVERLAY|FEATHER  blending backend (OpenCV detail)
  - --threshold <float>        RANSAC reprojection threshold in pixels (default 3.0)
  - --output <dir>             base output dir (anchored to project root when relative)

- Notes
  - Outputs are time‑stamped under `results/experiments/<timestamp>/`
  - Per image pair a folder named `<img1_stem>_<img2_stem>/` is created, containing:
    - Baselines: `my_panorama.jpg`, `opencv_panorama.jpg`
    - For each config: `<detector>/<matcher>/<blending>/tXX/` with
      - `stitch.jpg`, `matches.jpg`, `kps_img1.jpg`, `kps_img2.jpg`, `hist.jpg`
  - A CSV summary `experimental_results.csv` is written at the timestamp root
  - Each run writes `RUN_CONFIG.txt` and `PARAMETERS.md` with settings and parameter descriptions

Examples
- Experiment
  ./Assignment-1 --experiment --output results images/set1/img1.jpg images/set1/img2.jpg
