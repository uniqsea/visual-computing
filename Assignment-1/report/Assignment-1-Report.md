# Assignment 1 Report: Real‑Time Panorama Stitching with Experimental Evaluation

Visual Computing: Interactive Computer Graphics and Vision — Aarhus University (2025)

Author: <Your Name>

---

## Abstract

This report reproduces a classic panorama stitching pipeline as part of an assignment, with the goal of becoming familiar with the standard stages and underlying principles. The implementation supports SIFT/ORB/AKAZE detectors, BF/ANN (Approximate Nearest Neighbor) matching with robust filtering, RANSAC‑based homography estimation, and multiple blending schemes. As a practical robustness choice, an order‑invariant global stitching strategy is adopted: a central reference image is selected, pairwise homographies are estimated only between neighbors, each image transform is composed into the reference, and a single global warp with feathered blending is performed. Optional cylindrical/spherical pre‑warping is added for rotation‑dominant scenes, and an OpenCV `cv::Stitcher` baseline is included for comparison. Experiments on indoor/outdoor sequences analyze detector/matcher/blending choices, RANSAC thresholds, and the impact of pre‑warping on robustness and visual quality.

---

## 1. Introduction

Panorama stitching is a well‑established technique that combines multiple overlapping views into a single wide‑field image. The standard pipeline comprises keypoint detection and description, descriptor matching with outlier filtering, homography estimation via RANSAC, image warping to a common frame, and blending. This assignment focuses on faithfully reproducing that pipeline to understand how each component contributes to overall quality. The report also compares practical design choices—detector type, match filtering, RANSAC threshold, composition strategy, and blending.

The goals are to: (i) implement a complete reference pipeline, (ii) evaluate key design choices on representative data, (iii) compare against an OpenCV baseline, and (iv) summarize trade‑offs with qualitative and quantitative evidence. The emphasis is educational rather than novelty.

---

## 2. Method

### 2.1 Pre‑Warping (Optional)

To handle rotation‑dominant scenes and large fields of view, the pipeline can optionally pre‑warp each input to a cylindrical or spherical surface before keypoint detection. This reduces perspective distortion in the matching domain and improves robustness for pure‑rotation panoramas.

- Implementation: `src/ImageWarper.cpp`, interface `src/include/ImageWarper.hpp`.
- Backend: OpenCV stitching/detail warpers (`cv::detail::CylindricalWarper`, `cv::detail::SphericalWarper`) — replacing an earlier custom remap to avoid re‑inventing components.
- Types: `NONE` (default), `CYLINDRICAL`, `SPHERICAL`.
- Parameter: focal length in pixels (`--focal`), defaulting to ≈0.5×image width if unspecified.

The main pipeline can thus operate either on raw images (plane) or pre‑warped images (cylindrical/spherical coordinates).

### 2.2 Keypoint Detection and Description

The implementation supports SIFT, ORB, and AKAZE. All detectors operate on grayscale images for stability. Detectors output `cv::KeyPoint` with position, scale, and orientation; descriptors are used for matching.

- Implementation: `src/FeatureDetector.cpp` and `src/include/FeatureDetector.hpp`.
- Detectors: `SIFT`, `ORB`, `AKAZE` selected via `--detector`.
- We record detection time and keypoint counts for analysis.

### 2.3 Matching and Robust Filtering

BF and ANN (approximate nearest neighbor) matchers are available. For binary descriptors (e.g., ORB/AKAZE default), ANN is replaced by BF+Hamming to ensure compatibility and quality. The pipeline uses Lowe’s ratio test (k‑NN, k=2) or distance thresholds to filter spurious matches and records total vs. good matches and distance histograms.

- Implementation: `src/FeatureMatcher.cpp` and `src/include/FeatureMatcher.hpp`.
- Ratio test and statistics are corrected to report candidate pairs vs. filtered matches consistently.
- Empty descriptor safeguards prevent exceptions on difficult images.

### 2.4 Homography Estimation (RANSAC)

Given matched keypoints, a planar homography is estimated with RANSAC, varying the reprojection threshold to study inlier counts and runtime. Direction is standardized: for two‑image stitching, the stitcher receives H mapping the second image into the first image coordinates.

- Implementation: `src/HomographyEstimator.cpp` and `src/include/HomographyEstimator.hpp`.
- Parameters: reprojection threshold (`--threshold`), confidence, max iterations.

### 2.5 Global, Order‑Invariant Composition and Single‑Pass Blending (OpenCV blenders)

Instead of incrementally warping every new image into a growing panorama (which compounds interpolation and alignment errors), this implementation:

1) Choose the middle image as reference (index ⌊N/2⌋) to symmetrize error.
2) Estimate homographies only between adjacent pairs (i→i+1).
3) Compose transforms to the reference for each image: `T(i→ref)` by chaining/ inverses.
4) Compute one global canvas bounding box and apply a single warp per image.
5) Blend all warped images once using OpenCV stitching/detail blenders.

- Blending backend (replaces hand‑crafted overlay/feathering):
  - Overlay: `cv::detail::Blender::NO`
  - Feathering: `cv::detail::Blender::FEATHER` (with `setSharpness`)
  - Multiband: `cv::detail::MultiBandBlender`
- Code: `src/PanoramaStitcher.cpp`, interface `src/include/PanoramaStitcher.hpp`.
- Benefit: reduces order sensitivity and interpolation loss; reuses robust, battle‑tested components.

Terminology: “Global” here means a single common reference canvas and a single blending pass over all images, not global camera optimization (no bundle adjustment).

### 2.6 Baseline: OpenCV cv::Stitcher

We include a strong baseline using OpenCV `cv::Stitcher` with PANORAMA/SCANS modes and relaxed confidence fallbacks. The baseline typically uses spherical/cylindrical warping, seam finding, and exposure compensation internally.

- Integration: `main.cpp` (`runSimpleMode` and `runExperimentMode`), saved alongside our output for direct visual comparison.

---

## 3. Implementation Overview

- Entry point and CLI: `main.cpp`
  - Simple mode: global, order‑invariant pipeline with optional pre‑warping, plus OpenCV baseline.
  - Experiment mode: grid search over detector/matcher/blending/threshold and pair‑wise evaluation.
- Modules
  - `FeatureDetector` (SIFT/ORB/AKAZE): `src/FeatureDetector.cpp`, `src/include/FeatureDetector.hpp`.
  - `FeatureMatcher` (BF/ANN + ratio test): `src/FeatureMatcher.cpp`, `src/include/FeatureMatcher.hpp`.
  - `HomographyEstimator` (RANSAC): `src/HomographyEstimator.cpp`, `src/include/HomographyEstimator.hpp`.
  - `PanoramaStitcher` (global warp + feathering/overlay): `src/PanoramaStitcher.cpp`, `src/include/PanoramaStitcher.hpp`.
  - `ImageWarper` (cylindrical/spherical pre‑warp): `src/ImageWarper.cpp`, `src/include/ImageWarper.hpp`.
  - `Evaluator` (experiments + logging/visualization): `src/Evaluator.cpp`, `src/include/Evaluator.hpp`.

Engineering features include consistent homography direction, robust matcher selection, grayscale detection, unified output layout, and time‑stamped run directories for reproducibility.

---

## 4. Experimental Setup

### 4.1 Data

We captured three sets of overlapping photos (both indoor and outdoor): `images/set1`, `images/set2`, `images/set3`. Each set includes sequences with substantial overlap and moderate baselines. Conditions vary in lighting, texture richness, and motion. We use adjacent pairs for two‑image experiments and the full set for multi‑image composition in simple mode.

### 4.2 Configurations

- Detectors: SIFT, ORB, AKAZE  (`--detector`).
- Matchers: BF, ANN (approximate kNN) with automatic fallback to BF‑Hamming for binary descriptors.
- Blending: Simple overlay, Feathering (`--blending`).
- RANSAC reprojection threshold: {0.5, 1.0, 2.0, 3.0, 5.0}.
- Pre‑warping (simple mode only): NONE, CYLINDRICAL, SPHERICAL with `--focal`.
- Baseline: OpenCV `cv::Stitcher` [PANORAMA/SCANS + confidence fallback].

### 4.3 Metrics and Outputs

For each experiment we record:

- Keypoints per image; detection time.
- Total vs. good matches; match ratio; matching time; distance histogram.
- RANSAC inliers; estimation time; success flag.
- Stitching/blending time; panorama size.
- Total runtime per configuration.

Outputs are saved under `results/experiments/<timestamp>/` and include CSV (`experimental_results.csv`), a summary table, a short report, match visualizations, keypoint overlays (rendered as crosses), and per‑pair baselines. Simple mode outputs are in `results/simple/<timestamp>/` with our panorama and the OpenCV baseline.

### 4.4 Commands

- Simple mode (with cylindrical warping):
```
./build/Assignment-1 \
  --detector SIFT --matcher BF --blending FEATHER \
  --warper CYLINDRICAL --focal 800 \
  --threshold 3.0 --output results \
  images/set2/img1.jpg images/set2/img2.jpg images/set2/img3.jpg
```

- Experiment mode:
```
./build/Assignment-1 --experiment --output results \
  images/set1/img1.jpg images/set1/img2.jpg
```

---

## 5. Results and Analysis

This section summarizes key observations. Refer to the CSV for full quantitative results.

### 5.1 Detector and Matcher

- SIFT often produces more discriminative matches with higher inlier ratios, at a cost in runtime.
- ORB/AKAZE are faster and adequate in high‑texture scenes but may suffer on low‑contrast surfaces; fallback to BF+Hamming is necessary for binary descriptors.
- Ratio test thresholds around 0.7–0.8 provide a good recall/precision balance; too strict (<0.6) can reduce inliers noticeably.

### 5.2 RANSAC Threshold

- Lower thresholds (0.5–1.0 px) tighten geometric consistency but risk rejecting valid matches under noise; higher thresholds (3.0–5.0 px) increase inlier counts but may admit mild misalignments.
- In our data, ~2–3 px is often a robust choice.

### 5.3 Blending

- Feathering substantially improves seam quality over simple overlay, especially with exposure differences and small misalignments. Runtime overhead remains modest.

### 5.4 Pre‑Warping (Simple Mode)

- Cylindrical warping improves alignment in rotation‑dominant sequences (e.g., hand‑held pan), reducing ghosting and seam curvature. Spherical warping is beneficial for very wide fields of view.
- Pre‑warping may slightly reduce descriptor repeatability where strong local distortions occur; the net effect is positive for large FOV rotation panoramas.

### 5.5 Composition Strategy

- The global, order‑invariant strategy avoids the error accumulation and feature degradation seen in incremental pipelines that repeatedly detect on resampled panoramas. Qualitatively, alignment is more stable, and panoramas are more compact (tighter bounding boxes).

### 5.6 Baseline Comparison

- OpenCV `cv::Stitcher` combines multiple strong components (projection selection, seam finding, exposure compensation, bundle adjustment). The reproduced pipeline is competitive on many scenes with cylindrical warping and feathering. The baseline remains stronger on extreme cases (very large parallax, strong exposure shifts) due to seam and gain optimization.

> Tip: Include representative side‑by‑side crops from `results/simple/<timestamp>/my_panorama.jpg` vs. `opencv_panorama.jpg` and per‑pair outputs in experiments for qualitative evidence.

### 5.7 Representative Figures (Latest Experiment)

- Baseline vs. Ours (pair_0_1):

  ![Ours](../results/experiments/20250926-034804/pair_0_1/my_panorama.jpg)
  ![OpenCV Baseline](../results/experiments/20250926-034804/pair_0_1/opencv_panorama.jpg)

- Matches visualization（SIFT + BF, good matches）:

  ![Matches](../results/experiments/20250926-034804/matches_pair_0_1_SIFT_BruteForce.jpg)

- Keypoints as crosses（SIFT, Feathering, threshold=3.0）：

  ![Keypoints img1](../results/experiments/20250926-034804/kps_pair_0_1_SIFT_img1_Feathering_t30.jpg)
  ![Keypoints img2](../results/experiments/20250926-034804/kps_pair_0_1_SIFT_img2_Feathering_t30.jpg)

- Match distance histogram（SIFT + BF, Feathering, threshold=3.0）：

  ![Histogram](../results/experiments/20250926-034804/hist_pair_0_1_SIFT_BruteForce_Feathering_t30.jpg)

### 5.8 Summary Table (Selected Configurations)

| Detector | Matcher | Blending | Thr (px) | Keypoints | Good Matches | Inliers | Total Time (ms) | Success |
|---|---|---|---:|---:|---:|---:|---:|:--:|
| SIFT | BF | Feathering | 1.0 | 10796 | 414 | 115 | 1899.71 | Yes |
| SIFT | BF | Feathering | 3.0 | 10796 | 414 | 206 | 1842.75 | Yes |
| SIFT | BF | Feathering | 5.0 | 10796 | 414 | 233 | 1801.80 | Yes |
| ORB  | BF | Feathering | 3.0 | 1000  | 63  | 34  | 916.02  | Yes |
| AKAZE| BF | Feathering | 3.0 | 15442 | 657 | 320 | 1564.10 | Yes |



---

## 6. Discussion

Strengths (as a learning reproduction):
- Order‑invariant global composition reduces drift and interpolation loss.
- Optional cylindrical/spherical pre‑warping aligns better with rotation panoramas.
- Robust matching pipeline with descriptor‑type awareness and empty‑descriptor safeguards.
- Consistent outputs and a strong OpenCV baseline for fair comparison.

Limitations and Failure Modes:
- Pure homography assumes a single dominant plane or pure rotation; significant parallax can still cause misalignment.
- Without seam finding and exposure compensation, visible seams may remain under large illumination changes.
- Pre‑warping requires a focal parameter; poor choices can under‑ or over‑curve the result.

Potential Improvements:
- Add seam finding (graph‑cut or DP), exposure/gain compensation, and true multi‑band blending.
- Cross‑check filtering and spatial consistency checks pre‑RANSAC; parameter auto‑tuning.
- Incorporate bundle adjustment for multi‑image sets.

---

## 7. Conclusion

This report reproduces a standard panorama stitching pipeline with a comprehensive evaluation harness. The global, order‑invariant composition and optional cylindrical/spherical pre‑warping improve robustness and visual quality over a naive incremental baseline. Experiments show detector/matcher choices and RANSAC thresholds materially influence accuracy and runtime. The OpenCV `cv::Stitcher` baseline remains a strong reference, while the reproduced pipeline achieves competitive quality on many scenes. Implementation now uses OpenCV stitching/detail warpers and blenders in place of earlier custom code. Future work (for learning) includes seam and exposure handling and adding global optimization for larger panoramas.

---

## 8. Implementation Notes (Updates)

- Pre‑warping: switched from a custom remap to OpenCV stitching/detail warpers (`CylindricalWarper`, `SphericalWarper`).
- Blending: switched from custom overlay/feathering to OpenCV stitching/detail blenders (NO/FEATHER/MULTIBAND) for both two‑image and multi‑image pipelines.
- “Global” refers to a shared canvas and single blending pass; no bundle adjustment is performed.

---

## Appendix: Reproducibility and Usage

### A.1 Build & Run

```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Simple mode (global stitching, optional warper):
```
./Assignment-1 --detector SIFT --matcher BF --blending FEATHER \
  --warper CYLINDRICAL --focal 800 --threshold 3.0 --output results \
  <img1> <img2> [<img3> ...]
```

Experiment mode (pairwise sweep):
```
./Assignment-1 --experiment --output results <img1> <img2> [<img3> ...]
```

### A.2 Output Structure

- Simple: `results/simple/<timestamp>/my_panorama.jpg`, `opencv_panorama.jpg`.
- Experiments: `results/experiments/<timestamp>/experimental_results.csv`, `results_table.txt`, `experiment_report.md`, per‑pair `my_panorama.jpg` and `opencv_panorama.jpg`, match visualizations, and keypoint overlays.

### A.3 Code Map

- `main.cpp`: CLI, modes, baseline integration.
- `src/include/*.hpp`, `src/*.cpp`: modules for detection, matching, RANSAC, stitching, warping, and evaluation.



# 1. 简介

全景拼接是计算机视觉中的重要任务，广泛应用于手机相机、街景地图和虚拟现实。其目标是将一系列部分重叠的图像对齐并融合，生成一个无缝的全景图。在如今日常生活中已经很常见，是一项成熟的技术，其基于特征处理的流程十分经典，包括**特征检测、特征匹配、单应性估计、图像变换和融合等**。

本报告中我基于 OpenCV 实现了完整的 pipeline，并通过对比实验分析不同设计选择的影响。

# 2、实现

## 2.1 数据集

为了测试和对比全景拼接效果，我使用自己的设备(iPhone 13 pro)采集了5组照片，包括3组室内，2组室外。每组照片，保持50%左右的重叠率，并且保持是The camera only (mostly) rotates。数据集概况如表所示：

| 数据描述             | 图片数量 | lighting                     | motion blur                    | texture richness                         |
| -------------------- | -------- | ---------------------------- | ------------------------------ | ---------------------------------------- |
| 办公楼（室内）       | 2        | 光照充足，日光，部分区域过曝 | 无模糊，手持拍摄造成的轻微抖动 | 特殊墙壁造成的极高复杂度的纹理           |
| 办公楼（室外，仰拍） | 2        | 傍晚拍摄，暖光，对比度明显   | 无模糊，手持拍摄造成的轻微抖动 | 中等，建筑外墙和窗户                     |
| 宿舍楼（室外）       | 3        | 光照充足，日光               | 无模糊，十分清晰               | 高，建筑外墙(颜色不同)，树木，车辆，马路 |
| 图书馆（室内）       | 2        | 光照充足，日光               | 无模糊，手持拍摄造成的轻微抖动 | 中等（海报，盆栽）                       |
| 教学楼（室内）       | 2        | 光照充足，日光               | 无模糊，手持拍摄造成的轻微抖动 | 中等，木质楼梯和长椅，颜色单一           |

数据如图所示，

## 2.2 实验设置

为了完整实现pipline，并充分探究和对比不同方法对结果的影响，我进行以下参数的调整和对比：

- 特征检测 我比较了三种检测器：

  - **SIFT**：鲁棒但速度较慢；

  - **ORB**：速度快但对光照/旋转敏感；

  - **AKAZE**：折中方法。

对于每种方法，统计关键点数量与检测耗时。

- 特征匹配：Matcher: {BF, ANN}

### 2.2 特征匹配

采用 **Brute Force (BF)** ，输出指标包括匹配数、好匹配数和匹配时间。同时绘制匹配距离直方图。

### 2.3 单应性估计

通过 **RANSAC** 从匹配点对中估计单应性矩阵。改变重投影阈值（0.5,1,2,3,5 ），比较：

- 内点数
- 拼接成功率与视觉质量
- 运行时间

Lowe’s ratio统一调整为经验值0.8[Lowe],RANSAC max_iterations 2000次

### 2.5 图像融合

实现并比较三种融合策略：

1. **Overlay**：简单覆盖，计算快但接缝明显；
2. **Feathering**：线性加权，过渡更平滑；

### 2.6 实验评估框架

我实现了一个实验框架，可以批量运行 Detector × Blending × RANSAC 阈值组合，输出：

- 关键点数
- 匹配统计
- 内点数
- 运行时间
- 拼接结果截图
- 匹配可视化与直方图
- CSV 表格
- RANSAC 阈值: {0.5, 1, 2, 3, 5 px}

## Results



## Discussion 
