#include "FeatureDetector.hpp"
#include "FeatureMatcher.hpp"
#include "HomographyEstimator.hpp"
#include "PanoramaStitcher.hpp"
#include "Evaluator.hpp"
#include "ImageWarper.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>

using namespace std;

// Find project root by walking up from current working directory
static std::filesystem::path findProjectRoot() {
    std::filesystem::path start = std::filesystem::current_path();
    std::filesystem::path p = start;
    while (true) {
        if (std::filesystem::exists(p / "CMakeLists.txt") && std::filesystem::exists(p / "src")) {
            return p;
        }
        if (!p.has_parent_path() || p.parent_path() == p) {
            break;
        }
        p = p.parent_path();
    }
    return start; // fallback
}

static std::string resolveBaseOutputDir(const std::string& user_output_dir) {
    std::filesystem::path base = user_output_dir.empty() ? std::filesystem::path("results")
                                                         : std::filesystem::path(user_output_dir);
    if (base.is_absolute()) {
        return base.string();
    }
    auto root = findProjectRoot();
    return (root / base).string();
}

static void writeParametersReference(const std::string& dir) {
    std::filesystem::create_directories(dir);
    std::ofstream f(dir + "/PARAMETERS.md");
    if (!f.is_open()) return;
    f << "# Parameters Reference\n\n";
    f << "- `--experiment`: Run experiment sweep over detector/matcher/blending/RANSAC thresholds.\n";
    f << "- `--detector <SIFT|ORB|AKAZE>`: Feature detector/descriptor. Affects keypoint count and descriptor type.\n";
    f << "- `--matcher <BF|ANN>`: Descriptor matcher (ANN = approximate nearest neighbor).\n";
    f << "  - Float descriptors (e.g., SIFT): ANN uses FLANN KD-tree.\n";
    f << "  - Binary descriptors (e.g., ORB/AKAZE): ANN uses FLANN LSH; BF uses Hamming.\n";
    f << "- `--blending <OVERLAY|FEATHER|MULTIBAND>`: Panorama blending backend (OpenCV detail blenders).\n";
    f << "  - OVERLAY: simple overwrite; FEATHER: soft seam via weights; MULTIBAND: pyramid blending for complex seams.\n";
    f << "- `--threshold <float>`: RANSAC reprojection threshold (pixels). Lower is stricter (fewer, cleaner inliers).\n";
    f << "- `--warper <NONE|CYLINDRICAL|SPHERICAL>`: Optional pre‑warping for rotation panoramas.\n";
    f << "- `--focal <float>`: Focal length in pixels for warper. Controls projection scale/curvature.\n";
    f << "- `--output <dir>`: Base output directory (anchored to project root if relative).\n";
    f << "\nInternal (not CLI by default):\n";
    f << "- Lowe's ratio threshold (default 0.80): filters ambiguous matches (FeatureMatcher).\n";
    f << "- Homography RANSAC confidence/max iterations: see HomographyEstimator defaults.\n";
    f.close();
}

static void writeSimpleRunConfig(const std::string& dir,
                                 DetectorType det,
                                 MatcherType mat,
                                 BlendingType blend,
                                 WarperType warper,
                                 double focal,
                                 double thr,
                                 const std::vector<std::string>& images) {
    std::filesystem::create_directories(dir);
    std::ofstream f(dir + "/RUN_CONFIG.txt");
    if (!f.is_open()) return;
    f << "detector=" << FeatureDetector::detectorTypeToString(det) << "\n";
    f << "matcher=" << FeatureMatcher::matcherTypeToString(mat) << "\n";
    f << "blending=" << PanoramaStitcher::blendingTypeToString(blend) << "\n";
    f << "warper=" << ImageWarper::warperTypeToString(warper) << "\n";
    f << "focal=" << focal << "\n";
    f << "ransac_threshold=" << thr << "\n";
    f << "images=";
    for (size_t i = 0; i < images.size(); ++i) {
        if (i) f << ",";
        f << images[i];
    }
    f << "\n";
    f.close();
    writeParametersReference(dir);
}

static void writeExperimentRunConfig(const std::string& dir,
                                     const ExperimentConfig& cfg,
                                     const std::vector<std::string>& image_paths) {
    std::filesystem::create_directories(dir);
    std::ofstream f(dir + "/RUN_CONFIG.txt");
    if (!f.is_open()) return;
    auto join = [](const std::vector<std::string>& v) {
        std::ostringstream os; for (size_t i=0;i<v.size();++i){ if(i) os<<","; os<<v[i]; } return os.str(); };
    std::vector<std::string> dets; for (auto d: cfg.detector_types) dets.push_back(FeatureDetector::detectorTypeToString(d));
    std::vector<std::string> mats; for (auto m: cfg.matcher_types) mats.push_back(FeatureMatcher::matcherTypeToString(m));
    std::vector<std::string> blends; for (auto b: cfg.blending_types) blends.push_back(PanoramaStitcher::blendingTypeToString(b));

    f << "detectors=" << join(dets) << "\n";
    f << "matchers=" << join(mats) << "\n";
    f << "blendings=" << join(blends) << "\n";
    f << "ransac_thresholds="; for (size_t i=0;i<cfg.reprojection_thresholds.size();++i){ if(i) f<<","; f<<cfg.reprojection_thresholds[i]; } f << "\n";
    f << "save_intermediate_results=" << (cfg.save_intermediate_results?"true":"false") << "\n";
    f << "save_visualizations=" << (cfg.save_visualizations?"true":"false") << "\n";
    f << "images="; for (size_t i=0;i<image_paths.size();++i){ if(i) f<<","; f<<image_paths[i]; } f << "\n";
    f.close();
    writeParametersReference(dir);
}

static bool runOpenCVStitcherBaseline(const std::vector<cv::Mat>& images, cv::Mat& pano) {
    // Try default PANORAMA
    {
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
        auto status = stitcher->stitch(images, pano);
        if (status == cv::Stitcher::OK && !pano.empty()) return true;
    }
    // Lower confidence threshold
    {
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
        stitcher->setPanoConfidenceThresh(0.3);
        auto status = stitcher->stitch(images, pano);
        if (status == cv::Stitcher::OK && !pano.empty()) return true;
    }
    // SCANS mode (planar/weak perspective scenes)
    {
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
        auto status = stitcher->stitch(images, pano);
        if (status == cv::Stitcher::OK && !pano.empty()) return true;
    }
    {
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
        stitcher->setPanoConfidenceThresh(0.3);
        auto status = stitcher->stitch(images, pano);
        if (status == cv::Stitcher::OK && !pano.empty()) return true;
    }
    return false;
}

void printUsage(const char* program_name) {
    cout << "Usage: " << program_name << " [options] img1 img2 [img3 ...]" << endl;
    cout << "Options:" << endl;
    cout << "  --experiment     Run full experimental evaluation (simple mode disabled)" << endl;
    cout << "  --detector <type>   Detector type: SIFT, ORB, AKAZE (default: SIFT)" << endl;
    cout << "  --matcher <type>    Matcher type: BF, ANN (default: BF)" << endl;
    cout << "  --blending <type>   Blending type: OVERLAY, FEATHER, MULTIBAND (default: FEATHER)" << endl;
    cout << "  --warper <type>     Warper: NONE, CYLINDRICAL, SPHERICAL (default: NONE)" << endl;
    cout << "  --focal <val>       Warper focal length in pixels (default: 0.5*width)" << endl;
    cout << "  --threshold <val>   RANSAC threshold (default: 3.0)" << endl;
    cout << "  --output <dir>      Output directory (default: results)" << endl;
    cout << "  --help              Show this help message" << endl;
}

DetectorType parseDetectorType(const string& str) {
    if (str == "SIFT") return DetectorType::SIFT;
    if (str == "ORB") return DetectorType::ORB;
    if (str == "AKAZE") return DetectorType::AKAZE;
    cout << "Warning: Unknown detector type '" << str << "', using SIFT" << endl;
    return DetectorType::SIFT;
}

MatcherType parseMatcherType(const string& str) {
    if (str == "BF") return MatcherType::BRUTE_FORCE;
    if (str == "FLANN" || str == "ANN") return MatcherType::FLANN; // ANN alias for FLANN backend
    cout << "Warning: Unknown matcher type '" << str << "', using BF" << endl;
    return MatcherType::BRUTE_FORCE;
}

BlendingType parseBlendingType(const string& str) {
    if (str == "OVERLAY") return BlendingType::SIMPLE_OVERLAY;
    if (str == "FEATHER") return BlendingType::FEATHERING;
    if (str == "MULTIBAND") return BlendingType::MULTIBAND;
    cout << "Warning: Unknown blending type '" << str << "', using FEATHER" << endl;
    return BlendingType::FEATHERING;
}

WarperType parseWarperType(const string& str) {
    if (str == "NONE") return WarperType::NONE;
    if (str == "CYLINDRICAL") return WarperType::CYLINDRICAL;
    if (str == "SPHERICAL") return WarperType::SPHERICAL;
    cout << "Warning: Unknown warper type '" << str << "', using NONE" << endl;
    return WarperType::NONE;
}

void runExperimentMode(const vector<string>& image_paths, const string& output_dir) {
    cout << "=== Running Comprehensive Experimental Evaluation ===" << endl;

    // Prepare timestamped run directory under experiments/
    std::string base_dir = resolveBaseOutputDir(output_dir);
    std::time_t t = std::time(nullptr);
    std::tm tm_local;
#if defined(_WIN32)
    localtime_s(&tm_local, &t);
#else
    tm_local = *std::localtime(&t);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_local, "%Y%m%d-%H%M%S");
    std::string run_dir = base_dir + "/experiments/" + oss.str();
    std::filesystem::create_directories(run_dir);

    ExperimentConfig config;
    config.detector_types = {DetectorType::SIFT, DetectorType::ORB, DetectorType::AKAZE};
    // Focus experiments on detector, RANSAC threshold, and blending; use BF matcher only
    config.matcher_types = {MatcherType::BRUTE_FORCE};
    config.blending_types = {BlendingType::SIMPLE_OVERLAY, BlendingType::FEATHERING};
    config.reprojection_thresholds = {0.5, 1.0, 2.0, 3.0, 5.0};
    config.output_dir = run_dir; // ensure all experiment artifacts go under experiments/timestamp
    config.save_intermediate_results = true;
    config.save_visualizations = true;

    Evaluator evaluator(config);

    for (size_t i = 0; i < image_paths.size() - 1; i++) {
        cv::Mat img1 = cv::imread(image_paths[i]);
        cv::Mat img2 = cv::imread(image_paths[i + 1]);

        if (img1.empty() || img2.empty()) {
            cerr << "Error: Cannot load image pair " << i << " and " << i + 1 << endl;
            continue;
        }

        // Use image file stems as experiment name, e.g., imgA_imgB
        std::string stem1 = std::filesystem::path(image_paths[i]).stem().string();
        std::string stem2 = std::filesystem::path(image_paths[i + 1]).stem().string();
        // Sanitize: replace spaces with dashes (basic)
        std::replace(stem1.begin(), stem1.end(), ' ', '-');
        std::replace(stem2.begin(), stem2.end(), ' ', '-');
        string experiment_name = stem1 + "_" + stem2;
        evaluator.runExperiments(img1, img2, experiment_name);

        // Baseline outputs (my default pipeline vs OpenCV Stitcher) per pair
        std::string pair_dir = run_dir + "/" + experiment_name;
        std::filesystem::create_directories(pair_dir);

        // My default pipeline (SIFT + BF + FEATHER, threshold=3.0)
        try {
            FeatureDetector detector(DetectorType::SIFT);
            FeatureMatcher matcher(MatcherType::BRUTE_FORCE);
            HomographyEstimator estimator(3.0);
            PanoramaStitcher stitcher(BlendingType::FEATHERING);

            auto det1 = detector.detectAndCompute(img1);
            auto det2 = detector.detectAndCompute(img2);
            auto match = matcher.matchFeatures(det1.descriptors, det2.descriptors);
            cv::Mat my_panorama;
            if (match.good_matches_count >= 10) {
                auto Hres = estimator.estimateHomographyFromMatches(det1.keypoints, det2.keypoints, match.good_matches);
                if (Hres.valid) {
                    cv::Mat H21 = Hres.homography.inv();
                    auto sres = stitcher.stitchImages(img1, img2, H21);
                        if (sres.success) {
                            my_panorama = sres.panorama;
                        }
                    }
                }
            if (!my_panorama.empty()) {
                cv::imwrite(pair_dir + "/my_panorama.jpg", my_panorama);
            }
        } catch (const std::exception& e) {
            cerr << "My baseline (pair) exception: " << e.what() << endl;
        }

        // OpenCV Stitcher baseline (with fallbacks)
        try {
            cv::Mat cv_panorama;
            bool ok = runOpenCVStitcherBaseline(std::vector<cv::Mat>{img1, img2}, cv_panorama);
            if (ok) {
                cv::imwrite(pair_dir + "/opencv_panorama.jpg", cv_panorama);
            } else {
                cerr << "OpenCV Stitcher failed for " << experiment_name << " (after fallbacks)" << endl;
            }
        } catch (const std::exception& e) {
            cerr << "OpenCV Stitcher (pair) exception: " << e.what() << endl;
        }
    }

    writeExperimentRunConfig(run_dir, config, image_paths);
    evaluator.saveResults(run_dir + std::string("/experimental_results.csv"));
    evaluator.saveResultsAsTable(run_dir + std::string("/results_table.txt"));
    evaluator.generateReport(run_dir + std::string("/experiment_report.md"));

    cout << "=== Experimental evaluation completed ===" << endl;
    cout << "Results saved to: " << run_dir << endl;
}

void runSimpleMode(const vector<string>& image_paths, DetectorType detector_type,
                  MatcherType matcher_type, BlendingType blending_type,
                  double threshold, const string& output_dir,
                  WarperType warper_type = WarperType::NONE, double focal = 0.0) {
    if (image_paths.size() < 2) {
        cerr << "Error: Need at least 2 images for stitching" << endl;
        return;
    }

    cout << "=== Simple Panorama Stitching ===" << endl;
    cout << "Detector: " << FeatureDetector::detectorTypeToString(detector_type) << endl;
    cout << "Matcher: " << FeatureMatcher::matcherTypeToString(matcher_type) << endl;
    cout << "Blending: " << PanoramaStitcher::blendingTypeToString(blending_type) << endl;
    cout << "RANSAC threshold: " << threshold << endl;

    vector<cv::Mat> images;
    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            cerr << "Error: Cannot load image " << path << endl;
            return;
        }
        images.push_back(img);
    }

    // Optional pre-warp (cylindrical/spherical)
    vector<cv::Mat> proc_images = images;
    if (warper_type != WarperType::NONE) {
        ImageWarper warper(warper_type, focal);
        proc_images.clear();
        proc_images.reserve(images.size());
        for (const auto& im : images) proc_images.push_back(warper.warp(im));
    }

    FeatureDetector detector(detector_type);
    FeatureMatcher matcher(matcher_type);
    HomographyEstimator estimator(threshold);
    PanoramaStitcher stitcher(blending_type);

    // Compute features for all images
    struct Feat { vector<cv::KeyPoint> kps; cv::Mat desc; int count; };
    vector<Feat> feats(proc_images.size());
    for (size_t i = 0; i < proc_images.size(); ++i) {
        auto d = detector.detectAndCompute(proc_images[i]);
        feats[i] = {d.keypoints, d.descriptors, d.keypoint_count};
    }

    // Pairwise homographies between consecutive images (i -> i+1)
    vector<cv::Mat> H_pair(proc_images.size() - 1);
    for (size_t i = 0; i + 1 < proc_images.size(); ++i) {
        auto mr = matcher.matchFeatures(feats[i].desc, feats[i+1].desc);
        cout << "Processing pair " << i << "-" << i+1 << ": matches "
             << mr.good_matches_count << " / " << mr.total_matches << endl;
        if (mr.good_matches_count < 10) { H_pair[i] = cv::Mat(); continue; }
        auto Hres = estimator.estimateHomographyFromMatches(feats[i].kps, feats[i+1].kps, mr.good_matches);
        if (!Hres.valid) { H_pair[i] = cv::Mat(); continue; }
        H_pair[i] = Hres.homography; // maps i -> i+1
    }

    // Compose transforms to middle reference
    size_t n = proc_images.size();
    size_t ref = n / 2;
    vector<cv::Mat> T_to_ref(n);
    T_to_ref[ref] = cv::Mat::eye(3,3,CV_64F);
    // left side
    for (int k = static_cast<int>(ref) - 1; k >= 0; --k) {
        if (H_pair[k].empty() || T_to_ref[k+1].empty()) { T_to_ref[k] = cv::Mat(); continue; }
        T_to_ref[k] = T_to_ref[k+1] * H_pair[k];
    }
    // right side
    for (size_t k = ref + 1; k < n; ++k) {
        if (H_pair[k-1].empty() || T_to_ref[k-1].empty()) { T_to_ref[k] = cv::Mat(); continue; }
        cv::Mat Hinv = H_pair[k-1].inv(); // map k -> k-1
        T_to_ref[k] = T_to_ref[k-1] * Hinv;
    }

    // Global warp and single-pass blend on processed images
    auto global_res = stitcher.stitchImagesGlobal(proc_images, T_to_ref);
    if (global_res.success) {
        // Save outputs (and baseline below)
        // Create timestamped subdirectory under output_dir/simple
        std::time_t t = std::time(nullptr);
        std::tm tm_local;
    #if defined(_WIN32)
        localtime_s(&tm_local, &t);
    #else
        tm_local = *std::localtime(&t);
    #endif
        std::ostringstream oss;
        oss << std::put_time(&tm_local, "%Y%m%d-%H%M%S");
    std::string base_dir = resolveBaseOutputDir(output_dir);
    std::string run_dir = base_dir + "/simple/" + oss.str();
    filesystem::create_directories(run_dir);
    writeSimpleRunConfig(run_dir, detector_type, matcher_type, blending_type, warper_type, focal, threshold, image_paths);
        // Always save the user-selected blend as my_panorama.jpg
        string my_output_path = run_dir + "/my_panorama.jpg";
        cv::imwrite(my_output_path, global_res.panorama);
        cout << "My panorama saved to: " << my_output_path << endl;

        // If more than 2 images, also output all three blending variants for side-by-side comparison
        if (proc_images.size() > 2) {
            std::vector<std::pair<BlendingType, std::string>> blends = {
                {BlendingType::SIMPLE_OVERLAY, "overlay"},
                {BlendingType::FEATHERING, "feather"},
                {BlendingType::MULTIBAND, "multiband"}
            };
            for (const auto& b : blends) {
                PanoramaStitcher comp_stitcher(b.first);
                auto res = comp_stitcher.stitchImagesGlobal(proc_images, T_to_ref);
                if (res.success) {
                    std::string outp = run_dir + "/my_panorama_" + b.second + ".jpg";
                    cv::imwrite(outp, res.panorama);
                }
            }
        }
        // Compute OpenCV baseline (uses original images)
        cv::Mat cv_panorama;
        try {
            bool ok = runOpenCVStitcherBaseline(images, cv_panorama);
            if (ok) {
                string cv_output_path = run_dir + "/opencv_panorama.jpg";
                cv::imwrite(cv_output_path, cv_panorama);
                cout << "OpenCV baseline saved to: " << cv_output_path << endl;
            } else {
                cerr << "OpenCV Stitcher failed (after fallbacks)." << endl;
            }
        } catch (const std::exception& e) {
            cerr << "OpenCV Stitcher exception: " << e.what() << endl;
        }

        cv::imshow("Final Panorama (My Pipeline)", global_res.panorama);
        cout << "Press any key to exit..." << endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
        return;
    } else {
        cerr << "Global stitching failed." << endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return -1;
    }

    bool experiment_mode = false;
    DetectorType detector_type = DetectorType::SIFT;
    MatcherType matcher_type = MatcherType::BRUTE_FORCE;
    BlendingType blending_type = BlendingType::FEATHERING;
    WarperType warper_type = WarperType::NONE;
    double threshold = 3.0;
    string output_dir = "results";
    double focal = 0.0;
    vector<string> image_paths;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--experiment") {
            experiment_mode = true;
        } else if (arg == "--detector" && i + 1 < argc) {
            detector_type = parseDetectorType(argv[++i]);
        } else if (arg == "--matcher" && i + 1 < argc) {
            matcher_type = parseMatcherType(argv[++i]);
        } else if (arg == "--blending" && i + 1 < argc) {
            blending_type = parseBlendingType(argv[++i]);
        } else if (arg == "--warper" && i + 1 < argc) {
            warper_type = parseWarperType(argv[++i]);
        } else if (arg == "--focal" && i + 1 < argc) {
            focal = atof(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg[0] != '-') {
            image_paths.push_back(arg);
        }
    }

    if (image_paths.size() < 2) {
        cerr << "Error: Need at least 2 images" << endl;
        printUsage(argv[0]);
        return -1;
    }

    try {
        // Only experiment mode is supported now. If --experiment is not set, still run experiments.
        if (!experiment_mode) {
            cout << "[Info] Simple mode is disabled. Running experiment mode instead." << endl;
        }
        runExperimentMode(image_paths, output_dir);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
