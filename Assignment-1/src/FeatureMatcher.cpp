#include "FeatureMatcher.hpp"
#include <algorithm>
#include <opencv2/flann.hpp>
#include <chrono>

using namespace std;

FeatureMatcher::FeatureMatcher(MatcherType type, double ratio_threshold)
    : matcher_type_(type), ratio_threshold_(ratio_threshold) {
}

void FeatureMatcher::initializeMatcher(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    switch (matcher_type_) {
        case MatcherType::BRUTE_FORCE: {
            if (descriptors1.type() == CV_8U) {
                matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
            } else {
                matcher_ = cv::BFMatcher::create(cv::NORM_L2);
            }
            break;
        }
        case MatcherType::FLANN: {
            // ANN backend: for float descriptors use FLANN KD-tree; for binary use FLANN LSH
            bool is_binary = (descriptors1.type() == CV_8U) || (descriptors2.type() == CV_8U);
            if (is_binary) {
                // LSH index parameters: (table_number, key_size, multi_probe_level)
                auto indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
                auto searchParams = cv::makePtr<cv::flann::SearchParams>(50);
                matcher_ = cv::Ptr<cv::DescriptorMatcher>(new cv::FlannBasedMatcher(indexParams, searchParams));
            } else {
                matcher_ = cv::FlannBasedMatcher::create();
            }
            break;
        }
    }
}

MatchingResult FeatureMatcher::matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    MatchingResult result;
    result.matcher_type = matcher_type_;

    // Guard: empty descriptors lead to crashes/exceptions in matchers
    if (descriptors1.empty() || descriptors2.empty()) {
        result.matches.clear();
        result.good_matches.clear();
        result.matching_time_ms = 0.0;
        result.total_matches = 0;
        result.good_matches_count = 0;
        result.match_ratio = 0.0;
        return result;
    }

    initializeMatcher(descriptors1, descriptors2);

    auto start_time = chrono::high_resolution_clock::now();

    if (ratio_threshold_ > 0.0) {
        vector<vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(descriptors1, descriptors2, knn_matches, 2);
        result.good_matches = filterMatchesWithRatio(knn_matches);
        // Preserve the total number of candidate matches for metrics
        result.total_matches = static_cast<int>(knn_matches.size());
        result.matches = result.good_matches;
    } else {
        matcher_->match(descriptors1, descriptors2, result.matches);
        result.good_matches = filterMatchesWithDistance(result.matches);
        result.total_matches = static_cast<int>(result.matches.size());
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    result.matching_time_ms = duration.count() / 1000.0;

    result.good_matches_count = static_cast<int>(result.good_matches.size());
    result.match_ratio = result.total_matches > 0 ?
        static_cast<double>(result.good_matches_count) / result.total_matches : 0.0;

    result.match_distances.clear();
    for (const auto& match : result.good_matches) {
        result.match_distances.push_back(match.distance);
    }

    return result;
}

vector<cv::DMatch> FeatureMatcher::filterMatchesWithRatio(const vector<vector<cv::DMatch>>& knn_matches) {
    vector<cv::DMatch> good_matches;

    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() == 2) {
            if (match_pair[0].distance < ratio_threshold_ * match_pair[1].distance) {
                good_matches.push_back(match_pair[0]);
            }
        }
    }

    return good_matches;
}

vector<cv::DMatch> FeatureMatcher::filterMatchesWithDistance(const vector<cv::DMatch>& matches) {
    if (matches.empty()) return matches;

    double min_dist = min_element(matches.begin(), matches.end(),
        [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        })->distance;

    vector<cv::DMatch> good_matches;
    double threshold = max(2.0 * min_dist, 30.0);

    for (const auto& match : matches) {
        if (match.distance <= threshold) {
            good_matches.push_back(match);
        }
    }

    return good_matches;
}

string FeatureMatcher::matcherTypeToString(MatcherType type) {
    switch (type) {
        case MatcherType::BRUTE_FORCE: return "BruteForce";
        case MatcherType::FLANN: return "ANN";
        default: return "Unknown";
    }
}
