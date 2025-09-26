#include "PanoramaStitcher.hpp"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <opencv2/stitching/detail/blenders.hpp>

using namespace std;

PanoramaStitcher::PanoramaStitcher(BlendingType blending_type)
    : blending_type_(blending_type) {
}

StitchingResult PanoramaStitcher::stitchImages(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography) {
    StitchingResult result;
    result.blending_type = blending_type_;
    result.success = false;

    if (img1.empty() || img2.empty() || homography.empty()) {
        cerr << "PanoramaStitcher: Invalid input images or homography" << endl;
        return result;
    }

    auto start_time = chrono::high_resolution_clock::now();

    try {
        switch (blending_type_) {
            case BlendingType::SIMPLE_OVERLAY:
                result.panorama = simpleOverlayBlend(img1, img2, homography);
                break;
            case BlendingType::FEATHERING:
                result.panorama = featheringBlend(img1, img2, homography);
                break;
            case BlendingType::MULTIBAND:
                result.panorama = multibandBlend(img1, img2, homography);
                break;
        }

        if (!result.panorama.empty()) {
            result.success = true;
            result.output_size = result.panorama.size();
        }
    } catch (const cv::Exception& e) {
        cerr << "PanoramaStitcher: OpenCV error - " << e.what() << endl;
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    result.stitching_time_ms = duration.count() / 1000.0;

    return result;
}

// Removed unused incremental helpers (stitchMultipleImages, calculateOutputSize)

cv::Mat PanoramaStitcher::simpleOverlayBlend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography) {
    vector<cv::Point2f> img2_corners = getImageCorners(img2);
    vector<cv::Point2f> img2_corners_transformed;
    cv::perspectiveTransform(img2_corners, img2_corners_transformed, homography);

    vector<cv::Point2f> img1_corners = getImageCorners(img1);
    vector<cv::Point2f> all_corners;
    all_corners.insert(all_corners.end(), img1_corners.begin(), img1_corners.end());
    all_corners.insert(all_corners.end(), img2_corners_transformed.begin(), img2_corners_transformed.end());

    cv::Rect bounding_rect = getBoundingRect(all_corners);

    cv::Mat translation = (cv::Mat_<double>(3, 3) <<
        1, 0, -bounding_rect.x,
        0, 1, -bounding_rect.y,
        0, 0, 1);

    cv::Mat adjusted_homography = translation * homography;

    // Warp both images to canvas
    cv::Mat img1_warped, img2_warped;
    cv::warpPerspective(img1, img1_warped, translation, bounding_rect.size());
    cv::warpPerspective(img2, img2_warped, adjusted_homography, bounding_rect.size());

    // Build masks
    cv::Mat ones1(img1.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat ones2(img2.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat mask1, mask2;
    cv::warpPerspective(ones1, mask1, translation, bounding_rect.size(), cv::INTER_NEAREST);
    cv::warpPerspective(ones2, mask2, adjusted_homography, bounding_rect.size(), cv::INTER_NEAREST);

    // Use OpenCV detail blender (NO = overlay)
    cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, false);
    blender->prepare(cv::Rect(0, 0, bounding_rect.width, bounding_rect.height));
    cv::Mat im1_16s, im2_16s;
    img1_warped.convertTo(im1_16s, CV_16S);
    img2_warped.convertTo(im2_16s, CV_16S);
    blender->feed(im1_16s, mask1, cv::Point(0, 0));
    blender->feed(im2_16s, mask2, cv::Point(0, 0));
    cv::Mat pano_16s, pano_mask;
    blender->blend(pano_16s, pano_mask);
    cv::Mat result;
    pano_16s.convertTo(result, CV_8U);
    return result;
}

cv::Mat PanoramaStitcher::featheringBlend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography) {
    vector<cv::Point2f> img2_corners = getImageCorners(img2);
    vector<cv::Point2f> img2_corners_transformed;
    cv::perspectiveTransform(img2_corners, img2_corners_transformed, homography);

    vector<cv::Point2f> img1_corners = getImageCorners(img1);
    vector<cv::Point2f> all_corners;
    all_corners.insert(all_corners.end(), img1_corners.begin(), img1_corners.end());
    all_corners.insert(all_corners.end(), img2_corners_transformed.begin(), img2_corners_transformed.end());

    cv::Rect bounding_rect = getBoundingRect(all_corners);

    cv::Mat translation = (cv::Mat_<double>(3, 3) <<
        1, 0, -bounding_rect.x,
        0, 1, -bounding_rect.y,
        0, 0, 1);

    cv::Mat adjusted_homography = translation * homography;

    // Warp both images to canvas
    cv::Mat img1_warped, img2_warped;
    cv::warpPerspective(img1, img1_warped, translation, bounding_rect.size());
    cv::warpPerspective(img2, img2_warped, adjusted_homography, bounding_rect.size());

    // Build masks
    cv::Mat ones1(img1.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat ones2(img2.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat mask1, mask2;
    cv::warpPerspective(ones1, mask1, translation, bounding_rect.size(), cv::INTER_NEAREST);
    cv::warpPerspective(ones2, mask2, adjusted_homography, bounding_rect.size(), cv::INTER_NEAREST);

    // Use OpenCV detail feather blender
    cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(cv::detail::Blender::FEATHER, false);
    // Optional: tune sharpness via dynamic cast to FeatherBlender
    if (auto fb = dynamic_cast<cv::detail::FeatherBlender*>(blender.get())) {
        fb->setSharpness(0.02f);
    }
    blender->prepare(cv::Rect(0, 0, bounding_rect.width, bounding_rect.height));
    cv::Mat im1_16s, im2_16s;
    img1_warped.convertTo(im1_16s, CV_16S);
    img2_warped.convertTo(im2_16s, CV_16S);
    blender->feed(im1_16s, mask1, cv::Point(0, 0));
    blender->feed(im2_16s, mask2, cv::Point(0, 0));
    cv::Mat pano_16s, pano_mask;
    blender->blend(pano_16s, pano_mask);
    cv::Mat result;
    pano_16s.convertTo(result, CV_8U);
    return result;
}

cv::Mat PanoramaStitcher::multibandBlend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography) {
    // Implement real multiband blending using detail::MultiBandBlender
    vector<cv::Point2f> img2_corners = getImageCorners(img2);
    vector<cv::Point2f> img2_corners_transformed;
    cv::perspectiveTransform(img2_corners, img2_corners_transformed, homography);

    vector<cv::Point2f> img1_corners = getImageCorners(img1);
    vector<cv::Point2f> all_corners;
    all_corners.insert(all_corners.end(), img1_corners.begin(), img1_corners.end());
    all_corners.insert(all_corners.end(), img2_corners_transformed.begin(), img2_corners_transformed.end());

    cv::Rect bounding_rect = getBoundingRect(all_corners);
    cv::Mat translation = (cv::Mat_<double>(3, 3) <<
        1, 0, -bounding_rect.x,
        0, 1, -bounding_rect.y,
        0, 0, 1);
    cv::Mat adjusted_homography = translation * homography;

    cv::Mat img1_warped, img2_warped;
    cv::warpPerspective(img1, img1_warped, translation, bounding_rect.size());
    cv::warpPerspective(img2, img2_warped, adjusted_homography, bounding_rect.size());

    cv::Mat ones1(img1.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat ones2(img2.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat mask1, mask2;
    cv::warpPerspective(ones1, mask1, translation, bounding_rect.size(), cv::INTER_NEAREST);
    cv::warpPerspective(ones2, mask2, adjusted_homography, bounding_rect.size(), cv::INTER_NEAREST);

    cv::detail::MultiBandBlender* mb = new cv::detail::MultiBandBlender(false, 5);
    cv::Ptr<cv::detail::Blender> blender(mb);
    blender->prepare(cv::Rect(0, 0, bounding_rect.width, bounding_rect.height));
    cv::Mat im1_16s, im2_16s;
    img1_warped.convertTo(im1_16s, CV_16S);
    img2_warped.convertTo(im2_16s, CV_16S);
    blender->feed(im1_16s, mask1, cv::Point(0, 0));
    blender->feed(im2_16s, mask2, cv::Point(0, 0));
    cv::Mat pano_16s, pano_mask;
    blender->blend(pano_16s, pano_mask);
    cv::Mat result;
    pano_16s.convertTo(result, CV_8U);
    return result;
}

vector<cv::Point2f> PanoramaStitcher::getImageCorners(const cv::Mat& image) {
    return {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(image.cols), 0),
        cv::Point2f(static_cast<float>(image.cols), static_cast<float>(image.rows)),
        cv::Point2f(0, static_cast<float>(image.rows))
    };
}

cv::Rect PanoramaStitcher::getBoundingRect(const vector<cv::Point2f>& points) {
    if (points.empty()) return cv::Rect();

    float min_x = points[0].x, max_x = points[0].x;
    float min_y = points[0].y, max_y = points[0].y;

    for (const auto& point : points) {
        min_x = min(min_x, point.x);
        max_x = max(max_x, point.x);
        min_y = min(min_y, point.y);
        max_y = max(max_y, point.y);
    }

    return cv::Rect(
        static_cast<int>(floor(min_x)),
        static_cast<int>(floor(min_y)),
        static_cast<int>(ceil(max_x - min_x)),
        static_cast<int>(ceil(max_y - min_y))
    );
}

string PanoramaStitcher::blendingTypeToString(BlendingType type) {
    switch (type) {
        case BlendingType::SIMPLE_OVERLAY: return "SimpleOverlay";
        case BlendingType::FEATHERING: return "Feathering";
        case BlendingType::MULTIBAND: return "Multiband";
        default: return "Unknown";
    }
}

std::vector<cv::Point2f> PanoramaStitcher::getTransformedCorners(const cv::Mat& image, const cv::Mat& H) {
    std::vector<cv::Point2f> corners = getImageCorners(image);
    std::vector<cv::Point2f> out;
    cv::perspectiveTransform(corners, out, H);
    return out;
}

cv::Rect PanoramaStitcher::computeGlobalBoundingRect(const std::vector<cv::Mat>& images,
                                                     const std::vector<cv::Mat>& transforms) {
    std::vector<cv::Point2f> all;
    for (size_t i = 0; i < images.size(); ++i) {
        if (transforms[i].empty()) continue;
        auto tc = getTransformedCorners(images[i], transforms[i]);
        all.insert(all.end(), tc.begin(), tc.end());
    }
    return getBoundingRect(all);
}

StitchingResult PanoramaStitcher::stitchImagesGlobal(const std::vector<cv::Mat>& images,
                                                     const std::vector<cv::Mat>& transforms_to_ref) {
    StitchingResult result; result.success = false; result.blending_type = blending_type_;
    if (images.empty() || images.size() != transforms_to_ref.size()) return result;

    // Compute canvas size
    cv::Rect bbox = computeGlobalBoundingRect(images, transforms_to_ref);
    if (bbox.width <= 0 || bbox.height <= 0) return result;

    cv::Mat T = (cv::Mat_<double>(3,3) << 1,0,-bbox.x, 0,1,-bbox.y, 0,0,1);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create appropriate blender
    cv::Ptr<cv::detail::Blender> blender;
    if (blending_type_ == BlendingType::SIMPLE_OVERLAY) {
        blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, false);
    } else if (blending_type_ == BlendingType::FEATHERING) {
        blender = cv::detail::Blender::createDefault(cv::detail::Blender::FEATHER, false);
        if (auto fb = dynamic_cast<cv::detail::FeatherBlender*>(blender.get())) fb->setSharpness(0.02f);
    } else {
        blender = cv::Ptr<cv::detail::Blender>(new cv::detail::MultiBandBlender(false, 5));
    }
    blender->prepare(cv::Rect(0, 0, bbox.width, bbox.height));

    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].empty() || transforms_to_ref[i].empty()) continue;
        cv::Mat H = T * transforms_to_ref[i];
        cv::Mat warped;
        cv::warpPerspective(images[i], warped, H, bbox.size());
        cv::Mat mask_src(images[i].size(), CV_8UC1, cv::Scalar(255));
        cv::Mat mask;
        cv::warpPerspective(mask_src, mask, H, bbox.size(), cv::INTER_NEAREST);
        cv::Mat warped_16s; warped.convertTo(warped_16s, CV_16S);
        blender->feed(warped_16s, mask, cv::Point(0,0));
    }

    cv::Mat pano_16s, pano_mask;
    blender->blend(pano_16s, pano_mask);
    cv::Mat pano;
    pano_16s.convertTo(pano, CV_8U);
    result.panorama = pano;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.stitching_time_ms = duration.count() / 1000.0;
    result.output_size = result.panorama.size();
    result.success = !result.panorama.empty();
    return result;
}
