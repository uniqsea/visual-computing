#include "ImageWarper.hpp"
#include <opencv2/stitching/warpers.hpp>

using namespace cv;

Mat ImageWarper::warp(const Mat& image) const {
    if (image.empty()) return Mat();
    switch (type_) {
        case WarperType::CYLINDRICAL:
            return warpCylindrical(image);
        case WarperType::SPHERICAL:
            return warpSpherical(image);
        case WarperType::NONE:
        default:
            return image.clone();
    }
}

std::string ImageWarper::warperTypeToString(WarperType t) {
    switch (t) {
        case WarperType::NONE: return "NONE";
        case WarperType::CYLINDRICAL: return "CYLINDRICAL";
        case WarperType::SPHERICAL: return "SPHERICAL";
        default: return "UNKNOWN";
    }
}

Mat ImageWarper::warpCylindrical(const Mat& image) const {
    const int w = image.cols;
    const int h = image.rows;
    const float f = static_cast<float>((focal_ > 0.0) ? focal_ : (0.5 * w));
    const float cx = (w - 1) * 0.5f;
    const float cy = (h - 1) * 0.5f;

    Mat K = (Mat_<float>(3,3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    Mat R = Mat::eye(3, 3, CV_32F);

    cv::detail::CylindricalWarper warper(f);
    Mat dst;
    warper.warp(image, K, R, INTER_LINEAR, BORDER_CONSTANT, dst);
    return dst;
}

Mat ImageWarper::warpSpherical(const Mat& image) const {
    const int w = image.cols;
    const int h = image.rows;
    const float f = static_cast<float>((focal_ > 0.0) ? focal_ : (0.5 * w));
    const float cx = (w - 1) * 0.5f;
    const float cy = (h - 1) * 0.5f;

    Mat K = (Mat_<float>(3,3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    Mat R = Mat::eye(3, 3, CV_32F);

    cv::detail::SphericalWarper warper(f);
    Mat dst;
    warper.warp(image, K, R, INTER_LINEAR, BORDER_CONSTANT, dst);
    return dst;
}
