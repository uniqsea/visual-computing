#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class VideoCapture {
public:
    VideoCapture();
    ~VideoCapture();
    
    bool open(int deviceID = 0);
    void close();
    
    bool isOpened() const;
    bool getFrame(cv::Mat& frame);
    
    // Resolution control
    bool setResolution(int width, int height);
    void getResolution(int& width, int& height) const;
    
    // Predefined resolutions
    bool setResolution640x480();
    bool setResolution1280x720();
    bool setResolution1920x1080();
    
    std::string getResolutionString() const;
    
private:
    cv::VideoCapture capture;
    int currentWidth;
    int currentHeight;
};

