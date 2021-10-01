#ifndef ZED_OPEN3D_ZEDUTILS_H
#define ZED_OPEN3D_ZEDUTILS_H

#include <sl/Camera.hpp>
#include <Open3dUtils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cvconfig.h>

namespace zed_utils {
    void parseArgs(int argc, char **argv, sl::InitParameters &param);

    bool initCamera(sl::Camera &zed, sl::InitParameters initParameters);

    cv::Mat slMat2cvMat(sl::Mat &input);

    cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat &input);

    int getOCVtype(sl::MAT_TYPE type);

    open3d::geometry::RGBDImage createFromCvImage(const cv::Mat &image, const cv::Mat &depth);
}

#endif //ZED_OPEN3D_ZEDUTILS_H
