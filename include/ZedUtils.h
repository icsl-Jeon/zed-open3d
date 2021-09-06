#ifndef ZED_OPEN3D_ZEDUTILS_H
#define ZED_OPEN3D_ZEDUTILS_H

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

namespace zed_utils {
    void parseArgs(int argc, char **argv, sl::InitParameters &param);
    bool initCamera(sl::Camera& zed, sl::InitParameters initParameters);




}

#endif //ZED_OPEN3D_ZEDUTILS_H
