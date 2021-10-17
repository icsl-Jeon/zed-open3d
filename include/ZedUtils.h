#ifndef ZED_OPEN3D_ZEDUTILS_H
#define ZED_OPEN3D_ZEDUTILS_H

#include <sl/Camera.hpp>
#include <Open3dUtils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cvconfig.h>
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include "darknet/yolo_v2_class.hpp"


namespace zed_utils {
    using namespace sl;

    class Gaze{
    private:
        Eigen::Vector3f root;
        Eigen::Vector3f direction; // to be normalized
        Eigen::Matrix4f transformation; // z: normal outward from face, x: left eye to right eye (w.r.t actor)

    public:
        Gaze() = default;
        Gaze(const sl::ObjectData& humanObject);
        Eigen::Matrix4f getTransformation() const;
        bool isValid();

    };

    void parseArgs(int argc, char **argv, sl::InitParameters &param);

    bool initCamera(sl::Camera &zed, sl::InitParameters initParameters);

    cv::Mat slMat2cvMat(sl::Mat &input);

    cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat &input); // currently, not working

    int getOCVtype(sl::MAT_TYPE type);

    open3d::geometry::RGBDImage createFromCvImage(const cv::Mat &image, const cv::Mat &depth);




    
}

#endif //ZED_OPEN3D_ZEDUTILS_H
