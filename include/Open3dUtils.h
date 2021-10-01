//
// Created by jbs on 21. 9. 6..
//

#ifndef ZED_OPEN3D_OPEN3DUTILS_H
#define ZED_OPEN3D_OPEN3DUTILS_H

#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

namespace o3d_utils {
    void fromCvMat(const cv::Mat& cvImage, open3d::geometry::Image& o3dImage );
    void fromSlPoints(const sl::Mat& slPoints, open3d::geometry::PointCloud& o3dPoints );

}

#endif //ZED_OPEN3D_OPEN3DUTILS_H
