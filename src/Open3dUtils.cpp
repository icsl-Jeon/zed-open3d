//
// Created by jbs on 21. 9. 6..
//
#include <Open3dUtils.h>


void o3d_utils::fromCvMat(const cv::Mat& cvImage, open3d::geometry::Image& o3dImage ){
    assert((o3dImage.width_ == cvImage.cols) and
                   (o3dImage.height_== cvImage.rows));
    // Currently only 8U_C3 implemented
    int channels = cvImage.channels();
    int bytesPerChannel = cvImage.depth()/2 + 1;
    memcpy(o3dImage.data_.data(),
           cvImage.data, cvImage.total() * channels * bytesPerChannel);
}

void o3d_utils::fromSlPoints(const sl::Mat &slPoints, open3d::geometry::PointCloud &o3dPoints) {
    // Simple way
    o3dPoints.points_.clear();
    o3dPoints.colors_.clear();
    int ptsCount = slPoints.getHeight() * slPoints.getWidth();
    auto cloudPtr = slPoints.getPtr<sl::float4>();
    for (int cnt = 0 ; cnt < ptsCount ; cnt+=2){
        sl::Vector4<float>* subPtr =  &cloudPtr[cnt];
        if (subPtr->x == subPtr->x and (not isinf(subPtr->x))){
            o3dPoints.points_.emplace_back(Eigen::Vector3d(subPtr->x, subPtr->y, subPtr->z));
            auto colorPtr = (uchar *)&subPtr->w;
            float r = float(colorPtr[0])/255.0;
            float g = float(colorPtr[1])/255.0;
            float b = float(colorPtr[2])/255.0;
            o3dPoints.colors_.emplace_back(Eigen::Vector3d(r,g,b));

        }
    }

    std::cout << "points size = " << o3dPoints.points_.size() << std::endl;
}




