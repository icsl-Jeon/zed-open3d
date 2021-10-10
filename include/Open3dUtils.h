//
// Created by jbs on 21. 9. 6..
//

#ifndef ZED_OPEN3D_OPEN3DUTILS_H
#define ZED_OPEN3D_OPEN3DUTILS_H

#include <open3d/Open3D.h>
#include <open3d/core/MemoryManager.h>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
using namespace std;
using namespace sl;
namespace o3d_utils {

    const std::vector<std::pair< BODY_PARTS, BODY_PARTS>> SKELETON_BONES
            {
                    {
                            BODY_PARTS::NOSE, BODY_PARTS::NECK
                    },
                    {
                            BODY_PARTS::NECK, BODY_PARTS::RIGHT_SHOULDER
                    },
                    {
                            BODY_PARTS::RIGHT_SHOULDER, BODY_PARTS::RIGHT_ELBOW
                    },
                    {
                            BODY_PARTS::RIGHT_ELBOW, BODY_PARTS::RIGHT_WRIST
                    },
                    {
                            BODY_PARTS::NECK, BODY_PARTS::LEFT_SHOULDER
                    },
                    {
                            BODY_PARTS::LEFT_SHOULDER, BODY_PARTS::LEFT_ELBOW
                    },
                    {
                            BODY_PARTS::LEFT_ELBOW, BODY_PARTS::LEFT_WRIST
                    },
                    {
                            BODY_PARTS::RIGHT_HIP, BODY_PARTS::RIGHT_KNEE
                    },
                    {
                            BODY_PARTS::RIGHT_KNEE, BODY_PARTS::RIGHT_ANKLE
                    },
                    {
                            BODY_PARTS::LEFT_HIP, BODY_PARTS::LEFT_KNEE
                    },
                    {
                            BODY_PARTS::LEFT_KNEE, BODY_PARTS::LEFT_ANKLE
                    },
                    {
                            BODY_PARTS::RIGHT_SHOULDER, BODY_PARTS::LEFT_SHOULDER
                    },
                    {
                            BODY_PARTS::RIGHT_HIP, BODY_PARTS::LEFT_HIP
                    },
                    {
                            BODY_PARTS::NOSE, BODY_PARTS::RIGHT_EYE
                    },
                    {
                            BODY_PARTS::RIGHT_EYE, BODY_PARTS::RIGHT_EAR
                    },
                    {
                            BODY_PARTS::NOSE, BODY_PARTS::LEFT_EYE
                    },
                    {
                            BODY_PARTS::LEFT_EYE, BODY_PARTS::LEFT_EAR
                    }
            };


    void fromCvMat(const cv::Mat& cvImage, open3d::geometry::Image& o3dImage );
    void fromSlPoints(const sl::Mat& slPoints, open3d::geometry::PointCloud& o3dPoints );
    void fromSlObjects(const sl::ObjectData& object,
                       std::shared_ptr<open3d::geometry::LineSet> lineSet,
                       std::shared_ptr<open3d::geometry::TriangleMesh> attentionPointSet[4]
                       );
    void initViewport(open3d::visualization::Visualizer& vis); // todo parameterized
    void registerGeometrySet(open3d::visualization::Visualizer& vis,
                             const vector<shared_ptr<open3d::geometry::Geometry3D>>& geometryPtrSet) ;

}

#endif //ZED_OPEN3D_OPEN3DUTILS_H
