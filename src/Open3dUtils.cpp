//
// Created by jbs on 21. 9. 6..
//
#include <Open3dUtils.h>


void o3d_utils::fromCvMat(const cv::Mat& cvImage, open3d::geometry::Image& o3dImage ){
    assert((o3dImage.width_ == cvImage.cols) and
                   (o3dImage.height_== cvImage.rows));
    // Currently only 8U_C3 implemented
    int channels = cvImage.channels();
    int bytesPerChannel = cvImage.elemSize()/channels;
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

void o3d_utils::fromSlObjects(const sl::ObjectData &object,
                              std::shared_ptr<open3d::geometry::LineSet>  lineSet,
                              std::shared_ptr<open3d::geometry::TriangleMesh> attentionPointSet[4]
                              ) {

    lineSet->colors_.clear();
    lineSet->points_.clear();
    lineSet->lines_.clear();

    if (!object.keypoint.empty()) {

        int cntNanPoint = 0;
        int index = 0 ;
        for (const auto &pnt: object.keypoint) {
            auto keyPnt = Eigen::Vector3d(pnt.x, pnt.y, pnt.z);
            if ( not std::isfinite(keyPnt.norm())) {
                cntNanPoint++;
                // This zero-assigning is crucial!
                // open3d seems to use all the points
                // to decide the AABB
                // even though a points is not assigned to any line
                keyPnt = Eigen::Vector3d(0,0,0);
            }else{

                if (static_cast<BODY_PARTS>(index) == sl::BODY_PARTS::LEFT_EYE)
                    attentionPointSet[0]->Translate(keyPnt,false);

                if (static_cast<BODY_PARTS>(index) == sl::BODY_PARTS::RIGHT_EYE)
                    attentionPointSet[1]->Translate(keyPnt,false);

                if (static_cast<BODY_PARTS>(index) == sl::BODY_PARTS::LEFT_WRIST)
                    attentionPointSet[2]->Translate(keyPnt,false);

                if (static_cast<BODY_PARTS>(index) == sl::BODY_PARTS::RIGHT_WRIST)
                    attentionPointSet[3]->Translate(keyPnt,false);

            }
            lineSet->points_.emplace_back(keyPnt);
            index ++;
        }

        for (auto &limb : o3d_utils::SKELETON_BONES) {
            int idx1 = getIdx(limb.first);
            int idx2 = getIdx(limb.second);
            sl::float3 kp_1 = object.keypoint[idx1];
            sl::float3 kp_2 = object.keypoint[idx2];
            // draw line between two keypoints
            if (std::isfinite(kp_1.norm()) && std::isfinite(kp_2.norm()))
                lineSet->lines_.emplace_back(Eigen::Vector2i(idx1, idx2));
        }
    }

    /**
    printf("--------------------------------\n");
    for (auto line : lineSet.lines_) {
        int idx1 = line(0); auto pnt1 = lineSet.points_[idx1];
        int idx2 = line(1); auto pnt2 = lineSet.points_[idx2];
        printf("(%.3f, %.3f, %.3f) -  (%.3f, %.3f, %.3f) \n ",
               pnt1.x(), pnt1.y(), pnt1.z(),
               pnt2.x(), pnt2.y(), pnt2.z());
    }
    **/
    float r = 77.0 / 255.0;
    float g = 143.0 / 255.0;
    float b = 247.0 / 255.0;

    lineSet->PaintUniformColor(Eigen::Vector3d(r,g,b));
}


void o3d_utils::registerGeometrySet(open3d::visualization::Visualizer &vis,
                                    const vector<shared_ptr<open3d::geometry::Geometry3D>> &geometryPtrSet) {
    for (const auto& ptr: geometryPtrSet){
        vis.AddGeometry(ptr);
    }
}

void o3d_utils::initViewport(open3d::visualization::Visualizer& vis) {

    auto& ctl = vis.GetViewControl(); // note referance specifier
    auto& rtl = vis.GetRenderOption();
    vis.AddGeometry(
            open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.1)); // add coord at origin

    ctl.SetFront(Eigen::Vector3d(-1,0,0)); // seems that front is normal to the window ... (-x of world)
    ctl.SetUp(Eigen::Vector3d(0,0,1)); // direction of ceiling
    ctl.SetLookat(Eigen::Vector3d(0,0,0));
    ctl.SetZoom(0.1);
    rtl.SetPointSize(1);
    rtl.background_color_ = Eigen::Vector3d(0,0,0);


}






