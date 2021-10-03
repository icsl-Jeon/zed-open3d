#include <ZedUtils.h>
#include <Misc.h>

#include <csignal>
volatile sig_atomic_t stop;
using namespace std;
/**
 * This will draw skeleton from zed object detection, gaze volumne, hand sphere
 */

int main(int argc, char** argv) {

    // ZED initialization
    sl::Camera zed;
    sl::InitParameters initParameters;

    zed_utils::parseArgs(argc, argv, initParameters);
    zed_utils::initCamera(zed, initParameters);

    sl::RuntimeParameters runParameters;
    runParameters.confidence_threshold = 10;
    runParameters.texture_confidence_threshold = 100;
    sl::ObjectDetectionRuntimeParameters objRunParameters;
    objRunParameters.detection_confidence_threshold = 40;

    sl::Objects humanObjects;
    sl::Mat points;

    // Open3d
    bool isInit = false;
    auto pointsO3dPtr = std::make_shared<open3d::geometry::PointCloud>(); // pcl
    auto skeletonO3dPtr = std::make_shared<open3d::geometry::LineSet>(); // object skeleton
    vector<shared_ptr<open3d::geometry::Geometry3D>> geometryPtrSet;
    geometryPtrSet.push_back(pointsO3dPtr);
    geometryPtrSet.push_back(skeletonO3dPtr);

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Attention",720,404);
    auto& ctl = vis.GetViewControl(); // note referance specifier
    auto& rtl = vis.GetRenderOption();

    while (!stop){
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS) {
            // retrieve zed information
            zed.retrieveMeasure(points, sl::MEASURE::XYZRGBA, sl::MEM::CPU); // MEM GPU is faster than CPU 2 times.
            zed.retrieveObjects(humanObjects);

            // prepare geometry in open3d
            o3d_utils::fromSlPoints(points, *pointsO3dPtr);
            if (not humanObjects.object_list.empty())
                o3d_utils::fromSlObjects(humanObjects.object_list[0],*skeletonO3dPtr);

            if (not isInit){
                o3d_utils::registerGeometrySet(vis,geometryPtrSet);
                o3d_utils::initViewport(vis);
                isInit = true;
            }

            // visualizer
            vis.UpdateGeometry();
            vis.PollEvents();
            vis.UpdateGeometry();

        }else if (zed.grab(runParameters) == sl::ERROR_CODE::END_OF_SVOFILE_REACHED){
            printf("SVO reached end. Replay. \n");
            zed.setSVOPosition(1);
        }
        else
            printf("Grab failed. \n");
    }





}

