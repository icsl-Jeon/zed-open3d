#include <ZedUtils.h>
#include <csignal>
#include <Misc.h>

volatile sig_atomic_t stop;


int main(int argc, char** argv) {

    // Initialize ZED camera
    sl::Camera zed;
    sl::InitParameters initParameters;
    initParameters.coordinate_units = sl::UNIT::METER;

    zed_utils::parseArgs(argc, argv, initParameters);
    zed_utils::initCamera(zed, initParameters);

    sl::RuntimeParameters runParameters;
    runParameters.confidence_threshold = 10;
    runParameters.texture_confidence_threshold = 100;

    // Initialize ZED Objects
    auto cameraConfig = zed.getCameraInformation().camera_configuration;
    sl::Mat image(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::Mat depth(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::F32_C1, sl::MEM::CPU);
    sl::Mat points;

    // buffer binding for cv objects
    cv::Mat imageCv = zed_utils::slMat2cvMat(image);
    cv::Mat imageCv3Ch;
    cv::Size displaySize(720, 404);
    cv::Mat imageDisplay(displaySize, CV_8UC4);

    // open3d object
    auto pointsO3dPtr = std::make_shared<open3d::geometry::PointCloud>();
    bool isInit = false;
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Open3D image",720,404);
    auto& ctl = vis.GetViewControl(); // note referance specifier
    auto& rtl = vis.GetRenderOption();
    sl::Objects objects;

    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){
            // 1. retrieve zed
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::CPU);
            zed.retrieveMeasure(points, sl::MEASURE::XYZRGBA, sl::MEM::CPU); // MEM GPU is faster than CPU 2 times.
            o3d_utils::fromSlPoints(points, *pointsO3dPtr);
            cv::cvtColor(imageCv, imageCv3Ch, cv::COLOR_BGRA2RGB);
            printf("Image + points in %.3f ms \n" ,timer.stop());

            // 2. draw in open3d
            if (not isInit) {
                vis.AddGeometry(pointsO3dPtr);
                vis.AddGeometry(
                        open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.1)); // add coord at origin
                isInit = true;
                ctl.SetFront(Eigen::Vector3d(-1,0,0)); // seems that front is normal to the window ... (-x of world)
                ctl.SetUp(Eigen::Vector3d(0,0,1)); // direction of ceiling
                ctl.SetLookat(Eigen::Vector3d(0,0,0));
                ctl.SetZoom(0.1);
                rtl.SetPointSize(1);
                rtl.background_color_ = Eigen::Vector3d(0,0,0);

            }

            Eigen::Vector3d min_bound = pointsO3dPtr->GetMinBound();
            Eigen::Vector3d max_bound = pointsO3dPtr->GetMaxBound();
            open3d::utility::LogInfo(
                    "Bounding box is: ({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, "
                    "{:.4f})",
                    min_bound(0), min_bound(1), min_bound(2), max_bound(0),
                    max_bound(1), max_bound(2));

            vis.UpdateGeometry();
            vis.PollEvents();
            vis.UpdateRender();
            std::cout << vis.GetViewControl().GetViewMatrix() << std::endl;

            // 3. opencv window
            cv::resize(imageCv, imageDisplay, displaySize);
            cv::imshow("Image", imageDisplay);
            cv::waitKey(1);

        // replay option
        }else if (zed.grab(runParameters) == sl::ERROR_CODE::END_OF_SVOFILE_REACHED){
            printf("SVO reached end. Replay. \n");
            zed.setSVOPosition(1);
        }
        else
            printf("Grab failed. \n");


    printf("exit program. \n");

    return 0;
}
