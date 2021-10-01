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
    runParameters.confidence_threshold = 50;
    runParameters.texture_confidence_threshold = 100;

    // Initialize ZED Objects
    auto cameraConfig = zed.getCameraInformation().camera_configuration;
    sl::Mat image(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::Mat depth(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::F32_C1, sl::MEM::CPU);

    // buffer binding for cv objects
    cv::Mat imageCv = zed_utils::slMat2cvMat(image);
    cv::Mat imageCv3Ch;
    cv::Mat depthCv = zed_utils::slMat2cvMat(depth);
    cv::Size displaySize(720, 404);
    cv::Mat imageDisplay(displaySize, CV_8UC4);
    cv::Mat depthDisplay(displaySize, CV_32FC1);
    // open3d object
    auto  imageO3dPtr = std::make_shared<open3d::geometry::Image>();
    auto  depthO3dPtr = std::make_shared<open3d::geometry::Image>();
    imageO3dPtr->Prepare(cameraConfig.resolution.width, cameraConfig.resolution.height,
                         3,1); // 8U_3C
    depthO3dPtr->Prepare(cameraConfig.resolution.width, cameraConfig.resolution.height,
                         1, 4); // 32F_1C

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Open3D image",720,404);
    vis.AddGeometry(imageO3dPtr);

    sl::Objects objects;

    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){
            // 1. retrieve
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::CPU);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
            cv::cvtColor(imageCv, imageCv3Ch, cv::COLOR_BGRA2RGB);
            o3d_utils::fromCvMat(imageCv3Ch, *imageO3dPtr);
            o3d_utils::fromCvMat(depthCv, *depthO3dPtr);
            printf("Image + depth retrieved in %.3f ms \n" ,timer.stop());

            // 2. draw
            vis.UpdateGeometry();
            vis.PollEvents();
            vis.UpdateRender();

            cv::resize(imageCv, imageDisplay, displaySize);
            cv::resize(depthCv, depthDisplay, displaySize);
//            cv::imshow("Image", imageDisplay);
            cv::imshow("Depth", depthDisplay);
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
