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

    // Initialize Objects
    auto cameraConfig = zed.getCameraInformation().camera_configuration;
    sl::Mat pointCloud(cameraConfig.resolution, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
    sl::Mat image;


    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::GPU);
            zed.retrieveMeasure(pointCloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU);
            printf("Pointcloud retrieved in %.3f ms \n" ,timer.stop());
        }else
            printf("Grab failed. \n");

    printf("exit program. \n");

    return 0;
}
