#include <ZedUtils.h>
#include <csignal>
#include <Misc.h>

volatile sig_atomic_t stop;

namespace o3d_tensor =  open3d::t::geometry ;

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

    // open3d object
    auto  imageO3dPtr = std::make_shared<open3d::geometry::Image>();
    auto  depthO3dPtr = std::make_shared<open3d::geometry::Image>();
    imageO3dPtr->Prepare(cameraConfig.resolution.width, cameraConfig.resolution.height,
                         3,1); // 8U_3C
    depthO3dPtr->Prepare(cameraConfig.resolution.width, cameraConfig.resolution.height,
                         1, 4); // 32F_1C

    auto imageO3d_gpu = std::make_shared<o3d_tensor::Image>();
    imageO3d_gpu->To(open3d::core::Device("CUDA:0"));
    auto depthO3d_gpu = std::make_shared<o3d_tensor::Image>();
    depthO3d_gpu->To(open3d::core::Device("CUDA:0"));

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Open3D image",720,404);
    vis.AddGeometry(imageO3dPtr);

    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){
            // 1. retrieve ZED image to CPU
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::CPU);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
            cv::cvtColor(imageCv, imageCv3Ch, cv::COLOR_BGRA2RGB);

            // 2. Open3d legacy + gpu images
            o3d_utils::fromCvMat(imageCv3Ch, *imageO3dPtr);
            *imageO3d_gpu = imageO3d_gpu->FromLegacy(*imageO3dPtr,open3d::core::Device("CUDA:0"));
            o3d_utils::fromCvMat(depthCv, *depthO3dPtr);
            *depthO3d_gpu = depthO3d_gpu->FromLegacy(*depthO3dPtr,open3d::core::Device("CUDA:0"));
            printf("Image + depth retrieved to GPU in %.3f ms \n" ,timer.stop());

            vis.UpdateGeometry();
            vis.PollEvents();
            vis.UpdateRender();

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
