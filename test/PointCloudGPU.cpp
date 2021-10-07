#include <ZedUtils.h>
#include <csignal>
#include <Misc.h>

volatile sig_atomic_t stop;

namespace o3d_tensor =  open3d::t::geometry ;
namespace o3d_legacy = open3d::geometry;
namespace o3d_core = open3d::core;

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
    sl::Resolution imageResultion = zed.getCameraInformation().camera_resolution;
    const int row = imageResultion.height;
    const int col = imageResultion.width;
    sl::Mat image(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::U8_C4,  sl::MEM::GPU);
    sl::Mat depth(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::F32_C1, sl::MEM::GPU);
    sl::Objects objects;
    sl::CalibrationParameters intrinsicParam = zed.getCameraInformation().calibration_parameters;
    float fx = intrinsicParam.left_cam.fx;
    float fy = intrinsicParam.left_cam.fy;
    float cx = intrinsicParam.left_cam.cx;
    float cy = intrinsicParam.left_cam.fy;

    // buffer binding for cv objects
    cv::cuda::GpuMat imageCv = zed_utils::slMat2cvMatGPU(image); // bound buffer (rgba)
    cv::cuda::GpuMat depthCv = zed_utils::slMat2cvMatGPU(depth); // bound buffer
    cv::cuda::GpuMat imageCv3Ch = cv::cuda::createContinuous(row,col,CV_8UC3);
    cv::Mat imageCv3Ch_cpu; // will not be used. But needed for cuda scope

    // open3d object
    o3d_core::Device device_gpu("CUDA:0");
    o3d_core::Device device_cpu("CPU:0");
    o3d_core::Dtype rgbType = o3d_core::Dtype::UInt8;
    o3d_core::Dtype depthType = o3d_core::Dtype::Float32;

    // bind open3d and opencv in cuda memory
    auto rgbBlob = std::make_shared<o3d_core::Blob>(
            device_gpu,imageCv3Ch.cudaPtr(), nullptr);
    auto  rgbTensor = o3d_core::Tensor({row,col,3},{3*col,3,1},
                               rgbBlob->GetDataPtr(),rgbType,rgbBlob); // rgbTensor.IsContiguous() = true
    o3d_tensor::Image imageO3d(rgbTensor);

    auto depthBlob = std::make_shared<o3d_core::Blob>(
            device_gpu, depthCv.cudaPtr(), nullptr);
    auto depthTensor = o3d_core::Tensor({row,col,1}, {col,1,1},
                      depthBlob->GetDataPtr(),depthType,depthBlob);
    o3d_tensor::Image depthO3d(depthTensor);

    // pointcloud construction
    auto pointsO3d = o3d_tensor::PointCloud();
    auto pointsO3dPtr_cpu =
            std::make_shared<o3d_legacy::PointCloud>(); // Note: ToLegacy() critically slow for pcl.
    auto rgbdO3dPtr_cpu = std::make_shared<o3d_legacy::RGBDImage>();

    open3d::camera::PinholeCameraIntrinsic intrinsicO3d;
    intrinsicO3d.SetIntrinsics(col,row,fx,fy,cx,cy);
    o3d_core::Tensor intrinsicO3dTensor =
            open3d::core::eigen_converter::EigenMatrixToTensor(intrinsicO3d.intrinsic_matrix_);
    intrinsicO3dTensor.To(device_gpu);
    auto extrinsicO3dTensor = o3d_core::Tensor::Eye(4,o3d_core::Float64,device_gpu);

    // just for visualization in open3d
    open3d::visualization::Visualizer vis;
    bool isInit = false;
    vis.CreateVisualizerWindow("Open3D points",600,400);

    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){

            // retrieve ZED image in GPU
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::GPU);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::GPU);
            cv::cuda::cvtColor(imageCv, imageCv3Ch,cv::COLOR_BGRA2RGB);

            // construct pointcloud
            o3d_tensor::RGBDImage rgbdImage (imageO3d,depthO3d);
            pointsO3d = (o3d_tensor::PointCloud::CreateFromRGBDImage(rgbdImage,
                                                                     intrinsicO3dTensor,extrinsicO3dTensor,
                                                                     1000,5,2));
            printf("Image + depth + points retrieved in %.3f ms. \n" ,timer.stop());

            if (not isInit){
                /** Dummay operation to initialize cuda scope for .ToLegacy() **/
                imageCv3Ch.download(imageCv3Ch_cpu);
                o3d_core::Tensor tensorRgbDummy(
                        (imageCv3Ch_cpu.data), {row,col,3},rgbType,device_gpu);

                *pointsO3dPtr_cpu = pointsO3d.ToLegacy();
                vis.AddGeometry(pointsO3dPtr_cpu);
                isInit = true;
            }


            // 2. Open3d
            misc::Timer timerTransfer;
            *pointsO3dPtr_cpu = pointsO3d.ToLegacy();
            printf("transferred points in %.3f ms. \n" ,timerTransfer.stop());
            vis.UpdateGeometry();
            vis.PollEvents();
            vis.UpdateRender();


        }else if (zed.grab(runParameters) == sl::ERROR_CODE::END_OF_SVOFILE_REACHED){
            printf("SVO reached end. Replay. \n");
            zed.setSVOPosition(1);
        }
        else
            printf("Grab failed. \n");

    printf("exit program. \n");

    return 0;
}
