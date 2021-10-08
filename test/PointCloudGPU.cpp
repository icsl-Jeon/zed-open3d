#include <ZedUtils.h>
#include <csignal>
#include <Misc.h>

volatile sig_atomic_t stop;

namespace o3d_tensor =  open3d::t::geometry ;
namespace o3d_legacy = open3d::geometry;
namespace o3d_core = open3d::core;
namespace o3d_vis = open3d::visualization;

sl::Camera zed;

std::shared_ptr<o3d_tensor::PointCloud> pointsO3dPtr;
std::shared_ptr<o3d_vis::visualizer::O3DVisualizer> vis;
std::mutex locker;
string cloudName = "zed points";

void updateThread(){
    // ZED dynamic parameter
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
    auto rgbdO3dPtr_cpu = std::make_shared<o3d_legacy::RGBDImage>();

    open3d::camera::PinholeCameraIntrinsic intrinsicO3d;
    intrinsicO3d.SetIntrinsics(col,row,fx,fy,cx,cy);
    o3d_core::Tensor intrinsicO3dTensor =
            open3d::core::eigen_converter::EigenMatrixToTensor(intrinsicO3d.intrinsic_matrix_);
    intrinsicO3dTensor.To(device_gpu);
    auto extrinsicO3dTensor = o3d_core::Tensor::Eye(4,o3d_core::Float64,device_gpu);

    // open3d visualization
    auto mat = o3d_vis::rendering::Material();
    mat.shader = "defaultUnlit";
    auto pointsO3dPtr_cpu = std::make_shared<o3d_tensor::PointCloud>() ;

    // retrieving thread start
    bool isInit = false;
    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){

            // retrieve ZED image in GPU
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::GPU);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::GPU);
            cv::cuda::cvtColor(imageCv, imageCv3Ch,cv::COLOR_BGRA2RGB);

            // construct pointcloud. We lock with the same mutex in the posted thread on visualizer
            {
                std::lock_guard<std::mutex> lock(locker);
                o3d_tensor::RGBDImage rgbdImage (imageO3d,depthO3d);
                *pointsO3dPtr = (o3d_tensor::PointCloud::CreateFromRGBDImage(rgbdImage,
                                                                         intrinsicO3dTensor, extrinsicO3dTensor,
                                                                         1, 5, 2));
            }
            printf("Image + depth + points retrieved in %.3f ms. \n" ,timer.stop());

            // not mandatory..
            misc::Timer timerVis;
            *pointsO3dPtr_cpu = pointsO3dPtr->To(o3d_core::Device("CPU:0")); // ToLegacy() takes much time but To() is okay.
            printf("cpu transfer took %.3f ms. (only for visualization)\n" ,timerVis.stop());

            // initialize viewport
            if (not isInit){
                /** Dummay operation to initialize cuda scope for .ToLegacy() **/
                imageCv3Ch.download(imageCv3Ch_cpu);
                o3d_core::Tensor tensorRgbDummy(
                        (imageCv3Ch_cpu.data), {row,col,3},rgbType,device_gpu);

                // configure initial viewpoint
                auto pointsCenter = pointsO3dPtr->GetCenter();
                auto centerCPU = pointsCenter.ToFlatVector<float>();
                Eigen::Vector3f pointsCenterEigen(centerCPU[0],centerCPU[1],centerCPU[2]);
                Eigen::Vector3f eye = pointsCenterEigen + Eigen::Vector3f(0,0,-3);

                o3d_vis::gui::Application::GetInstance().PostToMainThread(
                    vis.get(), [mat,pointsCenterEigen,eye,pointsO3dPtr_cpu](){
                        std::lock_guard<std::mutex> lock (locker);
                        vis->AddGeometry(cloudName,pointsO3dPtr_cpu, &mat);
                        vis->ResetCameraToDefault();
                        vis->SetupCamera(60,pointsCenterEigen,eye,{0.0, -1.0, 0.0});
                    }
                );
                isInit = true;
            }else{
                // should UpdateGeometry used?
                o3d_vis::gui::Application::GetInstance().PostToMainThread(
                    vis.get(), [mat, pointsO3dPtr_cpu](){
                        std::lock_guard<std::mutex> lock (locker);
                        vis->GetScene()->GetScene()->UpdateGeometry(cloudName,*pointsO3dPtr_cpu,
                                                        o3d_vis::rendering::Scene::kUpdatePointsFlag |
                                                                    o3d_vis::rendering::Scene::kUpdateColorsFlag);
                        vis->SetPointSize(2); // this calls ForceRedraw(), readily update the viewport.
                    }
                );

            }
        }else if (zed.grab(runParameters) == sl::ERROR_CODE::END_OF_SVOFILE_REACHED){
            printf("SVO reached end. Replay. \n");
            zed.setSVOPosition(1);
        }
        else
            printf("Grab failed. \n");

}


int main(int argc, char** argv) {
    // Initialize ZED camera
    sl::InitParameters initParameters;
    initParameters.coordinate_units = sl::UNIT::METER;
    zed_utils::parseArgs(argc, argv, initParameters);
    zed_utils::initCamera(zed, initParameters);

    // Initialize open3d
    pointsO3dPtr = std::make_shared<o3d_tensor::PointCloud>();
    const char *const resource_path{"/usr/local/bin/Open3D/resources"};
    o3d_vis::gui::Application::GetInstance().Initialize(resource_path);
    vis = std::make_shared<o3d_vis::visualizer::O3DVisualizer>("Open3d - PointCloud",800,600);
    o3d_vis::gui::Application::GetInstance().AddWindow(vis);
    std::thread cameraThread (updateThread);

    // Run application
    o3d_vis::gui::Application::GetInstance().Run();
    cameraThread.join();

    return 0;
}
