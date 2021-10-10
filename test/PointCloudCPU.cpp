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
    sl::Mat image(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::U8_C4,  sl::MEM::CPU);
    sl::Mat depth(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::F32_C1, sl::MEM::CPU);
    sl::Objects objects;
    sl::CalibrationParameters intrinsicParam = zed.getCameraInformation().calibration_parameters;
    float fx = intrinsicParam.left_cam.fx;
    float fy = intrinsicParam.left_cam.fy;
    float cx = intrinsicParam.left_cam.cx;
    float cy = intrinsicParam.left_cam.cy;

    // buffer binding for cv objects
    cv::Mat imageCv = zed_utils::slMat2cvMat(image); // bound buffer (rgba)
    cv::Mat depthCv = zed_utils::slMat2cvMat(depth); // bound buffer
    cv::Mat imageCv3Ch (row,col,CV_8UC3);

    // open3d object
    o3d_core::Device device_gpu("CUDA:0");
    o3d_core::Device device_cpu("CPU:0");
    o3d_core::Dtype rgbType = o3d_core::Dtype::UInt8;
    o3d_core::Dtype depthType = o3d_core::Dtype::Float32;

    // bind open3d and opencv in cuda memory
    auto rgbBlob = std::make_shared<o3d_core::Blob>(
            device_cpu,imageCv3Ch.data, nullptr);
    auto  rgbTensor = o3d_core::Tensor({row,col,3},{3*col,3,1},
                                       rgbBlob->GetDataPtr(),rgbType,rgbBlob); // rgbTensor.IsContiguous() = true
    o3d_tensor::Image imageO3d(rgbTensor);

    auto depthBlob = std::make_shared<o3d_core::Blob>(
            device_cpu, depthCv.data, nullptr);
    auto depthTensor = o3d_core::Tensor({row,col,1}, {col,1,1},
                                        depthBlob->GetDataPtr(),depthType,depthBlob);
    o3d_tensor::Image depthO3d(depthTensor);

    // pointcloud construction
    auto rgbdO3dPtr_cpu = std::make_shared<o3d_legacy::RGBDImage>();

    open3d::camera::PinholeCameraIntrinsic intrinsicO3d;
    intrinsicO3d.SetIntrinsics(col,row,fx,fy,cx,cy);
    o3d_core::Tensor intrinsicO3dTensor =
            open3d::core::eigen_converter::EigenMatrixToTensor(intrinsicO3d.intrinsic_matrix_);
    intrinsicO3dTensor.To(device_cpu);
    auto extrinsicO3dTensor = o3d_core::Tensor::Eye(4,o3d_core::Float64,device_cpu);

    // open3d visualization
    auto mat = o3d_vis::rendering::Material();
    mat.shader = "defaultUnlit";

    // retrieving thread start
    bool isInit = false;
    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){

            // retrieve ZED image in GPU
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::CPU);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
            cv::cvtColor(imageCv, imageCv3Ch,cv::COLOR_BGRA2RGB);

            // construct pointcloud. We lock with the same mutex in the posted thread on visualizer
            {
                std::lock_guard<std::mutex> lock(locker);
                misc::Timer timerPCL;
                o3d_tensor::RGBDImage rgbdImage (imageO3d,depthO3d);
                *pointsO3dPtr = (o3d_tensor::PointCloud::CreateFromRGBDImage(rgbdImage,
                                                                         intrinsicO3dTensor, extrinsicO3dTensor,
                                                                         1, 5, 2));
//                printf("pointcloud was constructed in %.3f ms. \n" ,timerPCL.stop());

            }
            printf("Image + depth + points retrieved in %.3f ms. \n" ,timer.stop());


            // initialize viewport
            if (not isInit){

                // configure initial viewpoint
                auto pointsCenter = pointsO3dPtr->GetCenter();
                auto centerCPU = pointsCenter.ToFlatVector<float>();
                Eigen::Vector3f pointsCenterEigen(centerCPU[0],centerCPU[1],centerCPU[2]);
                Eigen::Vector3f eye = pointsCenterEigen + Eigen::Vector3f(0,0,-3);

                o3d_vis::gui::Application::GetInstance().PostToMainThread(
                    vis.get(), [mat,pointsCenterEigen,eye](){
                        std::lock_guard<std::mutex> lock (locker);
                        vis->AddGeometry(cloudName,pointsO3dPtr, &mat);
                        vis->ResetCameraToDefault();
                        vis->SetupCamera(60,pointsCenterEigen,eye,{0.0, -1.0, 0.0});
                    }
                );
                isInit = true;
            }else{
                o3d_vis::gui::Application::GetInstance().PostToMainThread(
                    vis.get(), [mat](){
                        std::lock_guard<std::mutex> lock (locker);
                        vis->GetScene()->GetScene()->UpdateGeometry(cloudName,*pointsO3dPtr,
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
