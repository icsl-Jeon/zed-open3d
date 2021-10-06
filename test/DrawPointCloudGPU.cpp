#include <ZedUtils.h>
#include <csignal>
#include <Misc.h>

volatile sig_atomic_t stop;
bool isInit = false;

// Zed
sl::Camera zed;

std::shared_ptr<open3d::visualization::gui::Window> window;
std::shared_ptr<open3d::visualization::gui::SceneWidget> widget;

void runZedThread(std::shared_ptr<open3d::t::geometry::PointCloud> pclGpu,
                  open3d::visualization::gui::Application* appPtr){

    // Setting zed running parameters
    sl::RuntimeParameters runParameters;
    runParameters.confidence_threshold = 10;
    runParameters.texture_confidence_threshold = 100;

    // Initialize ZED Objects
    auto cameraConfig = zed.getCameraInformation().camera_configuration;
    sl::Mat image(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    sl::Mat points;

    // buffer binding for cv objects
    cv::Mat imageCv = zed_utils::slMat2cvMat(image);
    cv::Mat imageCv3Ch;
    cv::Size displaySize(720, 404);
    cv::Mat imageDisplay(displaySize, CV_8UC4);

    // CPU open3d temp data (todo: removed)
    auto pointsO3dPtr = std::make_shared<open3d::geometry::PointCloud>();

    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){
            // 1. retrieve zed
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::CPU);
            cv::cvtColor(imageCv, imageCv3Ch, cv::COLOR_BGRA2RGB);
            zed.retrieveMeasure(points, sl::MEASURE::XYZRGBA, sl::MEM::CPU); // MEM GPU is faster than CPU 2 times.
            o3d_utils::fromSlPoints(points, *pointsO3dPtr);

            // 2. Transfer pointcloud to gpu. Note that the visualizer requires pcl in cpu for valid coloring.
            open3d::core::Device device("CUDA:0");
            open3d::core::Dtype dtype = open3d::core::Dtype::Float32;
            *pclGpu = open3d::t::geometry::PointCloud::FromLegacy(*pointsO3dPtr,dtype,device);

            auto cloudMat= open3d::visualization::rendering::Material();
            cloudMat.point_size = 2;
            if (not isInit) {
                appPtr->PostToMainThread(window.get(), [&pclGpu, &cloudMat](){
                    widget->GetScene()->GetScene()->AddGeometry("pcl",*pclGpu,cloudMat);
                                             auto bbox = widget->GetScene()
                                                     ->GetBoundingBox();
                                             auto center = bbox.GetCenter().cast<float>();
                                             widget->SetupCamera(18, bbox, center);
                                             widget->LookAt(
                                                     center, center - Eigen::Vector3f{-10, 5, 8},
                                                     {0.0f, -1.0f, 0.0f});
                                         }
                );
                isInit = true;
            }


//            open3d::utility::LogWarning(
//                    "Target Pointcloud color dtype = {}  ",
//                    pclGpu->GetPointColors().GetDtype().ToString());
            appPtr->PostToMainThread(window.get(),  [&pclGpu, &cloudMat](){
                widget->GetScene()->GetScene()->UpdateGeometry(
                        "pcl",*pclGpu,
                        open3d::visualization::rendering::Scene::kUpdatePointsFlag |
                        open3d::visualization::rendering::Scene::kUpdateColorsFlag);
            });


            // 3. opencv window
            cv::resize(imageCv, imageDisplay, displaySize);
            cv::imshow("Image", imageDisplay);
            cv::waitKey(1);

        }else if (zed.grab(runParameters) == sl::ERROR_CODE::END_OF_SVOFILE_REACHED){
            // replay option
            printf("SVO reached end. Replay. \n");
            zed.setSVOPosition(1);
        }
        else
            printf("Grab failed. \n");

}

int main(int argc, char** argv) {

    // Initialize ZED camera
    sl::InitParameters initParameters;
    zed_utils::parseArgs(argc, argv, initParameters);
    zed_utils::initCamera(zed, initParameters);

    // Initialize application
    auto  pointsO3dGPU  = std::make_shared<open3d::t::geometry::PointCloud>(); // GPU
    pointsO3dGPU->To(open3d::core::Device("CUDA:0"));
    const std::string resource_path{"/usr/local/bin/Open3D/resources"};
    auto& app = open3d::visualization::gui::Application::GetInstance();
    app.Initialize(resource_path.c_str());

    window= std::make_shared<open3d::visualization::gui::Window>(
            "PointCloud", 800, 600);
    widget = std::make_shared<open3d::visualization::gui::SceneWidget>();
    widget->SetScene(std::make_shared<open3d::visualization::rendering::Open3DScene>(window->GetRenderer()));
    widget->GetScene()->SetBackground({0,0,0,1});
    window->AddChild(widget);
    app.AddWindow(window);
//    // Run threads
    std::thread appThread = std::thread(runZedThread,pointsO3dGPU, &app);
    app.Run();

    return 0;
}
