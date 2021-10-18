#include <ZedUtils.h>
#include <csignal>
#include <Misc.h>

volatile sig_atomic_t stop;

namespace o3d_tensor =  open3d::t::geometry ;
namespace o3d_legacy = open3d::geometry;
namespace o3d_core = open3d::core;
namespace o3d_vis = open3d::visualization;

sl::Camera zed;
zed_utils::Gaze gaze;

std::shared_ptr<o3d_tensor::PointCloud> pointsO3dPtr;
std::shared_ptr<o3d_tensor::PointCloud> objPointsO3dPtr;
std::shared_ptr<o3d_legacy::PointCloud> objPointsO3dPtr_cpu;
std::shared_ptr<o3d_legacy::LineSet> skeletonO3dPtr ; // object skeleton
std::shared_ptr<o3d_vis::visualizer::O3DVisualizer> vis;
std::shared_ptr<o3d_legacy::TriangleMesh> attentionPointSet[4]; // left eye, right eye, left wrist, right wrist
std::shared_ptr<o3d_legacy::TriangleMesh> gazeCoordinate;

std::mutex locker;
string cloudName = "zed points";
string skeletonName = "human skeleton";
string attentionName[4] = {"left eye", "right eye", "left hand", "right hand"};
string gazeName = "gaze coordinate";

Detector* yoloDetectorPtr;
vector<string> objectNames;

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
                int current_det_fps = -1, int current_cap_fps = -1) {
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {

        int obj_id = i.obj_id;
        int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
        int const offset = obj_id * 123457 % 6;
        int const color_scale = 150 + (obj_id * 123457) % 100;
        cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
        color *= color_scale;

        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }

            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                          cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                          color,cv::FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
} // draw box


void updateThread(){
    // ZED dynamic parameter
    sl::RuntimeParameters runParameters;
    runParameters.confidence_threshold = 50;
    runParameters.texture_confidence_threshold = 100;

    sl::ObjectDetectionRuntimeParameters objectParameters;

    // Initialize ZED Objects
    auto cameraConfig = zed.getCameraInformation().camera_configuration;
    sl::Resolution imageResultion = zed.getCameraInformation().camera_resolution;
    const int row = imageResultion.height;
    const int col = imageResultion.width;
    sl::Mat image(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::U8_C4,  sl::MEM::GPU);
    sl::Mat depth(cameraConfig.resolution.width, cameraConfig.resolution.height, sl::MAT_TYPE::F32_C1, sl::MEM::GPU);
    sl::Objects humanObjects;
    sl::Pose zedPose;

    sl::CameraConfiguration intrinsicParam = zed.getCameraInformation().camera_configuration;
    float fx = intrinsicParam.calibration_parameters.left_cam.fx;
    float fy = intrinsicParam.calibration_parameters.left_cam.fy;
    float cx = intrinsicParam.calibration_parameters.left_cam.cx;
    float cy = intrinsicParam.calibration_parameters.left_cam.cy;

    // buffer binding for cv objects
    cv::cuda::GpuMat imageCv = zed_utils::slMat2cvMatGPU(image); // bound buffer (rgba)
    cv::cuda::GpuMat depthCv = zed_utils::slMat2cvMatGPU(depth); // bound buffer
    cv::cuda::GpuMat imageCv3Ch = cv::cuda::createContinuous(row,col,CV_8UC3);
    cv::Mat imageCv3Ch_cpu; // will not be used. But needed for cuda scope
    cv::Mat imageDetectCv3ch_cpu;

    // darknet objects
    cv::Size network_size = cv::Size(yoloDetectorPtr->get_net_width(),
                                     yoloDetectorPtr->get_net_height());
    cv::cuda::GpuMat resizedImage;
    shared_ptr<image_t> imageDarknet = make_shared<image_t>();
    uint w = network_size.width;
    uint h = network_size.height;
    uint c = 3;

    imageDarknet->w = w;
    imageDarknet->h = h;
    imageDarknet->c = 3;
    imageDarknet->data = (float *)calloc(w*h*c, sizeof(float));
    float threshold = 0.3; // larger than this

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

    auto nanTensor = o3d_core::Tensor::Full({row,col,1} ,NAN, depthType,device_gpu);


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

    auto matLine = o3d_vis::rendering::Material();
    matLine.shader = "unlitLine";
    matLine.line_width = 3.0;

    auto matAttention = o3d_vis::rendering::Material();
    matAttention.base_color = {0.9, 0.5,0.5, 0.5};
    matAttention.has_alpha = true; // seems not working

    auto pointsO3dPtr_cpu = std::make_shared<o3d_tensor::PointCloud>() ;

    // retrieving thread start
    bool isInit = false;
    bool isObject = false;
    while (!stop)
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS){

            // retrieve ZED in GPU
            misc::Timer timer;
            zed.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::GPU);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::GPU);
            cv::cuda::cvtColor(imageCv, imageCv3Ch,cv::COLOR_BGRA2RGB);
            imageCv3Ch.download(imageCv3Ch_cpu);
            auto isObjectSuccess = zed.retrieveObjects(humanObjects,objectParameters);

            double projectionElapse;
            // construct pointcloud. We lock with the same mutex in the posted thread on visualizer
            {
                std::lock_guard<std::mutex> lock(locker);

                o3d_tensor::RGBDImage rgbdImage (imageO3d,depthO3d);
                misc::Timer timerPclFromRGBD;
                *pointsO3dPtr = (o3d_tensor::PointCloud::CreateFromRGBDImage(rgbdImage,
                                                                         intrinsicO3dTensor, extrinsicO3dTensor,
                                                                         1, 3, 2));



                if (not humanObjects.object_list.empty()) {
                    o3d_utils::fromSlObjects(humanObjects.object_list[0],
                                             skeletonO3dPtr, attentionPointSet);
                    gaze = zed_utils::Gaze(humanObjects.object_list[0]);
                    isObject = true;
                }
            }
            printf("Image + depth + points + objects retrieved in %.3f ms. (pcl proj = %.3f ms) \n" ,timer.stop(),projectionElapse);

            // not mandatory..
            misc::Timer timerVis;
            *pointsO3dPtr_cpu = pointsO3dPtr->To(o3d_core::Device("CPU:0")); // ToLegacy() takes much time but To() is okay.
            printf("cpu transfer took %.3f ms. (only for visualization)\n" ,timerVis.stop());

            // detect in Darknet
            misc::Timer timerDetect;
            imageDetectCv3ch_cpu = imageCv3Ch_cpu.clone();
            cv::cvtColor(imageDetectCv3ch_cpu,imageDetectCv3ch_cpu,cv::COLOR_RGB2BGR);
            cv::cuda::resize(imageCv3Ch,resizedImage,network_size);
            cv::Mat resizedImageCpu; resizedImage.download(resizedImageCpu); // todo
            unsigned char *data = (unsigned char *)resizedImageCpu.data;
            int step = resizedImageCpu.step;
            for (int y = 0; y < h; ++y)
                for (int k = 0; k < c; ++k)
                    for (int x = 0; x < w; ++x)
                        imageDarknet->data[k*w*h + y*w + x] = data[y*step + x*c + (3-k)] / 255.0f;

            std::vector<bbox_t> resultBoundingBox =
                    yoloDetectorPtr->detect_resized(*imageDarknet,imageCv.cols,imageCv.rows,threshold,true);
            resultBoundingBox = yoloDetectorPtr->tracking_id(resultBoundingBox);
            draw_boxes(imageDetectCv3ch_cpu, resultBoundingBox, objectNames);
            printf("object detection in %.3f ms. \n" ,timerDetect.stop());
            cv::imshow("object detection",imageDetectCv3ch_cpu);
            cv::waitKey(1);

            // Extract points of detected objects
            misc::Timer timerObjPointsExtraction;
            auto objectsDepth = nanTensor.Clone();
            for (const auto& bb : resultBoundingBox){
                if (bb.obj_id == 39 ) { // 39 = bottle
                    auto bb_1_window = o3d_core::TensorKey::Slice(bb.y, bb.y + bb.h, 1);
                    auto bb_2_window = o3d_core::TensorKey::Slice(bb.x, bb.x + bb.w, 1);
                    objectsDepth.SetItem({bb_1_window, bb_2_window},
                                         depthTensor.GetItem({bb_1_window, bb_2_window}));
                }
            }

            *objPointsO3dPtr = o3d_tensor::PointCloud::CreateFromDepthImage(objectsDepth,
                                                                            intrinsicO3dTensor,extrinsicO3dTensor,
                                                                            1,3,1);
            *objPointsO3dPtr_cpu  = objPointsO3dPtr->ToLegacy();
            objPointsO3dPtr_cpu->PaintUniformColor({0,1,0});
            double elapse = timerObjPointsExtraction.stop();
            printf("%zu object points / %zu   processing : %.3f ms \n ",
                   objPointsO3dPtr_cpu->points_.size(),
                   pointsO3dPtr->GetPointPositions().GetLength(),
                   elapse);

            // initialize viewport
            if (not isInit){
                // Dummay operation to initialize cuda scope for .ToLegacy()
                o3d_core::Tensor tensorRgbDummy(
                        (imageCv3Ch_cpu.data), {row,col,3},rgbType,device_gpu);

                // configure initial viewpoint
                auto pointsCenter = pointsO3dPtr->GetCenter();
                auto centerCPU = pointsCenter.ToFlatVector<float>();
                Eigen::Vector3f pointsCenterEigen(centerCPU[0],centerCPU[1],centerCPU[2]);
                Eigen::Vector3f eye = pointsCenterEigen + Eigen::Vector3f(0,0,-3);

                o3d_vis::gui::Application::GetInstance().PostToMainThread(
                    vis.get(), [mat, matLine, matAttention, pointsCenterEigen,eye,pointsO3dPtr_cpu, isObject](){
                        std::lock_guard<std::mutex> lock (locker);
                        vis->AddGeometry("coordinate",
                                o3d_legacy::TriangleMesh::CreateCoordinateFrame(0.1)); // add coord at origin
                        vis->AddGeometry(cloudName,pointsO3dPtr_cpu, &mat);
                        vis->AddGeometry(cloudName + "_objects",objPointsO3dPtr_cpu, &mat);

                        if (isObject) {
                            vis->AddGeometry(skeletonName, skeletonO3dPtr, &matLine);
                            for (int i = 0; i<4 ;i++)
                                vis->AddGeometry(attentionName[i],attentionPointSet[i],&matAttention);
                            *gazeCoordinate = *o3d_legacy::TriangleMesh::CreateCoordinateFrame(0.1);
                            if (gaze.isValid()) {
                                gazeCoordinate->Transform(gaze.getTransformation().cast<double>());
                                vis->AddGeometry(gazeName, gazeCoordinate, &mat);
                            }
                        }
                        vis->ResetCameraToDefault();
                        vis->SetupCamera(60,pointsCenterEigen,eye,{0.0, -1.0, 0.0});
                    }
                );
                isInit = true;
            }else{
                o3d_vis::gui::Application::GetInstance().PostToMainThread(
                    vis.get(), [mat, matLine, matAttention, pointsO3dPtr_cpu, isObject](){
                        std::lock_guard<std::mutex> lock (locker);
                        vis->RemoveGeometry(cloudName);
                        vis->RemoveGeometry(cloudName+"_objects");

                        vis->AddGeometry(cloudName,pointsO3dPtr_cpu, &mat);
                        vis->AddGeometry(cloudName + "_objects",objPointsO3dPtr_cpu, &mat);
//                        vis->GetScene()->GetScene()->UpdateGeometry(cloudName,*pointsO3dPtr_cpu,
//                                                        o3d_vis::rendering::Scene::kUpdatePointsFlag |
//                                                                    o3d_vis::rendering::Scene::kUpdateColorsFlag);

                        if (isObject) {
                            // Idk why.. but
                            vis->RemoveGeometry(skeletonName);
                            vis->AddGeometry(skeletonName, skeletonO3dPtr, &matLine);
                            // visualizing attention points
                            for (int i = 0; i<4 ;i++) {
                                vis->RemoveGeometry(attentionName[i]);
                                vis->AddGeometry(attentionName[i], attentionPointSet[i], &matAttention);
                            }
                            if (gaze.isValid()) {
                                // visualizing gaze coordinate
                                vis->RemoveGeometry(gazeName);
                                *gazeCoordinate = *o3d_legacy::TriangleMesh::CreateCoordinateFrame(0.1);
                                gazeCoordinate->Transform(gaze.getTransformation().cast<double>());
                                vis->AddGeometry(gazeName, gazeCoordinate, &mat);
                            }
                        }
                        vis->SetPointSize(1); // this calls ForceRedraw(), readily update the viewport.
//                        vis->SetLineWidth(3);
                        vis->SetBackground({0,0,0,1});
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

    // Darknet (this should be instantiated first due to cuda context problem)
    string darknetPath = "/home/jbs/catkin_ws/src/darknet/";
    string  namesFile = darknetPath + "data/coco.names";
    string cfgFile = darknetPath + "cfg/yolov4-tiny.cfg"; // only tiny fits for fast displaying
    string weightsFile = darknetPath + "yolov4-tiny.weights";
    yoloDetectorPtr = new Detector(cfgFile, weightsFile); // take a bit for initialization
    ifstream  file(namesFile);
    for(std::string line; getline(file, line);)
        objectNames.push_back(line);

    // Initialize ZED camera
    sl::InitParameters initParameters;
    initParameters.sdk_cuda_ctx = (CUcontext)yoloDetectorPtr->get_cuda_context();
    initParameters.sdk_gpu_id = yoloDetectorPtr->cur_gpu_id;
    initParameters.coordinate_units = sl::UNIT::METER;
    zed_utils::parseArgs(argc, argv, initParameters);
    zed_utils::initCamera(zed, initParameters);
    cv::namedWindow("object detection",cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("object detection", 600,400);

    // Initialize open3d
    pointsO3dPtr = std::make_shared<o3d_tensor::PointCloud>();
    objPointsO3dPtr = std::make_shared<o3d_tensor::PointCloud>();
    objPointsO3dPtr_cpu = std::make_shared<o3d_legacy::PointCloud>();
    skeletonO3dPtr = std::make_shared<o3d_legacy::LineSet>();
    gazeCoordinate = std::make_shared<o3d_legacy::TriangleMesh>();

    for (auto & attentionPoint : attentionPointSet) {
        attentionPoint =
                o3d_legacy::TriangleMesh::CreateSphere(0.02, 20);
        float r= 250.0/255.0, g = 100.0/255.0, b = 100.0/255.0;
        attentionPoint->PaintUniformColor({r,g,b});
    }
    const char *const resource_path{"/usr/local/bin/Open3D/resources"};
    o3d_vis::gui::Application::GetInstance().Initialize(resource_path);
    vis = std::make_shared<o3d_vis::visualizer::O3DVisualizer>("Open3d - PointCloud",800,600);
    o3d_vis::gui::Application::GetInstance().AddWindow(vis);

    // Run application
    std::thread cameraThread (updateThread);
    o3d_vis::gui::Application::GetInstance().Run();
    cameraThread.join();

    return 0;
}
