#include <ZedUtils.h>
#include <csignal>
#include <Misc.h>

volatile sig_atomic_t stop;

namespace o3d_tensor =  open3d::t::geometry ;
namespace o3d_legacy = open3d::geometry;
namespace o3d_core = open3d::core;
namespace o3d_vis = open3d::visualization;

struct AttentionStatus{
    float angleFromGaze = INFINITY;
    float distToLeftHand = INFINITY;
    float distToRightHand = INFINITY;
    float weightHands = 0.1;
    bbox_t boundingBox; // bounding box in image
    float getAttentionCost () {
        return weightHands*(distToLeftHand + distToRightHand) + angleFromGaze;}

};

sl::Camera zed;
zed_utils::Gaze gaze;
Eigen::Vector3f leftHand; // left hand location
Eigen::Vector3f rightHand; // right hand location

const int nMaxOoiClusters = 4;
int nOoiClustersPrev = 0;
bool drawHistogram = false;
float gazeFov = M_PI /3.0 *2.0; // cone (height/rad) = tan(gazeFov/2)

o3d_tensor::TSDFVoxelGrid* volumePtr;
std::shared_ptr<o3d_legacy::TriangleMesh> meshPtr;

std::shared_ptr<o3d_tensor::PointCloud> objPointsO3dPtr;
std::shared_ptr<o3d_legacy::PointCloud> objPointsO3dPtr_cpu[nMaxOoiClusters];
std::shared_ptr<o3d_legacy::AxisAlignedBoundingBox> objBbO3dPtr[nMaxOoiClusters];
cv::Mat objPixelMask; // size = original stereo image. 1  = belongs to at least one object. Currently, simple box
cv::Mat humanPixelMask;
std::shared_ptr<o3d_legacy::TriangleMesh> objCentersO3dPtr[nMaxOoiClusters];


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

void signalHandler( int signum ) {
    cout << "Interrupt signal (" << signum << ") received.\n";

    // cleanup and close up stuff here
    // terminate program
    zed.disableRecording();
    zed.close();
    exit(signum);
}

// default yolo bounding box
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
                ss << std::fixed << std::setprecision(1) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
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


// Attention score display
void draw_attention_box(cv::Mat mat_img,
                        std::vector<AttentionStatus> attentionStatusSet) {
    int nObject = attentionStatusSet.size();
    string obj_name = "OOI_" ; // following original code convention
    std::sort(attentionStatusSet.begin(),
                                      attentionStatusSet.end(),
                                      [](AttentionStatus& attentionLhs, AttentionStatus& attentionRhs){
                                        return attentionLhs.getAttentionCost() < attentionRhs.getAttentionCost();}
                                      );

    int const colors[4][3] = {{51,51,255}, {0,128,255}, {0,255,255},{51,255,51}}; // BGR
    for (int nn = 0 ; nn < nObject; nn++) {
        // determine color: INF = black. the others = red, orange, yellow, green
        bbox_t box = attentionStatusSet[nn].boundingBox;
        cv::Scalar color;
        bool isAttentionCalculated = attentionStatusSet[nn].getAttentionCost() != INFINITY;
        if (not isAttentionCalculated) {
            color = cv::Scalar(10, 10, 10);
        } else {
            int colorIdx = min(nn, 3);
            color = cv::Scalar(colors[colorIdx][0],
                               colors[colorIdx][1],
                               colors[colorIdx][2]);
        }
        cv::rectangle(mat_img, cv::Rect(box.x, box.y, box.w, box.h), color, 2); // rectangle frame

        if (isAttentionCalculated) {
            float score = attentionStatusSet[nn].getAttentionCost();

            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << "OOI_" << score ;
            string scoreString = ss.str();

            obj_name =  scoreString;
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > box.w + 2) ? text_size.width : (box.w + 2);
            max_width = std::max(max_width, (int) box.w + 2);
            cv::rectangle(mat_img, cv::Point2f(std::max((int) box.x - 1, 0), std::max((int) box.y - 35, 0)),
                          cv::Point2f(std::min((int) box.x + max_width, mat_img.cols - 1),
                                      std::min((int) box.y, mat_img.rows - 1)),
                          color, cv::FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(box.x, box.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2,
                    cv::Scalar(0, 0, 0), 2);
        }
    } // iter: detection box
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
    sl::Mat points;
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
    cv::Mat depthCv_cpu;
    cv::Mat imageDetectCv3ch_cpu;
    cv::Mat depthDetectCv_cpu;
    cv::Mat depthColorized_cpu;

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
    matAttention.has_alpha = true; // seems not working ...

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
            auto isObjectSuccess = zed.retrieveObjects(humanObjects,objectParameters);

            double projectionElapse;
            {
                std::lock_guard<std::mutex> lock(locker);


                if (not humanObjects.object_list.empty()) {
                    // skeleton extraction
                    o3d_utils::fromSlObjects(humanObjects.object_list[0],
                                             skeletonO3dPtr, attentionPointSet);
                    // gaze extraction
                    gaze = zed_utils::Gaze(humanObjects.object_list[0]);

                    // pixel mask extraction (todo gpu?)
                    auto human_bb = humanObjects.object_list[0].bounding_box_2d;
                    cv::Mat humanPixelMaskSub = zed_utils::slMat2cvMat(humanObjects.object_list[0].mask).clone();
                    int kernelSize = 10;
                    cv::erode(humanPixelMaskSub, humanPixelMaskSub, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize)));
                    cv::dilate(humanPixelMaskSub, humanPixelMaskSub, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize)));

                    cv::Range rowRangeHuman(human_bb[0].y, human_bb[0].y + humanPixelMaskSub.rows );
                    cv::Range colRangeHuman(human_bb[0].x, human_bb[0].x + humanPixelMaskSub.cols);
                    humanPixelMask = cv::Mat::zeros(imageCv3Ch.size(),CV_8UC1);
                    cv::Mat humanPixelMaskSubPartPtr = humanPixelMask(rowRangeHuman,colRangeHuman); // shallow copy
                    humanPixelMaskSub.copyTo(humanPixelMaskSubPartPtr);

                    cv::bitwise_not(humanPixelMask,humanPixelMask);

                    isObject = true;
                }
            }
            printf("Image + depth + points + objects retrieved in %.3f ms. (pcl proj = %.3f ms) \n" ,timer.stop(),projectionElapse);


            // detect in Darknet
            depthCv.download(depthCv_cpu); // depthCv = war depth
            depthDetectCv_cpu  = depthCv_cpu.clone() ;
            depthDetectCv_cpu.convertTo(depthDetectCv_cpu,CV_8UC3,255, 0); // modified for visualization
            cv::cvtColor(depthDetectCv_cpu,depthDetectCv_cpu,cv::COLOR_GRAY2BGR); // modified for visualization
//            cv::imshow("depth",depthDetectCv_cpu);
//            cv::waitKey(1);

            misc::Timer timerDetect;
            imageCv3Ch.download(imageCv3Ch_cpu);
            imageCv3Ch.download(imageCv3Ch_cpu);
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
            std::vector<AttentionStatus> attentionStatusSet;

            // flushing previous object points
            for (auto & pcl : objPointsO3dPtr_cpu)
                pcl->Clear();

            // Extract points of detected objects
            misc::Timer timerObjPointsExtraction;
            float scale = 0.8; // scaling dimensions by this
            int nClusterOoi = 0;

            objPixelMask  =
                    cv::Mat(imageCv.rows,imageCv.cols,CV_8UC1,cv::Scalar(0));// initialize object mask

            for (const auto& bb : resultBoundingBox){
                if (bb.obj_id == 39 and (nClusterOoi < nMaxOoiClusters)) { // 39 = bottle
                    attentionStatusSet.push_back(AttentionStatus());

                    cv::Range rowRange(
                            bb.y+ bb.h * (1 - scale) / 2.0,bb.y + bb.h - bb.h * (1 - scale) / 2.0 );
                    cv::Range colRange (
                            bb.x+ bb.w * (1 - scale) / 2.0,bb.x + bb.w - bb.w * (1 - scale) / 2.0 );

                    cv::Mat subMat = depthCv_cpu(rowRange,colRange);
                    cv::Mat subMask;
                    cv::MatND histogram;
                    float channel_range[2] = { 0.01 , 5.0 };
                    const float* channel_ranges[1] = { channel_range };
                    int histSize[1] = { 50 };
                    int channel[1] = { 0 };  // Blue
                    cv::calcHist(&subMat,1,channel,cv::Mat(),histogram,1,histSize,channel_ranges);
                    histogram.at<float>(0) = 0 ; // garbage values suppressed
                    auto histDataPtr = (float *) histogram.data;
                    vector<float> histData(histDataPtr, histDataPtr + histSize[0]);
                    int peakIdx = max_element(histData.begin(),histData.end()) - histData.begin();
                    float peakDepth = channel_range[0] +  (channel_range[1]- channel_range[0]) / float(histSize[0]) * peakIdx;

                    cv::inRange(subMat,
                                cv::Scalar (-0.3 + peakDepth),cv::Scalar (0.3+peakDepth),
                                subMask);

                    // visualizing  masked pixel
                    float alpha = 0.5;
                    auto maskDataPtr = (uchar * ) subMask.data;
                    for (int rr = 0 ; rr < rowRange.size() ; rr++)
                        for (int cc = 0; cc < colRange.size(); cc++)
                            if (maskDataPtr[rr *colRange.size() + cc]){
                                // fill 1 to object mask
                                objPixelMask.at<uchar>(rr + rowRange.start,cc + colRange.start)=255;

                                auto& bgr = depthDetectCv_cpu.at<cv::Vec3b>(rr + rowRange.start,
                                                                 cc + colRange.start);
                                int colorIdx = 2; // red
                                bgr(colorIdx) = (1-alpha)*bgr(colorIdx) + alpha * 255;
                            }

                    // computing points of objects
                    bool isPointExistInBox = true;
                    for (int rr = 0 ; rr < rowRange.size() ; rr++)
                        for (int cc = 0; cc < colRange.size(); cc++)
                            if (maskDataPtr[rr *colRange.size() + cc]){
                                float depthVal = depthCv_cpu.at<float>( rr + rowRange.start , cc + colRange.start);
                                Eigen::Vector3d point;
                                point.z() = depthVal;
                                point.x() = (cc + colRange.start - cx) * depthVal / fx;
                                point.y() = (rr + rowRange.start - cy) * depthVal / fy;
                                objPointsO3dPtr_cpu[nClusterOoi]->points_.push_back(point);
                            }
                    objPointsO3dPtr_cpu[nClusterOoi]->PaintUniformColor({0.8,0.2,0.2});
                    Eigen::Vector3d objCenter = objPointsO3dPtr_cpu[nClusterOoi]->GetCenter();
                    objCentersO3dPtr[nClusterOoi]->Translate(objCenter,false);
                    isPointExistInBox = objPointsO3dPtr_cpu[nClusterOoi]->HasPoints();
                    if (not isPointExistInBox){
                        objPointsO3dPtr_cpu[nClusterOoi]->points_.emplace_back(0.1,0.1,0.1);
                        objPointsO3dPtr_cpu[nClusterOoi]->points_.emplace_back(0.11,0.11,0.11);
                    }


                    // Compute attention
                    AttentionStatus attention;
                    if (isPointExistInBox){
                        Eigen::Vector3f objCenterf = objCenter.cast<float>();
                        attention.angleFromGaze = gaze.measureAngleToPoint(objCenterf);
                        attention.distToLeftHand  = (leftHand - objCenter.cast<float>()).norm();
                        attention.distToRightHand = (rightHand - objCenter.cast<float>()).norm();
                        attention.boundingBox = bb;
                        attentionStatusSet[nClusterOoi] = attention;
                    }else{
                        printf("An object is detected in image, but no points were found in bb.\n");
                    }

                    nClusterOoi ++;
                } // if bottle
            } // detected bounding box

            int kernelSize = 10;
//            cv::erode(objPixelMask, objPixelMask,getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize)));
            cv::dilate(objPixelMask, objPixelMask,getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize)));

            // Now we gathered all mask (objects + human). Subtract them
            cv::bitwise_not(objPixelMask,objPixelMask);
            cv::Mat survivedMask;
            cv::bitwise_and(objPixelMask,humanPixelMask,survivedMask);

            o3d_tensor::RGBDImage rgbdImage (imageO3d,depthO3d);

            cv::imshow("mask: objects + human",survivedMask);
            cv::waitKey(1);



            // fill dummy points to suppress warning message (not in the scene graph..)
            for (int nn = nClusterOoi ; nn < nMaxOoiClusters ; nn++){
                objPointsO3dPtr_cpu[nn]->points_.emplace_back(0.1,0.1,0.1);
                objPointsO3dPtr_cpu[nn]->points_.emplace_back(0.11,0.11,0.11);
            }

            draw_attention_box(depthDetectCv_cpu,attentionStatusSet );
            printf("object detection in %.3f ms. \n" ,timerDetect.stop());
            cv::imshow("object detection",depthDetectCv_cpu);
            cv::waitKey(1);

            double elapse = timerObjPointsExtraction.stop();


            // initialize viewport
            if (not isInit) {
                // Dummay operation to initialize cuda scope for .ToLegacy()
                o3d_core::Tensor tensorRgbDummy(
                        (imageCv3Ch_cpu.data), {row, col, 3}, rgbType, device_gpu);
                isInit = true;

                auto pointsCenter = skeletonO3dPtr->GetCenter().cast<float>();
                Eigen::Vector3f eye = pointsCenter + Eigen::Vector3f(0,0,-3);

                o3d_vis::gui::Application::GetInstance().PostToMainThread(
                        vis.get(), [pointsCenter,eye, pointsO3dPtr_cpu, mat](){
                        // this is important! as visible range is determined
                        vis->AddGeometry(cloudName,pointsO3dPtr_cpu, &mat);
                        vis->ResetCameraToDefault();
                        vis->SetupCamera(60,pointsCenter,eye,{0.0, -1.0, 0.0});
                });

                vis->SetLineWidth(2);
            }

            // Draw in the view port
            o3d_vis::gui::Application::GetInstance().PostToMainThread(
                    vis.get(), [mat, matLine,  matAttention, pointsO3dPtr_cpu, isObject](){
                        std::lock_guard<std::mutex> lock (locker);
                        // flushing
                        for (auto geoNames: vis->GetScene()->GetGeometries())
                            vis->RemoveGeometry(geoNames);

                        // coordinates
                        vis->AddGeometry("coordinate",
                                         o3d_legacy::TriangleMesh::CreateCoordinateFrame(0.1)); // add coord at origin
                        vis->AddGeometry(cloudName,pointsO3dPtr_cpu, &mat);

                        if (isObject) {
                            // skeleton
                            vis->AddGeometry(skeletonName, skeletonO3dPtr, &matLine);
                            // attention points (hands + eyes )
                            for (int i = 0; i<4 ;i++)
                                vis->AddGeometry(attentionName[i],attentionPointSet[i],&matAttention);
                            // gaze coordinate & cone
                            *gazeCoordinate = *o3d_legacy::TriangleMesh::CreateCoordinateFrame(0.1);
                            if (gaze.isValid()) {
                                gazeCoordinate->Transform(gaze.getTransformation().cast<double>());
                                vis->AddGeometry(gazeName, gazeCoordinate, &mat);
                            }


                            // detected objects
                            for (int nn = 0; nn < nMaxOoiClusters ; nn++) {
                                // object center point
                                vis->AddGeometry(
                                        "object_points_" + to_string(nn),
                                        objPointsO3dPtr_cpu[nn],&mat);

                                // object pointcloud
                                vis->AddGeometry(
                                        "object_center_" + to_string(nn),
                                        objCentersO3dPtr[nn],&mat);
                            }

                        }

                        // force redraw
                        vis->SetPointSize(1); // this calls ForceRedraw(), readily update the viewport.
                        vis->SetBackground({0,0,0,1});

                    } // lambda function body
            );



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
    string cfgFile = darknetPath + "cfg/yolov4.cfg"; // only tiny fits for fast displaying
    string weightsFile = darknetPath + "yolov4.weights";
    yoloDetectorPtr = new Detector(cfgFile, weightsFile); // take a bit for initialization
    ifstream  file(namesFile);
    for(std::string line; getline(file, line);)
        objectNames.push_back(line);

    // Initialize ZED camera
    sl::InitParameters initParameters;
    sl::RecordingParameters recordingParameters;
    initParameters.sdk_cuda_ctx = (CUcontext)yoloDetectorPtr->get_cuda_context();
    initParameters.sdk_gpu_id = yoloDetectorPtr->cur_gpu_id;
    initParameters.coordinate_units = sl::UNIT::METER;
    bool doRecord = zed_utils::parseArgs(argc, argv, initParameters);
    zed_utils::initCamera(zed, initParameters);
    if (doRecord) {
        signal(SIGINT, signalHandler);
        recordingParameters.compression_mode = SVO_COMPRESSION_MODE::H264;
        auto wallTime = chrono::system_clock::now();
        string timeString = to_string(wallTime.time_since_epoch().count());
        recordingParameters.video_filename = ("/home/jbs/Documents/ZED/attention_record_" + timeString +".svo").c_str();
        auto returned_state = zed.enableRecording(recordingParameters);
        if (returned_state != ERROR_CODE::SUCCESS) {
            zed_utils::print("Recording ZED : ", returned_state,"");
            zed.close();
            return EXIT_FAILURE;
        }
    }
    cv::namedWindow("object detection",cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("object detection", 600,400);

    // Initialize open3d
    meshPtr = std::make_shared<o3d_legacy::TriangleMesh>();
    objPointsO3dPtr = std::make_shared<o3d_tensor::PointCloud>();
    for (int nn = 0 ; nn < nMaxOoiClusters ; nn++) {
        objPointsO3dPtr_cpu[nn] = std::make_shared<o3d_legacy::PointCloud>();
        objBbO3dPtr[nn] = std::make_shared<o3d_legacy::AxisAlignedBoundingBox>();

        // marker type intialize
        objCentersO3dPtr[nn] = std::make_shared<o3d_legacy::TriangleMesh>();
        *objCentersO3dPtr[nn] =
            *(o3d_legacy::TriangleMesh::CreateSphere(0.02,20));
        objCentersO3dPtr[nn]->PaintUniformColor({0,1,0});

    }
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

    if (doRecord)
        zed.disableRecording();
    zed.close();
    cout << "exiting..." << endl;
    return 0;
}
