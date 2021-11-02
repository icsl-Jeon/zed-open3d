//
// Created by jbs on 21. 10. 31..
//

#include <SceneInterpreter.h>



namespace iswy{

    ElapseMonitor::~ElapseMonitor(){
        elapseMeasured = timerPtr->stop();
        delete timerPtr;
        auto it  = monitorResult.find(tag);
        if (it != monitorResult.end()) // exist
            it->second = elapseMeasured;
        else
            monitorResult.insert({tag,elapseMeasured});
        printf("(%s) recorded elapse time %.4f \n", tag.c_str(),elapseMeasured);
    }

    void INThandler(int sig){
        signal(sig, SIG_IGN);
        printf("exiting while loop");
        activeWhile = false;
    }

    map<string,float> ElapseMonitor::monitorResult = map<string,float>();

    /**
     * Initialize the zed params from yaml file
     * @param paramterFilePath
     */
    CameraParam::CameraParam(string paramterFilePath) {
        // todo
        string svoFileDir = "/home/jbs/Documents/ZED/attention_coke.svo";
        initParameters.input.setFromSVOFile(svoFileDir.c_str());
        isSvo = true;

        initParameters.coordinate_units = sl::UNIT::METER;
//    initParameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
        initParameters.depth_mode = sl::DEPTH_MODE::ULTRA;
        initParameters.depth_maximum_distance = 7.0;
        initParameters.depth_minimum_distance = 0.1;

        detectionParameters.detection_model = sl::DETECTION_MODEL::HUMAN_BODY_FAST;
        detectionParameters.enable_tracking = true;
        detectionParameters.enable_body_fitting = true;
        detectionParameters.enable_mask_output = true;

        runtimeParameters.confidence_threshold = 50;
    }


    /**
     * 1. open zed camera from initialization parameter
     * 2. fill intrinsic parameters
     * 3. initialize image data
     * @param zed
     * @return
     */
    bool CameraParam::open(ZedState& zed) {

        // open camera
        auto returned_state = zed.camera.open(initParameters);
        if (returned_state != sl::ERROR_CODE::SUCCESS) {
            printf("Enabling positional tracking failed: \n");
            return false;
        }

        returned_state = zed.camera.enablePositionalTracking();
        if(returned_state != sl::ERROR_CODE::SUCCESS) {
            printf("Enabling positional tracking failed: \n");
            zed.camera.close();
            return false;
        }

        returned_state = zed.camera.enableObjectDetection(detectionParameters);
        if(returned_state != sl::ERROR_CODE::SUCCESS) {
            printf("Enabling object detection failed: \n");
            zed.camera.close();
            return false;
        }

        sl::CameraConfiguration intrinsicParam = zed.camera.getCameraInformation().camera_configuration;
        fx = intrinsicParam.calibration_parameters.left_cam.fx;
        fy = intrinsicParam.calibration_parameters.left_cam.fy;
        cx = intrinsicParam.calibration_parameters.left_cam.cx;
        cy = intrinsicParam.calibration_parameters.left_cam.cy;
        width = zed.camera.getCameraInformation().camera_resolution.width;
        height = zed.camera.getCameraInformation().camera_resolution.height;

        // initialize image matrix
        zed.image.alloc(width,height, sl::MAT_TYPE::U8_C4,  sl::MEM::GPU);
        zed.depth.alloc(width,height, sl::MAT_TYPE::F32_C1, sl::MEM::GPU);

        return true;
    }

    bool ZedState::grab(const CameraParam& runParam) {

        ElapseMonitor monitor("ZedGrab");

        bool isOk;
        auto rtParam = runParam.getRtParam();
        if (camera.grab(rtParam) == sl::ERROR_CODE::SUCCESS) {
            isOk = true;
        } else if (camera.grab(rtParam) == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            printf("SVO reached end. Replay. \n");
            camera.setSVOPosition(1);
            isOk = true;
        } else {
            printf("Grab failed. \n");
            isOk = false;
        }

        // update states
        camera.retrieveImage(image,sl::VIEW::LEFT,sl::MEM::GPU);
        camera.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::GPU);
        camera.retrieveObjects(humans,runParam.getObjRtParam());
        if (not humans.object_list.empty())
            actor = humans.object_list[0];



        return isOk;
    }

    bool ZedState::markHumanPixels(cv::cuda::GpuMat &maskMat) {
        if (humans.object_list.empty())
            return false;
        auto human_bb_min = actor.bounding_box_2d[0];
        auto human_bb_max = actor.bounding_box_2d[2];
        int w = human_bb_max.x - human_bb_min.x;
        int h = human_bb_max.y - human_bb_min.y;

        cv::Mat subMaskCpu = zed_utils::slMat2cvMat(actor.mask);
        cv::cuda::GpuMat subMask; subMask.upload(subMaskCpu);
        cv::Range rowRangeHuman(human_bb_min.y,human_bb_max.y);
        cv::Range colRangeHuman(human_bb_min.x,human_bb_max.x);

        auto subPtr = maskMat(rowRangeHuman,colRangeHuman);
        subMask.copyTo(subPtr);


        return true;
    }

    /**
     * Bind memory in gpu (device)
     */
    void SceneInterpreter::bindDevice() {
        if (not (zedState.camera.isOpened() and (zedState.image.getWidth() > 0) and  (zedState.image.getHeight() > 0) ))
            throw error::ZedException("Initialize zed camera first, then bind!");
        // zed - opencv gpu
        deviceData.depthCv = zed_utils::slMat2cvMatGPU(zedState.depth); // bound buffer
        deviceData.imageCv = zed_utils::slMat2cvMatGPU(zedState.image);
        deviceData.bgDepthCv = cv::cuda::createContinuous(deviceData.depthCv.size(),CV_32FC1); // pixels of obj + human = NAN

        // opencv gpu - open3d
    }

    SceneInterpreter::SceneInterpreter() : zedParam(""){

        // zed init
        zedParam.open(zedState);
        try {
            bindDevice();
        } catch (error::ZedException& caught) {
            cout <<  caught.what() << endl;
        }

        // detector init
        std::ifstream ifs(paramDetect.classNameDir.c_str());
        if (!ifs.is_open())
            CV_Error(cv::Error::StsError, "File " + paramDetect.classNameDir + " not found");
        std::string line;
        while (std::getline(ifs, line))
            paramDetect.classNames.push_back(line);


        net = cv::dnn::readNetFromDarknet(paramDetect.modelConfig,paramDetect.modelWeight);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);


    }


    void SceneInterpreter::cameraThread() {

        signal(SIGINT, INThandler);

        while (activeWhile){
            // initialize fg masks
            deviceData.fgMask = cv::cuda::GpuMat(zedParam.getCvSize(),CV_8UC1,cv::Scalar(0));

            // image + depth + human grab
            {
                lock_guard<mutex> lck(mutex_);
                zedState.grab(zedParam);
                cv::cuda::cvtColor(deviceData.imageCv, deviceData.imageCv3ch, cv::COLOR_BGRA2BGR);
                zedState.markHumanPixels(deviceData.fgMask);
            }

            // compute human attention
            attention.gaze = zed_utils::Gaze(zedState.actor);
            int leftHandIdx = 7, rightHandIdx = 4;
            attention.leftHand = {zedState.actor.keypoint[leftHandIdx].x,
                                  zedState.actor.keypoint[leftHandIdx].y,
                                  zedState.actor.keypoint[leftHandIdx].z};

            attention.rightHand = {zedState.actor.keypoint[rightHandIdx].x,
                                   zedState.actor.keypoint[rightHandIdx].y,
                                   zedState.actor.keypoint[rightHandIdx].z};




            // yolo obj detectors
            detect();



        }
        printf("terminating camera thread. \n");

    }

    void SceneInterpreter::detect() {
        ElapseMonitor elapseYolo("yolo detection");
        // renew detection result
        detectedObjects.clear();

        // preprocess from deviceData.imageCv3ch
        static cv::Mat blob, frame;
        deviceData.imageCv3ch.download(frame);
        cv::dnn::blobFromImage(frame, blob, 1.0, {608,608}, cv::Scalar(), true, false);
        net.setInput(blob,"",1.0/255);

        // inference
        std::vector<cv::Mat> outs;
        std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
        net.forward(outs,outNames);

        // postprocess
        float confThreshold = paramDetect.confidence;
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;


        static std::vector<int> outLayers = net.getUnconnectedOutLayers();
        static std::string outLayerType = net.getLayer(outLayers[0])->type;

        if (outLayerType == "DetectionOutput")
        {
            // Network produces output blob with a shape 1x1xNx7 where N is a number of
            // detections and an every detection is a vector of values
            // [batchId, classId, confidence, left, top, right, bottom]
            CV_Assert(outs.size() > 0);
            for (size_t k = 0; k < outs.size(); k++)
            {
                float* data = (float*)outs[k].data;
                for (size_t i = 0; i < outs[k].total(); i += 7)
                {
                    float confidence = data[i + 2];
                    if (confidence > confThreshold)
                    {
                        int left   = (int)data[i + 3];
                        int top    = (int)data[i + 4];
                        int right  = (int)data[i + 5];
                        int bottom = (int)data[i + 6];
                        int width  = right - left + 1;
                        int height = bottom - top + 1;
                        if (width <= 2 || height <= 2)
                        {
                            left   = (int)(data[i + 3] * frame.cols);
                            top    = (int)(data[i + 4] * frame.rows);
                            right  = (int)(data[i + 5] * frame.cols);
                            bottom = (int)(data[i + 6] * frame.rows);
                            width  = right - left + 1;
                            height = bottom - top + 1;
                        }
                        classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                        boxes.push_back(cv::Rect(left, top, width, height));
                        confidences.push_back(confidence);
                    }
                }
            }
        }
        else if (outLayerType == "Region")
        {
            for (size_t i = 0; i < outs.size(); ++i)
            {
                // Network produces output blob with a shape NxC where N is a number of
                // detected objects and C is a number of classes + 4 where the first 4
                // numbers are [center_x, center_y, width, height]
                float* data = (float*)outs[i].data;
                for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
                {
                    cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                    cv::Point classIdPoint;
                    double confidence;
                    minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                    if (confidence > confThreshold)
                    {
                        int centerX = (int)(data[0] * frame.cols);
                        int centerY = (int)(data[1] * frame.rows);
                        int width = (int)(data[2] * frame.cols);
                        int height = (int)(data[3] * frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        classIds.push_back(classIdPoint.x);
                        confidences.push_back((float)confidence);
                        boxes.push_back(cv::Rect(left, top, width, height));
                    }
                }
            }
        }
        else
            CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

        // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
        // or NMS is required if number of outputs > 1
        if (outLayers.size() > 1 || (outLayerType == "Region" ))
        {
            std::map<int, std::vector<size_t> > class2indices;
            for (size_t i = 0; i < classIds.size(); i++)
            {
                if (confidences[i] >= confThreshold)
                {
                    class2indices[classIds[i]].push_back(i);
                }
            }
            std::vector<cv::Rect> nmsBoxes;
            std::vector<float> nmsConfidences;
            std::vector<int> nmsClassIds;
            for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
            {
                std::vector<cv::Rect> localBoxes;
                std::vector<float> localConfidences;
                std::vector<size_t> classIndices = it->second;
                for (size_t i = 0; i < classIndices.size(); i++)
                {
                    localBoxes.push_back(boxes[classIndices[i]]);
                    localConfidences.push_back(confidences[classIndices[i]]);
                }
                std::vector<int> nmsIndices;
                cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, paramDetect.nmsThreshold, nmsIndices);
                for (size_t i = 0; i < nmsIndices.size(); i++)
                {
                    size_t idx = nmsIndices[i];
                    nmsBoxes.push_back(localBoxes[idx]);
                    nmsConfidences.push_back(localConfidences[idx]);
                    nmsClassIds.push_back(it->first);
                }
            }
            boxes = nmsBoxes;
            classIds = nmsClassIds;
            confidences = nmsConfidences;
        }

        // collect result
        for (size_t idx = 0; idx < boxes.size(); ++idx){
            DetectedObject obj;
            obj.boundingBox = boxes[idx];
            obj.classLabel = classIds[idx];
            obj.confidence = confidences[idx];
            obj.className =paramDetect.classNames[classIds[idx] ];

            detectedObjects.emplace_back(obj);

        }
    }

    void DetectedObject::drawMe(cv::Mat& image, int ooi = -1) const{
        auto& box = boundingBox;
        misc::drawPred(className,confidence, box.x, box.y,
                       box.x + box.width, box.y + box.height, image);
    }

    void AttentionEvaluator::drawMe(cv::Mat &image, CameraParam camParam) {
        // gaze vector
        auto gazeLineSeg = gaze.getGazeLineSeg(0.3);
        cv::Point2f p1 = camParam.project(get<0>(gazeLineSeg));
        cv::Point2f p2 = camParam.project(get<1>(gazeLineSeg));
        uchar r = 77.0 ;
        uchar g = 143.0 ;
        uchar b = 247.0 ;
        cv::arrowedLine(image,p1,p2,{b,g,r},2);

        // hands
        float handRad = 0.08; float circleWidth = 3;

        cv::Point2f p31 = camParam.project(leftHand - Eigen::Vector3f(handRad,0,0));
        cv::Point2f p32 = camParam.project(leftHand +  Eigen::Vector3f(handRad,0,0));
        cv::Point2f p3 = (p31 + p32)/2.0; float rad3 = abs(norm((p31 - p32)));
        cv::circle(image,p3,rad3/2,cv::Scalar(b,g,r),circleWidth);

        cv::Point2f p41 = camParam.project(rightHand - Eigen::Vector3f(handRad,0,0));
        cv::Point2f p42 = camParam.project(rightHand +  Eigen::Vector3f(handRad,0,0));
        cv::Point2f p4 = (p41 + p42)/2.0; float rad4 = abs(norm((p41 - p42)));
        cv::circle(image,p4,rad4/2,cv::Scalar(b,g,r),circleWidth);

    }

    void SceneInterpreter::forwardToVisThread(){
            deviceData.imageCv3ch.download(visOpenCv.image);
            if (not detectedObjects.empty())
                visOpenCv.curObjVis = detectedObjects;


    }

    void SceneInterpreter::visThread() {

        // vis - opencv init
        cv::namedWindow(paramVis.nameImageWindow, cv::WINDOW_KEEPRATIO | cv::WINDOW_OPENGL);
        cv::resizeWindow(paramVis.nameImageWindow, 600, 400);

        while(activeWhile){

            forwardToVisThread();

            if (not visOpenCv.image.empty()) {
                // object
                for (const auto& obj: visOpenCv.curObjVis )
                    if (obj.classLabel == paramAttention.ooi)
                        obj.drawMe(visOpenCv.image);
                // human
                attention.drawMe(visOpenCv.image,zedParam);

                cv::imshow(paramVis.nameImageWindow, visOpenCv.image);
                cv::waitKey(1);
            }
        }
    }



}