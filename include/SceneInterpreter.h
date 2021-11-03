//
// Created by jbs on 21. 10. 31..
//

// todo yaml parameterize

#ifndef ZED_OPEN3D_SCENEINTERPRETER_H
#define ZED_OPEN3D_SCENEINTERPRETER_H


#include <ZedUtils.h>
#include <Misc.h>
#include <mutex>


using namespace std;
namespace o3d_tensor =  open3d::t::geometry ;
namespace o3d_legacy = open3d::geometry;
namespace o3d_core = open3d::core;
namespace o3d_vis = open3d::visualization;

static bool activeWhile = true;


namespace iswy { // i see with you
    struct ElapseMonitor {
        static map<string,float> monitorResult;
        float elapseMeasured =-1 ;
        string tag;
        misc::Timer* timerPtr;

        ElapseMonitor(string tag): tag(tag) {
            timerPtr = new misc::Timer();
        }
        ~ElapseMonitor();
    };


    void INThandler(int sig);

    struct SceneParam {


    };
    // visualization with open3d and opencv ...
    struct VisParam{
        string nameImageWindow = "image perception";


    };


    struct CameraParam;

    struct ZedState{ // raw camera state

        sl::Pose pose;
        sl::Mat image;
        sl::Mat depth;
        sl::Objects humans;
        sl::ObjectData actor;

        sl::Camera camera;


        bool grab(const CameraParam & runParam);
        bool markHumanPixels (cv::cuda::GpuMat& maskMat);

    };

    struct CameraParam{
    private:
        bool isSvo = true;
        // these will be filled once camera opened !
        float fx;
        float fy;
        float cx;
        float cy;
        int width;
        int height;
        sl::InitParameters initParameters;
        sl::ObjectDetectionParameters detectionParameters;
        sl::RuntimeParameters runtimeParameters;
        sl::ObjectDetectionRuntimeParameters objectDetectionRuntimeParameters;

    public:
        CameraParam(string paramterFilePath);
        bool open(ZedState& zed);
        sl::RuntimeParameters getRtParam() const {return runtimeParameters;};
        sl::ObjectDetectionRuntimeParameters getObjRtParam() const {return objectDetectionRuntimeParameters;}
        cv::Size getCvSize() const {return cv::Size(width,height);};
        cv::Point2f project(const Eigen::Vector3f& pnt) const {return {fx*pnt.x()/pnt.z() + cx, fx*pnt.y()/pnt.z() + cy};};
    };




    struct AttentionParam{
        int ooi = 39; // bottle
        float gazeFov = M_PI /3.0 *2.0; // cone (height/rad) = tan(gazeFov/2)

    };


    struct ObjectDetectParam{
        string rootDir = "/home/jbs/catkin_ws/src/zed-open3d/param/";
        string modelConfig = rootDir + "yolov4.cfg";
        string modelWeight = rootDir + "yolov4.weights";
        string classNameDir = rootDir + "coco.names";
        vector<string> classNames;
        float confidence = 0.2;
        float nmsThreshold = 0.2;
        float objectDepthMin = 0.01;
        float objectDepthMax = 5.0;

    };

    struct humanDetectParam{

    };

    struct DetectedObject{
        cv::Rect boundingBox;
        cv::cuda::GpuMat mask; // 255 masking
        int classLabel;
        string className;
        float confidence ;
        Eigen::Vector3f centerPoint;

        void drawMe (cv::Mat& image, int ooi) const;
        float attentionCost = INFINITY;
        void updateAttentionCost (float cost) {attentionCost = cost; }

    };

    struct AttentionEvaluator{
        zed_utils::Gaze gaze;
        Eigen::Vector3f leftHand; // left hand location
        Eigen::Vector3f rightHand; // right hand location
        float evalAttentionCost(DetectedObject& object, bool updateObject = true);
        bool isValid() {return (not isnan(leftHand.norm()) and (not isnan(leftHand.norm())) and gaze.isValid()); }
        void drawMe (cv::Mat& image, CameraParam camParam); // todo extrinsic
    };


    struct VisOpen3d{

        shared_ptr<o3d_legacy::TriangleMesh> attentionPointSet[4]; // left eye, right eye, left wrist, right wrist
        shared_ptr<o3d_legacy::TriangleMesh> gazeCoordinate;
    };

    struct VisOpenCv{
        cv::Mat image;
        vector<DetectedObject> curObjVis; // renewed from update thread


    };

    struct DeviceData{
        // raw image bound with ZED
        cv::cuda::GpuMat imageCv;
        cv::cuda::GpuMat imageCv3ch;
        cv::cuda::GpuMat depthCv;
        cv::cuda::GpuMat fgMask; // {human, objects} = 255
        cv::cuda::GpuMat bgDepthCv; // depth - {human, objects}

    };


    class SceneInterpreter {
    private:
        mutex mutex_; // mutex between cameraThread and vis thread
        // visualizers
        VisOpen3d visOpen3d;
        VisOpenCv visOpenCv;
        VisParam paramVis;

        // detector
        ObjectDetectParam paramDetect;
        cv::dnn::Net net;
        vector<DetectedObject> detectedObjects;
        void detect();

        // zed
        CameraParam zedParam;
        ZedState zedState;

        // attention
        AttentionParam paramAttention;
        AttentionEvaluator attention;

        // variables in device
        DeviceData deviceData;

        void bindDevice();
        void forwardToVisThread();



    public:
        SceneInterpreter();
        void cameraThread();
        void visThread();
        ~SceneInterpreter() {};
    };

}
#endif //ZED_OPEN3D_SCENEINTERPRETER_H


