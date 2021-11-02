//
// Created by jbs on 21. 9. 5..
//
#include <ZedUtils.h>
#include <Open3dUtils.h>

#include <string>
using namespace std;
using namespace zed_utils;

bool zed_utils::parseArgs(int argc, char **argv,sl::InitParameters& param)
{
    bool doRecord = false;
    if (argc > 1 && string(argv[1]).find(".svo")!=string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout<<"[Sample] Using SVO File input: "<<argv[1]<<endl;
    } else if (argc > 1 && string(argv[1]).find(".svo")==string::npos) {
        string arg = string(argv[1]);
        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()),port);
            cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<<endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout<<"[Sample] Using Stream input, IP : "<<argv[1]<<endl;
        }
        else if (arg.find("HD2K")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            cout<<"[Sample] Using Camera in resolution HD2K"<<endl;
        } else if (arg.find("HD1080")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            cout<<"[Sample] Using Camera in resolution HD1080"<<endl;
        } else if (arg.find("HD720")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            cout<<"[Sample] Using Camera in resolution HD720"<<endl;
        } else if (arg.find("VGA")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            cout<<"[Sample] Using Camera in resolution VGA"<<endl;
        }
    } else {
        // Default initialization

        doRecord = true;
    }
    return doRecord;
}

bool zed_utils::initCamera(sl::Camera &zed, sl::InitParameters initParameters) { // will be deprecated


    // Parameter setting
    initParameters.coordinate_units = sl::UNIT::METER;
//    initParameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    initParameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    initParameters.depth_maximum_distance = 7.0;
    initParameters.depth_minimum_distance = 0.1;

    sl::ObjectDetectionParameters detectionParameters;
    detectionParameters.detection_model = sl::DETECTION_MODEL::HUMAN_BODY_MEDIUM;
    detectionParameters.enable_tracking = true;
    detectionParameters.enable_body_fitting = true;
    detectionParameters.enable_mask_output = true;


    // Enabling functions
    auto returned_state = zed.open(initParameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling positional tracking failed: \n");
        return false;
    }

    returned_state = zed.enablePositionalTracking();
    if(returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling positional tracking failed: \n");
        zed.close();
        return false;
    }

    returned_state = zed.enableObjectDetection(detectionParameters);
    if(returned_state != sl::ERROR_CODE::SUCCESS) {
        printf("Enabling object detection failed: \n");
        zed.close();
        return false;
    }


    return true;
}

// Mapping between MAT_TYPE and CV_TYPE
int zed_utils::getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

cv::Mat zed_utils::slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(),
                   zed_utils::getOCVtype(input.getDataType()),
                   input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}


cv::cuda::GpuMat zed_utils::slMat2cvMatGPU(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(),
                            zed_utils::getOCVtype(input.getDataType()),
                            input.getPtr<sl::uchar1>(sl::MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}

open3d::geometry::RGBDImage createFromZedImage(const cv::Mat& image, const cv::Mat& depth){
    open3d::geometry::Image imageO3d, depthO3d;
    imageO3d.Prepare(image.cols, image.rows, 3, 1);
    depthO3d.Prepare(depth.cols, depth.rows, 1, 4);
    memcpy(imageO3d.data_.data(), image.data, imageO3d.data_.size());
    memcpy(depthO3d.data_.data(), depth.data, depthO3d.data_.size());
    return *open3d::geometry::RGBDImage::CreateFromColorAndDepth(imageO3d, depthO3d);
}

Gaze::Gaze(const sl::ObjectData &humanObject) {
    // todo transformation was not considered
    auto keypoint = humanObject.keypoint;
    sl::float3 landMarks[3] = {keypoint[int(sl::BODY_PARTS::LEFT_EYE)],
                               keypoint[int(sl::BODY_PARTS::RIGHT_EYE)],
                               keypoint[int(sl::BODY_PARTS::NOSE)]};
    vector<Eigen::Vector3f> landMarksVec(3);

    // root
    root.setZero();
    for (int i = 0; i < 3 ; i++){
        landMarksVec[i] = Eigen::Vector3f(landMarks[i].x,
                                          landMarks[i].y,
                                          landMarks[i].z);
        root+=landMarksVec[i];
        }
    root /= 3.0;

    // direction
    Eigen::Vector3f nose2eyes[2] = {
                                    landMarksVec[0] - landMarksVec[2], // left
                                    landMarksVec[1] - landMarksVec[2] // right
                                    };
    direction = nose2eyes[0].cross(nose2eyes[1]);
    direction.normalize();

    // transformation
    Eigen::Vector3f ex = (landMarksVec[1] - landMarksVec[0]); ex.normalize();
    Eigen::Vector3f ez = direction;
    Eigen::Vector3f ey = ez.cross(ex); ey.normalize();
    transformation = Eigen::Matrix4f::Identity();
    transformation.block(0,3,3,1) = root; // translation
    transformation.block(0,2,3,1) = direction; // ez
    transformation.block(0,0,3,1) = ex; // from left to right
    transformation.block(0,1,3,1) = ey;

    // down from normal vector of eyes - nose
    float noseDownAngle = 45 * M_PI / 180.0;
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::Matrix2f R;
    R << cos(noseDownAngle) , sin(noseDownAngle) ,
         -sin(noseDownAngle) , cos(noseDownAngle);
    T.block(1,1,2,2) = R;
    transformation = transformation * T;
    direction = transformation.block(0,2,3,1);
}

Eigen::Matrix4f Gaze::getTransformation() const {
    return transformation;
}


bool Gaze::isValid()  const {
    return not (isinf(transformation.norm()) or isnan(transformation.norm())) ;
}

float Gaze::measureAngleToPoint(const Eigen::Vector3f &point) const {
    if (not this->isValid()){
        printf("[Gaze] gaze is not valid. But angle-measure requested. Returning inf\n");
        return INFINITY;
    }

    Eigen::Vector3f viewVector = (point - root); viewVector.normalize();
    return atan2(viewVector.cross(direction).norm(), viewVector.dot(direction));
}


tuple<Eigen::Vector3f,Eigen::Vector3f> Gaze::getGazeLineSeg(float length)  const{
    auto p1 = root;
    auto p2 = p1 + direction * length;
    return make_tuple(p1,p2);

}; // p1 ~ p2




void zed_utils::print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {

    cout <<"[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}