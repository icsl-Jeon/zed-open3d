//
// Created by jbs on 21. 9. 5..
//
#include <ZedUtils.h>
#include <Open3dUtils.h>

#include <string>
using namespace std;

void zed_utils::parseArgs(int argc, char **argv,sl::InitParameters& param)
{
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

    }

}

bool zed_utils::initCamera(sl::Camera &zed, sl::InitParameters initParameters) {

    // Parameter setting
    initParameters.coordinate_units = sl::UNIT::METER;
    initParameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    initParameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    initParameters.depth_maximum_distance = 5.0;

    sl::ObjectDetectionParameters detectionParameters;
    detectionParameters.detection_model = sl::DETECTION_MODEL::HUMAN_BODY_MEDIUM;
    detectionParameters.enable_tracking = false;
    detectionParameters.enable_body_fitting = true;


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
