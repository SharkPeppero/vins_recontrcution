#include "parameters.h"

using namespace std;
using namespace cv;

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

vector<Eigen::Matrix3d> RIC;
vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};
std::string MODEL_TYPE;
std::string CAMERA_NAME;
std::string OUTPUT_PATH;
double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
string EX_CALIB_RESULT_PATH;
string VINS_RESULT_PATH;
// string IMU_TOPIC;
double ROW, COL;
double TD, TR;


int FOCAL_LENGTH;
string IMAGE_TOPIC;
string IMU_TOPIC;
string FISHEYE_MASK;
vector<string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
// int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
bool STEREO_TRACK;
int EQUALIZE;
int FISHEYE;
bool PUB_THIS_FRAME;


void readParameters(string config_file) {
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "1 readParameters ERROR: Wrong path to settings!" << endl;
        return;
    }

    LOG(INFO) << "Common Parameters:";
    fsSettings["imu_topic"] >> IMU_TOPIC;
    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.txt";
    LOG(INFO) << "  imu_topic: " << IMU_TOPIC;
    LOG(INFO) << "  image_topic: " << IMAGE_TOPIC;
    LOG(INFO) << "  output_path: " << OUTPUT_PATH;
    LOG(INFO) << "  vio_res_out_path: " << VINS_RESULT_PATH;



    LOG(INFO) << "Camera Calibration :";
    fsSettings["model_type"] >> MODEL_TYPE;
    fsSettings["camera_name"] >> CAMERA_NAME;
    fsSettings["image_width"] >> COL;
    fsSettings["image_height"] >> ROW;



    LOG(INFO) << "Extrinsic parameter between IMU and Camera:";
    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2) {
        LOG(WARNING) << "no prior about extrinsic param, calibrate extrinsic param";
        RIC.emplace_back(Eigen::Matrix3d::Identity());
        TIC.emplace_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
    } else {
        if (ESTIMATE_EXTRINSIC == 1) {
            LOG(WARNING) << "Optimize extrinsic param around initial guess!";
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0) {
            LOG(WARNING) << "fix extrinsic param";
        }
        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
    }



    LOG(INFO) << "feature traker paprameters:";
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    FOCAL_LENGTH = fsSettings["focal_length"];



    LOG(INFO) << "optimization parameters";
    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;


    LOG(INFO) << "imu parameters:";
    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];


    LOG(INFO) << "loop closure parameters:";


    LOG(INFO) << "unsynchronization parameters:";
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];

    LOG(INFO) << "rolling shutter parameters:";
    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
        TR = fsSettings["rolling_shutter_tr"];
    else
        TR = 0;

    LOG(INFO) << "visualization parameters:";

    // 暂时未使用的参数
    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;
    // WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    PUB_THIS_FRAME = false;

    fsSettings.release();

    // 初始化相机对象
    CAM_NAMES.push_back(config_file);
}
