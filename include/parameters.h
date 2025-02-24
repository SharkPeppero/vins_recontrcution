#pragma once

#include <vector>
#include <fstream>
#include "utility/utility.h"
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "glog/logging.h"

// Common Parameters
extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string OUTPUT_PATH;

// Camera Calibration
extern std::string MODEL_TYPE;
extern std::string CAMERA_NAME;
extern double COL;
extern double ROW;
extern std::vector<double> distortion_parameters;
extern std::vector<double> projection_parameters;

// Extrinsic parameter between IMU and Camera
extern std::string EX_CALIB_RESULT_PATH;



// feature traker paprameters:
const int NUM_OF_CAM = 1;
extern int FOCAL_LENGTH;


// optimization parameters

// imu parameters

// loop closure parameters

// unsynchronization parameters


extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern bool STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

//estimator
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;

extern std::string VINS_RESULT_PATH;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;


void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION {
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder {
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder {
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
