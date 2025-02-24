#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>

#include "../factor/integration_base.h"
#include "../utility/utility.h"
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame {
public:
    ImageFrame() {};

    ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, double _t) : t{_t}, is_key_frame{false} {
        points = _points;
    };
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points;
    double t;           // 时间戳
    Matrix3d R;         // orient
    Vector3d T;         // position
    IntegrationBase *pre_integration; // 预积分器结果
    bool is_key_frame;                // 是否为关键帧
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x);