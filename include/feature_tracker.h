#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "utility/tic_toc.h"

#define TimeCostDebug

using namespace std;
using namespace camodocal;
using namespace Eigen;

/**
 * @brief 特征提取的Pipline
 *
 */

class FeatureTrackerOptions {
    bool image_equalize = true;

    int lk_window_size = 21;
    int lk_max_level = 3;

    int ransac_focal_length = 460;
    double ransac_project_threshold = 3.0;
    double ransac_confidence = 0.99;

    int mask_circle_radius = 30;

    bool show_track = true;


};


class FeatureTracker {
public:
    FeatureTracker() = default;

    ~FeatureTracker() = default;

    void readImage(const cv::Mat &_img, double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();


    camodocal::CameraPtr m_camera;
    cv::Mat mask;

    cv::Mat prev_img;
    cv::Mat cur_img;
    cv::Mat forw_img;
    vector<cv::Point2f> prev_pts;
    vector<cv::Point2f> cur_pts;
    vector<cv::Point2f> forw_pts;

    vector<int> ids;        // 最新图像上的关键点id
    vector<int> track_cnt;  // 最新图像上关键点历史被跟踪的次数


    vector<cv::Point2f> n_pts;

    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    vector<cv::Point2f> pts_velocity;


    double cur_time;
    double prev_time;

    uint64_t n_id;
    cv::Mat fisheye_mask; // 鱼眼相机原图掩码

private:
    static bool inBorder(const cv::Point2f &pt) {
        const int BORDER_SIZE = 1;
        int img_x = cvRound(pt.x);
        int img_y = cvRound(pt.y);
        return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
    }

    static void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
        int j = 0;
        for (int i = 0; i < int(v.size()); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }

    static void reduceVector(vector<int> &v, vector<uchar> status) {
        int j = 0;
        for (int i = 0; i < int(v.size()); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }
};
