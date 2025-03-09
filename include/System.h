#pragma once

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <glog/logging.h>

#include <fstream>
#include <condition_variable>

#include <pangolin/pangolin.h>

#include "estimator.h"
#include "parameters.h"
#include "feature_tracker.h"

// IMU的消息结构体
struct IMU_MSG {
  double header;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;
};
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

// 图像的消息结构体
struct IMG_MSG {
  double header;              // 时间戳
  vector<Vector3d> points;    // 特征点坐标
  vector<int> id_of_point;    // 特征点ID
  vector<float> u_of_point;   // 特征点在图像上的坐标
  vector<float> v_of_point;   // 特征点在图像上的坐标
  vector<float> velocity_x_of_point;  // 特征点在图像上的速度
  vector<float> velocity_y_of_point;  // 特征点在图像上的速度
};
typedef std::shared_ptr<IMG_MSG const> ImgConstPtr;

// VIO系统处理类
class System {
 public:
  explicit System(std::string sConfig_files);

  ~System();

  void PubImageData(double dStampSec, cv::Mat &img);

  void PubImuData(double dStampSec,
                  const Eigen::Vector3d &vGyr,
                  const Eigen::Vector3d &vAcc);

  void ProcessBackEnd();

  void Draw();

  pangolin::OpenGlRenderState s_cam;
  pangolin::View d_cam;
  FeatureTracker trackerData[NUM_OF_CAM];

 private:
  // 获取多组配对的IMU和图像数据
  vector<pair<vector<ImuConstPtr>, ImgConstPtr>> getMeasurements() {
    vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;

    while (true) {
      // 数据为空
      if (imu_buf.empty() || feature_buf.empty()) {
        return measurements;
      }

      // 最新的IMU数据小于最早的图像数据
      if (imu_buf.back()->header <= feature_buf.front()->header + estimator.td) {
        sum_of_wait++;
        return measurements;
      }

      // 最早的IMU数据大于最新的图像数据
      if (imu_buf.front()->header >= feature_buf.front()->header + estimator.td) {
        feature_buf.pop();
        continue;
      }

      // 获取最新的图像数据
      ImgConstPtr img_msg = feature_buf.front();
      feature_buf.pop();

      // 获取与图像对应的IMU数据
      vector<ImuConstPtr> IMUs;
      while (imu_buf.front()->header < img_msg->header + estimator.td) {
        IMUs.emplace_back(imu_buf.front());
        imu_buf.pop();
      }

      // 多加入一个IMU数据
      IMUs.emplace_back(imu_buf.front());

      if (IMUs.empty()) {
        cerr << "no imu between two image" << endl;
      }

      measurements.emplace_back(IMUs, img_msg);
    }
  }

  //feature tracker
  std::vector<uchar> r_status;
  std::vector<float> r_err;

  double first_image_time{};
  int pub_count = 1;
  bool first_image_flag = true;
  double last_image_time = 0;
  bool init_pub = 0;

  //estimator
  Estimator estimator;

  std::condition_variable con;
  double current_time = -1;
  std::queue<ImuConstPtr> imu_buf;
  std::queue<ImgConstPtr> feature_buf;
  int sum_of_wait = 0;

  std::mutex m_buf;
  std::mutex m_state;
  std::mutex i_buf;
  std::mutex m_estimator;

  double latest_time{};
  Eigen::Vector3d tmp_P;
  Eigen::Quaterniond tmp_Q;
  Eigen::Vector3d tmp_V;
  Eigen::Vector3d tmp_Ba;
  Eigen::Vector3d tmp_Bg;
  Eigen::Vector3d acc_0;
  Eigen::Vector3d gyr_0;
  bool init_feature = false;
  bool init_imu = 1;
  double last_imu_t = 0;

  bool bStart_backend;
  std::vector<Eigen::Vector3d> vPath_to_draw;

  std::ofstream ofs_pose; // 数据流写出对象 todo: 修改成 CSVUtils管理
};
