#include "System.h"

#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;
using namespace pangolin;

System::System(string sConfig_file_) : bStart_backend(true) {
  // 解析数据
  string sConfig_file = sConfig_file_ + "euroc_config.yaml";
  LOG(INFO) << "VIO系统的配置文件路径: " << sConfig_file;
  readParameters(sConfig_file);

  // 初始化相机光流跟踪的对象
  trackerData[0].readIntrinsicParameter(sConfig_file);

  // 初始化VIO估计器对象
  estimator.setParameter();

  // 初始化数据写出流的对象
  ofs_pose.open("./pose_output.txt", fstream::out);
  if (!ofs_pose.is_open()) {
    LOG(ERROR) << "无法打开 pose_output.txt 文件";
  }

  LOG(INFO) << "VIO 系统化初始化完成.";
}

System::~System() {
  bStart_backend = false;

  pangolin::QuitAll();

  m_buf.lock();
  while (!feature_buf.empty())
    feature_buf.pop();

  while (!imu_buf.empty())
    imu_buf.pop();
  m_buf.unlock();

  m_estimator.lock();
  estimator.clearState();
  m_estimator.unlock();

  ofs_pose.close();
}


// 接收图像数据并组织成图像特征点的形式进行发布
void System::PubImageData(double dStampSec, Mat &img) {
  // 测试数据是否接收完成
  if (!init_feature) {
    LOG(INFO) << "第一帧图像数据接收完成.";
    init_feature = true;
    return;
  }

  // 跳过第一帧图像特征
  if (first_image_flag) {
    LOG(INFO) << "跳过第一次处理，等待第二次图像数据.";
    first_image_flag = false;
    first_image_time = dStampSec;
    last_image_time = dStampSec;
    return;
  }

  // 检查数据流实际否normal
  if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time) {
    cerr << "3 PubImageData image discontinue! reset the feature tracker!" << endl;
    first_image_flag = true;
    last_image_time = 0;
    pub_count = 1;
    return;
  }

  // 频率控制
  // 需要进行重置，如果不进行重置，由于数据量的增大，可能会掩盖当前数据流频率不稳定的问题。
  if (round(1.0 * pub_count / (dStampSec - first_image_time)) <= FREQ) {
    PUB_THIS_FRAME = true;
    if (abs(1.0 * pub_count / (dStampSec - first_image_time) - FREQ) < 0.01 * FREQ) {
      first_image_time = dStampSec;
      pub_count = 0;
    }
  } else {
    PUB_THIS_FRAME = false;
  }

  // 计算当前帧的关键点以及速度
  TicToc tracker_timer;
  trackerData[0].readImage(img, dStampSec);
  LOG(INFO) << "特征点跟踪耗时: " << tracker_timer.toc();

  // 更新最新提取的关键点的id
  for (unsigned int i = 0;; i++) {
    bool completed = false;
    completed |= trackerData[0].updateID(i);
    if (!completed)
      break;
  }

  // 将图像特征数据打包
  if (PUB_THIS_FRAME) {
    pub_count++;
    shared_ptr<IMG_MSG> feature_points(new IMG_MSG());

    // 更新时间戳
    feature_points->header = dStampSec;

    for (int i = 0; i < NUM_OF_CAM; i++) {
      auto &un_pts = trackerData[i].cur_un_pts;
      auto &cur_pts = trackerData[i].cur_pts;
      auto &ids = trackerData[i].ids;
      auto &pts_velocity = trackerData[i].pts_velocity;
      for (unsigned int j = 0; j < ids.size(); j++) {
        if (trackerData[i].track_cnt[j] > 1) {
          int p_id = ids[j];
          feature_points->points.push_back(Vector3d(un_pts[j].x, un_pts[j].y, 1.0));
          feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
          feature_points->u_of_point.push_back(cur_pts[j].x);
          feature_points->v_of_point.push_back(cur_pts[j].y);
          feature_points->velocity_x_of_point.push_back(pts_velocity[j].x);
          feature_points->velocity_y_of_point.push_back(pts_velocity[j].y);
        }
      }

      // skip the first image; since no optical speed on frist image
      if (!init_pub) {
        cout << "4 PubImage init_pub skip the first image!" << endl;
        init_pub = 1;
      } else {
        m_buf.lock();
        feature_buf.push(feature_points);
        m_buf.unlock();
        con.notify_one();
      }
    }
  }

  // 可视化跟踪结果
  if (SHOW_TRACK) {
    cv::Mat show_img;
    cv::cvtColor(img, show_img, cv::COLOR_GRAY2BGR);
    for (unsigned int j = 0; j < trackerData[0].cur_pts.size(); j++) {
      double len = min(1.0, 1.0 * trackerData[0].track_cnt[j] / WINDOW_SIZE);
      cv::circle(show_img, trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }

    cv::namedWindow("IMAGE", cv::WINDOW_AUTOSIZE);
    cv::imshow("IMAGE", show_img);
    cv::waitKey(1);
  }

  last_image_time = dStampSec;
}

// 接收IMU的数据并加入数据缓存
void System::PubImuData(double dStampSec,
                        const Eigen::Vector3d &vGyr,
                        const Eigen::Vector3d &vAcc) {
  shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
  imu_msg->header = dStampSec;
  imu_msg->linear_acceleration = vAcc;
  imu_msg->angular_velocity = vGyr;

  if (dStampSec <= last_imu_t) {
    cerr << "imu message in disorder!" << endl;
    return;
  }
  last_imu_t = dStampSec;

  m_buf.lock();
  imu_buf.push(imu_msg);
  m_buf.unlock();
  con.notify_one();
}



// VIO处理的主流水线
void System::ProcessBackEnd() {
  while (bStart_backend) {
    vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;

    unique_lock<mutex> buffer_mtx(m_buf);
    auto getValidMeasurements = [&]() -> bool {
      return !(measurements = getMeasurements()).empty();
    };
    con.wait(buffer_mtx, getValidMeasurements);
    buffer_mtx.unlock();

    // 获取图像数据然后打印
    LOG(INFO) << "获取的观测数据组的数量: " << measurements.size();
    for (int i = 0; i < measurements.size(); i++) {
      LOG(INFO) << "The " << i << " group measurement";
      LOG(INFO) << " imus timestamp range:"
                << measurements[i].first.front()->header << " ~ "
                << measurements[i].first.back()->header;
      LOG(INFO) << " image timestamp:"
                << measurements[i].second->header;
    }

    // 开始进行VINS的估计
    m_estimator.lock();
    for (auto &measurement : measurements) {
      // 计算补偿后的图像时间戳
      auto img_msg = measurement.second;
      double image_refer_timestamp = img_msg->header + estimator.td;

      // STEP1：遍历IMU的数据进行预积分处理
      double acc_x = 0, acc_y = 0, acc_z = 0, gyr_x = 0, gyr_y = 0, gyr_z = 0;
      for (auto &imu_msg : measurement.first) {
        double imu_refer_timestamp = imu_msg->header;
        // IMU数据中，大部分是小于图像数据时间戳，
        // 但也是有一帧是大于图像时间戳
        if (imu_refer_timestamp <= image_refer_timestamp) {

          // 计算当前IMU数据与上一个IMU数据之间的时间间隔
          if (current_time < 0)
            current_time = imu_refer_timestamp;
          double dt = imu_refer_timestamp - current_time;
          assert(dt >= 0);
          current_time = imu_refer_timestamp;

          // 将数据添加到imu处理pipline
          acc_x = imu_msg->linear_acceleration.x();
          acc_y = imu_msg->linear_acceleration.y();
          acc_z = imu_msg->linear_acceleration.z();
          gyr_x = imu_msg->angular_velocity.x();
          gyr_y = imu_msg->angular_velocity.y();
          gyr_z = imu_msg->angular_velocity.z();
          estimator.processIMU(dt,
                               Vector3d(acc_x, acc_y, acc_z),
                               Vector3d(gyr_x, gyr_y, gyr_z));
        } else {
          double dt_1 = image_refer_timestamp - current_time;
          double dt_2 = imu_refer_timestamp - image_refer_timestamp;
          current_time = image_refer_timestamp;
          assert(dt_1 >= 0);
          assert(dt_2 >= 0);
          assert(dt_1 + dt_2 > 0);

          double w1 = dt_2 / (dt_1 + dt_2);
          double w2 = dt_1 / (dt_1 + dt_2);
          acc_x = w1 * acc_x + w2 * imu_msg->linear_acceleration.x();
          acc_y = w1 * acc_y + w2 * imu_msg->linear_acceleration.y();
          acc_z = w1 * acc_z + w2 * imu_msg->linear_acceleration.z();
          gyr_x = w1 * gyr_x + w2 * imu_msg->angular_velocity.x();
          gyr_y = w1 * gyr_y + w2 * imu_msg->angular_velocity.y();
          gyr_z = w1 * gyr_z + w2 * imu_msg->angular_velocity.z();
          estimator.processIMU(dt_1,
                               Vector3d(acc_x, acc_y, acc_z),
                               Vector3d(gyr_x, gyr_y, gyr_z));
        }
      }

      // STEP2：组织图像数据
      // feature_id  camera_id  [x,y,z,ux,uy,vx,vy]
      map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
      for (unsigned int i = 0; i < img_msg->points.size(); i++) {
        // 计算当前特征点的ID以及对应相机的ID
        int v = img_msg->id_of_point[i] + 0.5;
        int feature_id = v / NUM_OF_CAM;
        int camera_id = v % NUM_OF_CAM;

        double x = img_msg->points[i].x();
        double y = img_msg->points[i].y();
        double z = img_msg->points[i].z();
        double p_u = img_msg->u_of_point[i];
        double p_v = img_msg->v_of_point[i];
        double velocity_x = img_msg->velocity_x_of_point[i];
        double velocity_y = img_msg->velocity_y_of_point[i];
        assert(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
      }

      // STEP3: 进行
      TicToc t_processImage;
      estimator.processImage(image, img_msg->header);

      //
      if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
        Vector3d p_wi;
        Quaterniond q_wi;
        q_wi = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        p_wi = estimator.Ps[WINDOW_SIZE];
        vPath_to_draw.push_back(p_wi);
        double dStamp = estimator.Headers[WINDOW_SIZE];
        cout << "1 BackEnd processImage dt: " << fixed << t_processImage.toc() << " stamp: " << dStamp << " p_wi: "
             << p_wi.transpose() << endl;
        ofs_pose << fixed << dStamp << " " << p_wi(0) << " " << p_wi(1) << " " << p_wi(2) << " "
                 << q_wi.w() << " " << q_wi.x() << " " << q_wi.y() << " " << q_wi.z() << endl;
      }
    }
    m_estimator.unlock();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

// Pangolin绘制可视化结果图
void System::Draw() {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  s_cam = pangolin::OpenGlRenderState(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
      pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
  );

  d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
    glColor3f(0, 0, 1);
    pangolin::glDrawAxis(3);

    // 绘制pos
    glColor3f(0, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINES);
    int nPath_size = vPath_to_draw.size();
    for (int i = 0; i < nPath_size - 1; ++i) {
      glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
      glVertex3f(vPath_to_draw[i + 1].x(), vPath_to_draw[i + 1].y(), vPath_to_draw[i + 1].z());
    }
    glEnd();

    // 绘制点云（如果估计器的求解状态是非线性）
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
      glPointSize(5);
      glBegin(GL_POINTS);
      for (int i = 0; i < WINDOW_SIZE + 1; ++i) {
        Vector3d p_wi = estimator.Ps[i];
        glColor3f(1, 0, 0);
        glVertex3d(p_wi[0], p_wi[1], p_wi[2]);
      }
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }
}
