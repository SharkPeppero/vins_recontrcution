
#include <unistd.h>

#include <iostream>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Dense>
#include "System.h"

#include <glog/logging.h>

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
const std::string reference_path = "/home/westwell/AJbin/SelfLearning/vins_recontrcution/";
DEFINE_string(sData_path, reference_path + "data/MH_05_difficult/mav0/", "EuRoC数据集路径");
DEFINE_string(sConfig_path, reference_path + "config/", "EuRoC数据集配置文件");
DEFINE_string(sDebugDir, reference_path, "DEBUG存储的数据路径");

int main(int argc, char **argv) {

    // 初始化GLOG参数
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = reference_path + "log";
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "数据集的路径: ";
    LOG(INFO) << "\tData_path: " << FLAGS_sData_path;
    LOG(INFO) << "\tConfig_path: " << FLAGS_sConfig_path;

    std::string debug_dir = std::string(DEBUGDIR);
    LOG(INFO) << "\tdebug_dir: " << debug_dir;

    std::shared_ptr<System> pSystem = std::make_shared<System>(FLAGS_sConfig_path);
    std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);

    // 读取配置文件的imu数据，数据传入到 VIO-System
    auto PubImuData = [&pSystem]() {
        // IMU的路径
        string sImu_data_file = FLAGS_sConfig_path + "MH_05_imu0.txt";
        LOG(INFO) << "imu txt: " << sImu_data_file;

        ifstream fsImu;
        fsImu.open(sImu_data_file.c_str());
        if (!fsImu.is_open()) {
            cerr << "Failed to open imu file! " << sImu_data_file << endl;
            return;
        }

        std::string sImu_line;
        double dStampNSec = 0.0;
        Vector3d vAcc;
        Vector3d vGyr;
        while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) {
            std::istringstream ssImuData(sImu_line);
            ssImuData >> dStampNSec >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
            pSystem->PubImuData(dStampNSec / 1e9, vGyr, vAcc);
            usleep(5000 * nDelayTimes);
        }
        fsImu.close();
        LOG(INFO) << "完成IMU的读取";
    };

    // 读取图像数据，数据传入 VIO-System
    auto PubImageData = [&pSystem]() {
        string sImage_file = FLAGS_sConfig_path + "MH_05_cam0.txt";
        LOG(INFO) << "image txt: " << sImage_file;

        ifstream fsImage;
        fsImage.open(sImage_file.c_str());
        if (!fsImage.is_open()) {
            cerr << "Failed to open image file! " << sImage_file << endl;
            return;
        }

        std::string sImage_line;
        double dStampNSec;
        string sImgFileName;
        uint64_t image_nums = 0;
        while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {
            std::istringstream ssImuData(sImage_line);
            ssImuData >> dStampNSec >> sImgFileName;

            string imagePath = FLAGS_sData_path + "cam0/data/" + sImgFileName;
            Mat img = imread(imagePath.c_str(), IMREAD_GRAYSCALE);
            if (img.empty()) {
                cerr << "image is empty! path: " << imagePath << endl;
                return;
            }
            pSystem->PubImageData(dStampNSec / 1e9, img);
            usleep(50000 * nDelayTimes);
        }
        fsImage.close();
    };

    // 设置三个线程并行，主线程依次等待线程完成
    std::thread thd_PubImuData(PubImuData);       // 按照固定频率发布IMU数据
    std::thread thd_PubImageData(PubImageData);   // 按照固定频率发布图像数据
    std::thread thd_Draw(&System::Draw, pSystem); // 绘制

    thd_PubImuData.join();
    thd_PubImageData.join();
    thd_Draw.join();

    LOG(INFO) << "程序执行结束...";
    gflags::ShutDownCommandLineFlags();
    return 0;
}
