#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_geometry/pinhole_camera_model.h>  // カメラモデルを利用するため
#include <image_transport/image_transport.h>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>  // ROSを使用するときは必ずincludeする。
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

static const float SIZE_BOX = 0.0285;
static const float EST_RESO = 0.001;

static int int_method;

class BlockFinder {
  ros::NodeHandle nh_;

  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

  image_geometry::PinholeCameraModel cam_model_;

  ros::Subscriber info_sub_;
  ros::Publisher pub_position_image_;
  ros::Publisher pub_position_space_;
  ros::Publisher pub_point_pattern_;
  ros::Publisher pub_point_world_;
  ros::Publisher pub_block_size_;

  tf::TransformListener tf_listener_;
  tf::TransformBroadcaster tf_broadcaster_;
  tf::Transform target_transform;

  std::string fixed_frame;
  std::string camera_frame;
  std::string target_frame;

  cv::Matx33d mat_k_;             // カメラの内部パラメーター
  cv::Mat mat_d_;                 // 歪み係数
  cv::Mat mat_tvec_;              // 並進ベクトル
  cv::Mat mat_rvec_;              // 回転行列
  cv::Mat ideal_points_;          // チェッカーボードの点の座標
  cv::Point3f point3f_area_max_;  // チェッカーボードの端の座標

  cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG2;

  cv::Mat mat_img_result;

  int int_max_area_;

  int int_thre_bin_;

  bool is_param_;

 public:
  void convertCVtoEigen(const cv::Mat& mat_tvec_, const cv::Mat& R,
                        Eigen::Vector3f& translation,
                        Eigen::Quaternionf& orientation) {
    // This assumes that cv::Mats are stored as doubles. Is there a way to check
    // this?
    // Since it_'s templated...
    translation =
        Eigen::Vector3f(mat_tvec_.at<double>(0, 0), mat_tvec_.at<double>(0, 1),
                        mat_tvec_.at<double>(0, 2));

    Eigen::Matrix3f Rmat;
    Rmat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

    orientation = Eigen::Quaternionf(Rmat);
  }

  // 基準となるチェッカーボードの３次元位置の計算
  cv::Mat calcChessboardCorners(cv::Size boardSize, float squareSize,
                                cv::Point3f offset) {
    cv::Mat corners;

    for (int i = 0; i < boardSize.height; i++) {
      for (int j = 0; j < boardSize.width; j++) {
        corners.push_back(cv::Point3f(j * squareSize, i * squareSize, 0.0f) +
                          offset);
      }
    }

    point3f_area_max_ = cv::Point3f((boardSize.width - 1) * squareSize,
                                    (boardSize.height - 1) * squareSize, 0.0f);

    return corners;
  }

  // カメラ情報の取得
  void infoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg) {
    cam_model_.fromCameraInfo(info_msg);

    mat_k_ = cam_model_.intrinsicMatrix();
    mat_d_ = cam_model_.distortionCoeffs();
  }

  BlockFinder() : it_(nh_) {
    ROS_INFO("OpenCV Version: %d.%d", CV_MAJOR_VERSION, CV_MINOR_VERSION);

    info_sub_ = nh_.subscribe("/camera/camera_info", 1,
                              &BlockFinder::infoCallback, this);
    image_sub_ =
        it_.subscribe("/camera/image_raw", 1, &BlockFinder::imageCb, this);

    image_pub_ = it_.advertise("/block_finder/image_block", 1);
    pub_position_image_ =
        nh_.advertise<geometry_msgs::Pose2D>("/block_finder/pose_image", 1);
    pub_position_space_ =
        nh_.advertise<geometry_msgs::Pose2D>("/block_finder/pose", 1);
    pub_point_pattern_ = nh_.advertise<geometry_msgs::PointStamped>(
        "/block_finder/pose_point_pattern", 1);
    pub_point_world_ = nh_.advertise<geometry_msgs::PointStamped>(
        "/block_finder/pose_point_world", 1);
    pub_block_size_ =
        nh_.advertise<std_msgs::Int32>("/block_finder/block_size_max", 1);

    nh_.param<std::string>("fixed_frame", fixed_frame, "/world");
    nh_.param<std::string>("camera_frame", camera_frame, "/camera_link");
    nh_.param<std::string>("target_frame", target_frame, "/pattern_link");

    ROS_INFO("Method %d selected!", int_method);

    cv::namedWindow("Original");
    cv::namedWindow("Result");
    cv::moveWindow("Original", 0, 0);
    cv::moveWindow("Result", 0, 550);  // 640, 0

    int_thre_bin_ = 150;
    cv::createTrackbar("Subtracter", "Result", &int_thre_bin_, 255);

    // ３次元位置を求める。
    ideal_points_ = calcChessboardCorners(cv::Size(8, 6), SIZE_BOX,
                                          cv::Point3f(0.0, 0.0, 0.0));

    pMOG2 = cv::createBackgroundSubtractorMOG2();

    int_max_area_ = 0;

    is_param_ = false;
  }

  ~BlockFinder() {
    cv::destroyWindow("Original");
    cv::destroyWindow("Result");
  }

  //===========================================================================
  geometry_msgs::Pose2D rsjImageProcessing(cv::Mat& mat_img_input_c,
                                           cv::Mat& mat_img_input_g,
                                           int method = 0) {
    geometry_msgs::Pose2D pose2d_block_image;
    pose2d_block_image.x = 0.0f;
    pose2d_block_image.y = 0.0f;
    pose2d_block_image.theta = 0.0f;

    if (method == 1) {
      // 平滑化する。
      cv::Mat mat_img_bin, mat_img_bin_temp;
      cv::GaussianBlur(mat_img_input_g, mat_img_input_g, cv::Size(9, 9), 0,
                       0);  // 両方とも正の奇数

      // 二値化する。
      cv::threshold(mat_img_input_g, mat_img_bin, int_thre_bin_, 255,
                    cv::THRESH_BINARY);

      // 輪郭を求める。
      std::vector<std::vector<cv::Point> > contours;  // 輪郭を表現するベクトル
      mat_img_bin_temp = mat_img_bin.clone();
      cv::findContours(mat_img_bin_temp, contours, CV_RETR_EXTERNAL,
                       CV_CHAIN_APPROX_NONE);

      // 各輪郭の面積を求める。
      if (contours.empty() != true) {
        int_max_area_ = 0;
        int max_area_contour = -1;
        for (int i = 0; i < contours.size(); i++) {
          polylines(mat_img_input_c, contours.at(i), true,
                    cv::Scalar(0, 255, 0), 2);

          double area = cv::contourArea(contours.at(i));
          if (area > int_max_area_) {
            int_max_area_ = area;
            max_area_contour = i;
          }
        }

        // 重心を求める。
        if (max_area_contour != -1) { // 輪郭が点で表現されることがある。
          int count = contours.at(max_area_contour).size();
          double x = 0.0;
          double y = 0.0;
          for (int i = 0; i < count; i++) {
            x += contours.at(max_area_contour).at(i).x;
            y += contours.at(max_area_contour).at(i).y;
          }
          x /= count;
          y /= count;
          circle(mat_img_input_c, cv::Point(x, y), 8, cv::Scalar(0, 0, 255), -1,
                 CV_AA);

          pose2d_block_image.x = x;
          pose2d_block_image.y = y;
        }
      }
      mat_img_result = mat_img_bin.clone();
    } else if (method == 2) {
      cv::Mat mat_img_mask, mat_img_blur, mat_img_mog;
      cv::Mat mat_img_mask_bin;

      cv::GaussianBlur(mat_img_input_c, mat_img_blur, cv::Size(9, 9), 0, 0);
      pMOG2->apply(mat_img_blur, mat_img_mask);
      // オープニング処理でノイズを除去する。
      cv::morphologyEx(mat_img_mask, mat_img_mask, cv::MORPH_OPEN, cv::Mat(),
                       cv::Point(-1, -1), 3);
      // ビット毎の論理積を求める。
      cv::bitwise_and(mat_img_input_c, mat_img_input_c, mat_img_mog,
                      mat_img_mask);

      cv::threshold(mat_img_mask, mat_img_mask_bin, int_thre_bin_, 255,
                    cv::THRESH_BINARY);

      // ラベリングを行う。OpenCV3.0から
      cv::Mat LabelImg;
      cv::Mat stats;
      cv::Mat centroids;
      int nLab = cv::connectedComponentsWithStats(mat_img_mask_bin, LabelImg,
                                                  stats, centroids);

      // 最大面積を求める。
      int_max_area_ = 0;
      int int_max_area_num = -1;
      for (int i = 1; i < nLab; ++i) {
        int* param = stats.ptr<int>(i);

        if (param[4] > int_max_area_) {
          int_max_area_ = param[4];
          int_max_area_num = i;
        }
      }

      // 最大面積を有するクラスターの中心点を求める。
      if (int_max_area_num != -1) {
        int* param = stats.ptr<int>(int_max_area_num);

        cv::Point pose2d_block_disp;
        pose2d_block_disp.x = pose2d_block_image.x =
            static_cast<int>(param[0] + param[2] / 2);
        pose2d_block_disp.y = pose2d_block_image.y =
            static_cast<int>(param[1] + param[3] / 2);

        circle(mat_img_input_c, pose2d_block_disp, 8, cv::Scalar(0, 0, 255), -1,
               CV_AA);
      }

      mat_img_result = mat_img_mask_bin.clone();
    } else {
      ROS_WARN("No method selected!");
    }

    return pose2d_block_image;
  }
  //===========================================================================

  void imageCb(const sensor_msgs::ImageConstPtr& msg) {
    bool is_block = false;

    geometry_msgs::Pose2D pose2d_block_image;  // 画像のブロックの位置
    geometry_msgs::PointStamped
        point_block_pattern;  // 空間のブロックの位置（pattern座標系）
    geometry_msgs::PointStamped
        point_block_world;  // 空間のブロックの位置（world座標系）

    cv::Size patternsize(8, 6);
    cv::Mat board_corners;

    cv_bridge::CvImagePtr cv_img_ptr;
    cv::Mat mat_img_color;
    try {
      cv_img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      mat_img_color = cv_img_ptr->image.clone();
    } catch (cv_bridge::Exception& ex) {
      ROS_ERROR("cv_bridge exception:\n%s", ex.what());
      return;
    }

    // グレースケール変換
    cv::Mat mat_img_gray;
    cvtColor(mat_img_color, mat_img_gray, CV_BGR2GRAY);

    if (!is_param_) {
      // チェスボード検出
      bool patternfound = findChessboardCorners(
          mat_img_gray, patternsize, board_corners,
          cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
              cv::CALIB_CB_FAST_CHECK);
      if (patternfound) { // 検出できた場合
        cv::drawChessboardCorners(mat_img_gray, patternsize, board_corners,
                                  patternfound);
        try {
          tf::StampedTransform base_transform;

          Eigen::Vector3f translation(0.0f, 0.0f, 0.0f);
          Eigen::Quaternionf orientation(0.0f, 0.0f, 0.0f, 0.0f);

          cv::Mat observation_points_ = board_corners;
          cv::Mat R;

          // 画像上の２次元位置と空間内の３次元位置を対応付ける。
          if (!cv::solvePnP(ideal_points_, observation_points_, mat_k_, mat_d_,
                            mat_rvec_, mat_tvec_, false)) {
            ROS_ERROR("Failed to calculate chessboard position");
            return;
          }
          // 回転ベクトルを回転行列へ変換する。
          cv::Rodrigues(mat_rvec_, R);
          // OpenCVからEigenへ変換する。
          convertCVtoEigen(mat_tvec_, R, translation, orientation);
          // 変換ベクトルを登録する。
          target_transform.setOrigin(
              tf::Vector3(translation.x(), translation.y(), translation.z()));
          target_transform.setRotation(
              tf::Quaternion(orientation.x(), orientation.y(), orientation.z(),
                             orientation.w()));
          // フラグを変更し、変換ベクトルを確認する。
          is_param_ = true;
          ROS_INFO("Chessboard detected!");
          ROS_INFO("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f", translation.x(),
                   translation.y(), translation.z(), orientation.x(),
                   orientation.y(), orientation.z(), orientation.w());
        } catch (tf::TransformException& ex) {
          ROS_ERROR("TF Exception:\n%s", ex.what());
          return;
        }
      } else {
        ROS_INFO("No Chessboard found!");
      }
    }

    // target_transformが設定済みなら
    if (target_transform.getOrigin().length() > 0) {
      // フレームをSendする。
      tf_broadcaster_.sendTransform(tf::StampedTransform(
          target_transform, ros::Time::now() - ros::Duration(0.1), camera_frame,
          target_frame));
    }

    // 画像を処理する。
    pose2d_block_image =
        rsjImageProcessing(mat_img_color, mat_img_gray, int_method);

    // 検出範囲が求まっているなら
    if (!mat_tvec_.empty()) {
      // ブロックの位置を三次元空間へ変換する。
      point_block_pattern.header.frame_id = target_frame;
      point_block_world.header.frame_id = fixed_frame;

      std::vector<cv::Point3f> vec_point3f_block;
      std::vector<cv::Point2f> vec_point2f_block;
      for (float j = 0; j < point3f_area_max_.y; j += EST_RESO) {
        for (float i = 0; i < point3f_area_max_.x; i += EST_RESO) {
          vec_point3f_block.push_back(cv::Point3f(i, j, 0.0f));
        }
      }
      int int_count_horizontal = floor(point3f_area_max_.x / EST_RESO);
      std::vector<cv::Point> vec_point_area_board;  // 検出可能範囲
      if (vec_point3f_block.size() > 0) {
        try {
          // 三次元空間の位置を二次元平面の位置へ変換する。
          cv::projectPoints(vec_point3f_block, mat_rvec_, mat_tvec_, mat_k_,
                            mat_d_, vec_point2f_block);
        } catch (cv::Exception& ex) {
          ROS_ERROR("CV Exception: %s", ex.what());
          return;
        }
      }
      float f_error_best = FLT_MAX;
      int i_best = -1;
      for (int i = 0; i < vec_point2f_block.size(); i++) {
        float f_error =
            sqrtf(pow(pose2d_block_image.x - vec_point2f_block.at(i).x, 2) +
                  pow(pose2d_block_image.y - vec_point2f_block.at(i).y, 2));
        if (f_error < f_error_best) {
          f_error_best = f_error;
          i_best = i;
        }
        if ((i == 0) || (i == (int_count_horizontal - 1)) ||
            (i == (vec_point2f_block.size() - int_count_horizontal)) ||
            (i == (vec_point2f_block.size() - 1))) {
          vec_point_area_board.push_back(
              cv::Point(vec_point2f_block.at(i).x, vec_point2f_block.at(i).y));
        }
      }
      if (i_best != -1) {
        point_block_pattern.point.x = vec_point3f_block.at(i_best).x;
        point_block_pattern.point.y = vec_point3f_block.at(i_best).y;
        point_block_pattern.point.z = 0.0f;
      } else {
        ROS_INFO("Out of Detectable Area");
      }

      // ブロックの位置をカメラ座標系へ変換する。
      try {
        ros::Time now = ros::Time::now();
        tf_listener_.waitForTransform(target_frame, fixed_frame, now,
                                      ros::Duration(0.1));

        tf_listener_.transformPoint(
            fixed_frame, point_block_pattern,
            point_block_world); // 移動先のframe名、移動元、移動先
      } catch (tf::TransformException& ex) {
        ROS_ERROR("TF Exception: %s", ex.what());
        return;
      }

      // 検出可能範囲を表示する。（一部の要素の順序を入れ替える。）
      cv::Point vec_point_area_board_temp = vec_point_area_board[2];
      vec_point_area_board.erase(vec_point_area_board.begin() + 2);
      vec_point_area_board.push_back(vec_point_area_board_temp);
      polylines(mat_img_color, vec_point_area_board, true,
                cv::Scalar(0, 255, 255), 2);

      // 検出可能範囲の内側かを判定する。
      cv::Point2f pose2d_block_test;
      pose2d_block_test.x = pose2d_block_image.x;
      pose2d_block_test.y = pose2d_block_image.y;
      if (cv::pointPolygonTest(vec_point_area_board, pose2d_block_test,
                               false) != -1) {
        is_block = true;
      }
    }

    // 画像の表示
    cv::imshow("Original", mat_img_color);
    cv::imshow("Result", mat_img_result);
    cv::waitKey(50);  // 単位は[ms]

    // 結果の出力
    image_pub_.publish(cv_img_ptr->toImageMsg());
    if (is_block) {
      std_msgs::Int32 max_area_msg;
      max_area_msg.data = int_max_area_;
      pub_block_size_.publish(max_area_msg);

      // 大きすぎるものと小さすぎるものを排除する。
      if ((3000 <= int_max_area_) && (int_max_area_ <= 9000)) {
        // World座標系のPoint(xyz)をPose2D(xy)へ変換する。
        geometry_msgs::Pose2D pose2d_temp;
        pose2d_temp.x = point_block_world.point.x;
        pose2d_temp.y = point_block_world.point.y;
        pose2d_temp.theta = 0;

        // Publishする。
        pub_position_image_.publish(pose2d_block_image);
        pub_position_space_.publish(pose2d_temp);
        pub_point_pattern_.publish(point_block_pattern);
        pub_point_world_.publish(point_block_world);
      }
    }
  }
};

int main(int argc, char** argv) {
  for (int i = 0; i < argc; i++) {
    std::cout << "arg[" << i << "]: " << argv[i] << std::endl;
  }

  // 画像処理手法を選択する。
  if (argc == 4) {
    int_method = std::stoi(argv[1]);
  } else {
    int_method = 2;
  }

  ros::init(argc, argv, "block_finder");

  BlockFinder bf;

  ros::spin();
  return 0;
}
