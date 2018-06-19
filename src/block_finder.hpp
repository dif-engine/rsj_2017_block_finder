#ifndef RSJ_2017_BLOCK_FINDER_BLOCK_FINDER_HPP
#define RSJ_2017_BLOCK_FINDER_BLOCK_FINDER_HPP

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

  int int_method_;

  bool is_headless_;

public:
  BlockFinder(const int method, const bool is_headless) ;
  ~BlockFinder();
  void convertCVtoEigen(const cv::Mat& mat_tvec_, const cv::Mat& R,
                        Eigen::Vector3f& translation,
                        Eigen::Quaternionf& orientation);
  cv::Mat calcChessboardCorners(cv::Size boardSize, float squareSize,
                                cv::Point3f offset);
  void infoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg);
  geometry_msgs::Pose2D rsjImageProcessing(cv::Mat& mat_img_input_c,
                                           cv::Mat& mat_img_input_g);
  void imageCb(const sensor_msgs::ImageConstPtr& msg);

};
#endif
