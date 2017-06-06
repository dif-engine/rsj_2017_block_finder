#include <ros/ros.h>//ROSを使用するときは必ずincludeする。
#include <geometry_msgs/Pose2D.h>//PoseやTwistを使用する場合は適宜includeする。
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <pcl_ros/transforms.h>//#include <Eigen/Geometry>
#include <image_geometry/pinhole_camera_model.h>//カメラモデルを利用するため

static const std::string WINDOW_O = "Original";
static const std::string WINDOW_R = "Result";

static int method_num = 0;
static int threshold_bin = 160;
static float EST_RESOLUTION = 0.001;

class BlockFinder
{
	
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	ros::Subscriber info_sub_;
	ros::Publisher pose_pub_;
	ros::Time begin_;//現在時刻
	tf::TransformListener tf_listener_;
	tf::TransformBroadcaster tf_broadcaster_;

	image_geometry::PinholeCameraModel cam_model_;

	std::string fixed_frame;
	std::string camera_frame;
	std::string target_frame;

	cv::Matx33d K;//カメラの内部パラメーター
	cv::Mat D;//歪み係数
	cv::Mat tvec;//並進ベクトル
	cv::Mat rvec;//回転行列

	cv::Mat ideal_points_;
	cv::Point3f point3f_area_max_;//ボードの端の座標

	bool is_param;
	bool is_block;

public:

	void convertCVtoEigen(const cv::Mat& tvec, const cv::Mat& R, Eigen::Vector3f& translation, Eigen::Quaternionf& orientation)
	{
		// This assumes that cv::Mats are stored as doubles. Is there a way to check this?
		// Since it_'s templated...
		translation = Eigen::Vector3f(float(tvec.at<double>(0, 0)), float(tvec.at<double>(0, 1)), float(tvec.at<double>(0, 2)));

		Eigen::Matrix3f Rmat;
		Rmat << float(R.at<double>(0, 0)), float(R.at<double>(0, 1)), float(R.at<double>(0, 2)),
				float(R.at<double>(1, 0)), float(R.at<double>(1, 1)), float(R.at<double>(1, 2)),
			    float(R.at<double>(2, 0)), float(R.at<double>(2, 1)), float(R.at<double>(2, 2));

		orientation = Eigen::Quaternionf(Rmat);
	}


	//基準となるチェッカーボードの３次元位置の計算
	cv::Mat calcChessboardCorners(cv::Size boardSize, float squareSize, cv::Point3f offset)
	{
		cv::Mat corners;

		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				corners.push_back(cv::Point3f(float(j * squareSize), float(i * squareSize), 0.0f) + offset);
			}
		}

		point3f_area_max_ = cv::Point3f(float((boardSize.width - 1) * squareSize), float((boardSize.height - 1) * squareSize), 0.0f);

		return corners;
	}

	//カメラ情報の取得
	void infoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg)
	{
		cam_model_.fromCameraInfo(info_msg);

		K = cam_model_.intrinsicMatrix();
		D = cam_model_.distortionCoeffs();
	}


	BlockFinder(): it_(nh_)
	{
		info_sub_ = nh_.subscribe("/usb_cam_node/camera_info", 1, &BlockFinder::infoCallback, this);
		image_sub_ = it_.subscribe("/usb_cam_node/image_raw", 1, &BlockFinder::imageCb, this);
		image_pub_ = it_.advertise("/block_finder/image_block", 1);
		pose_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/block_finder/pose", 1);

		nh_.param<std::string>("fixed_frame", fixed_frame, "/world");
		nh_.param<std::string>("camera_frame", camera_frame, "/camera_link");
		nh_.param<std::string>("target_frame", target_frame, "/pattern_link");

		cv::namedWindow(WINDOW_O);
		cv::namedWindow(WINDOW_R);
		cv::moveWindow(WINDOW_O, 0, 0);
		cv::moveWindow(WINDOW_R, 0, 500);//640, 0
		cv::createTrackbar("Subtracter", WINDOW_R, &threshold_bin, 255);

		//３次元位置を求める。
		ideal_points_ = calcChessboardCorners(cv::Size(8, 6), 0.025, cv::Point3f(0.0,0.0,0.0));
		
		is_param = false;
		is_block = false;
	}

	~BlockFinder()
	{
		cv::destroyWindow(WINDOW_O);
		cv::destroyWindow(WINDOW_R);
	}

	geometry_msgs::Pose2D rsjImageProcessing(cv::Mat& mat_img_input_c, cv::Mat& mat_img_input_g)
	{
		geometry_msgs::Pose2D pose2d_block;
		pose2d_block.x = 0.0f;
		pose2d_block.y = 0.0f;
		pose2d_block.theta = 0.0f;

		//二値化する。
		cv::Mat mat_img_bin, mat_img_bin_temp;
		cv::medianBlur(mat_img_input_g, mat_img_input_g, 3);
		if(method_num == 0)
		{
			cv::threshold(mat_img_input_g, mat_img_bin, threshold_bin, 255, cv::THRESH_BINARY);
		}
		else
		{
			cv::threshold(mat_img_input_g, mat_img_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		}

		//輪郭を求める。
		std::vector<std::vector<cv::Point> > contours;//輪郭を表現するベクトル
		mat_img_bin_temp = mat_img_bin.clone();
		cv::findContours(mat_img_bin_temp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		//各輪郭の面積を求める。
		if(contours.empty() != true)
		{
			double max_area = 0.0;
			int max_area_contour = -1;
			for(int i = 0; i < contours.size(); i++)
			{
				polylines(mat_img_input_c, contours.at(i), true, cv::Scalar(0, 255, 0), 2);

				double area = cv::contourArea(contours.at(i));
				if(area > max_area)
				{
					max_area = area;
					max_area_contour = i;
				}
			}

			//重心を求める。
			if(max_area_contour != -1)//輪郭が点で表現されることがある。
			{
				int count=contours.at(max_area_contour).size();
				double x = 0.0;
				double y = 0.0;
				for(int i = 0; i < count; i++)
				{
					x+=contours.at(max_area_contour).at(i).x;
					y+=contours.at(max_area_contour).at(i).y;
				}
				x/=count;
				y/=count;
				circle(mat_img_input_c, cv::Point(x,y), 8, cv::Scalar(0,0,255), -1, CV_AA);

				pose2d_block.x = x;
				pose2d_block.y = y;
			}
		}

		return pose2d_block;
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{
		begin_ = ros::Time::now();

		geometry_msgs::Pose2D pose2d_block;//画像中のブロックの位置

		cv::Size patternsize(8,6);
		cv::Mat board_corners;

		cv_bridge::CvImagePtr cv_img_ptr;
		cv::Mat mat_img_color;
		try
		{
			cv_img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
			mat_img_color = cv_img_ptr->image.clone();
		}
		catch (cv_bridge::Exception& ex)
		{
			ROS_ERROR("cv_bridge exception:\n%s", ex.what());
			return;
		}

		//グレースケール変換
		cv::Mat mat_img_gray;
		cvtColor(mat_img_color, mat_img_gray, CV_BGR2GRAY);

		if(!is_param)
		{
			//チェスボード検出
			bool patternfound = findChessboardCorners(mat_img_gray, patternsize, board_corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
			if(patternfound)//検出できた場合
			{
				cv::drawChessboardCorners(mat_img_gray, patternsize, board_corners, patternfound);
				try
				{
					tf::Transform target_transform;
					tf::StampedTransform base_transform;

					Eigen::Vector3f translation(0.0f, 0.0f, 0.0f);
					Eigen::Quaternionf orientation(0.0f, 0.0f, 0.0f, 0.0f);

					cv::Mat observation_points_ = board_corners;
					cv::Mat R;

					//画像上の２次元位置と空間内の３次元位置を対応付ける。
					cv::solvePnP(ideal_points_, observation_points_, K, D, rvec, tvec, false);
					//回転ベクトルを回転行列へ変換する。
					cv::Rodrigues(rvec, R);
					//OpenCVからEigenへ変換する。
					convertCVtoEigen(tvec, R, translation, orientation);
					//変換ベクトルを確認する。
					//ROS_INFO("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f", translation.x(), translation.y(), translation.z(), orientation.x(), orientation.y(), orientation.z(), orientation.w());
					//変換ベクトルを登録する。
					target_transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
					target_transform.setRotation(tf::Quaternion(orientation.x(), orientation.y(), orientation.z(), orientation.w()));
					//フレームを発信する。
					tf_broadcaster_.sendTransform(tf::StampedTransform(target_transform, begin_, camera_frame, target_frame));
					//フラグを変更する。
					is_param = true;
				}
				catch (tf::TransformException& ex)
				{
					ROS_ERROR("TF exception:\n%s", ex.what());
					return;
				}
			}
			else
			{
				//ROS_INFO("No Chessboard");
			}
		}
		
		//画像を処理する。
		pose2d_block = rsjImageProcessing(mat_img_color, mat_img_gray);

		//ブロックの位置を三次元空間へ変換する。
		geometry_msgs::PointStamped pose3d_block;
		geometry_msgs::PointStamped pose3d_block_tf;
		pose3d_block.header.frame_id = fixed_frame;
		pose3d_block_tf.header.frame_id = camera_frame;
		pose3d_block.header.stamp = ros::Time();
		pose3d_block_tf.header.stamp = ros::Time();
		//
		std::vector<cv::Point3f> vec_point3f_block;
		std::vector<cv::Point2f> vec_point2f_block;
		//ROS_INFO("x: %f, y: %f",point3f_area_max_.x, point3f_area_max_.y);
		for (float j = 0; j < point3f_area_max_.y; j += EST_RESOLUTION)
		{
			for (float i = 0; i < point3f_area_max_.x; i += EST_RESOLUTION)
			{
				vec_point3f_block.push_back(cv::Point3f(i, j, 0.0f));
			}
		}
		int int_count_horizontal = int(floor(point3f_area_max_.x/EST_RESOLUTION));
		int int_count_vertical =  int(floor(point3f_area_max_.y/EST_RESOLUTION));
		std::vector<cv::Point> vec_point_area_board;//検出可能範囲
		//ROS_INFO("v: %d, h: %d",int_count_vertical, int_count_horizontal);
		if(int(vec_point3f_block.size()) > 0)
		{
			try
			{
				//三次元空間の位置を二次元平面の位置へ変換する。
				cv::projectPoints(vec_point3f_block, rvec, tvec, K, D, vec_point2f_block);
			}
			catch(cv::Exception& ex)
			{
				ROS_ERROR("cv exception:\n%s", ex.what());
				return;
			}
		}
		float error, error_best = FLT_MAX;
		int i_best = -1;
		for (int i = 0; i < vec_point2f_block.size(); i++)
		{
			error = sqrtf(pow(pose2d_block.x - vec_point2f_block.at(i).x, 2)+pow(pose2d_block.y - vec_point2f_block.at(i).y, 2));
			if (error < error_best)
			{
				error_best = error;
				i_best = i;
			}
			if((i == 0)||(i == (int_count_horizontal - 1))||(i == (vec_point2f_block.size() - int_count_horizontal))||(i == (vec_point2f_block.size() - 1)))
			{
				vec_point_area_board.push_back(cv::Point(vec_point2f_block.at(i).x, vec_point2f_block.at(i).y));
			}		
		}
		if(i_best != -1)
		{
			//ROS_INFO("%.0f / %.0f", i_best, float(vec_point2f_block.size()));

			pose3d_block.point.x = vec_point3f_block.at(i_best).x;
			pose3d_block.point.y = vec_point3f_block.at(i_best).y;
			pose3d_block.point.z = 0.0f;
			is_block = true;
		}
		else
		{
			ROS_INFO("Out of Area");
		}

		//カメラ座標系へ変換する。
		try
		{
			tf_listener_.transformPoint(fixed_frame, begin_, pose3d_block, camera_frame, pose3d_block_tf);
		}
		catch(tf::TransformException& ex)
		{
			ROS_ERROR("tf exception:\n%s", ex.what());
			return;
		}

		cv::Point vec_point_area_board_temp = vec_point_area_board[2];
		vec_point_area_board.erase(vec_point_area_board.begin() + 2);
		vec_point_area_board.push_back(vec_point_area_board_temp);
		polylines(mat_img_color, vec_point_area_board, true, cv::Scalar(0, 255, 255), 2);

		//画像の表示
		cv::imshow(WINDOW_O, mat_img_color);//処理前
		cv::waitKey(10);
		cv::imshow(WINDOW_R, mat_img_gray);//処理後
		cv::waitKey(10);

		//結果の出力
		image_pub_.publish(cv_img_ptr->toImageMsg());
		if(is_block)
		{
			pose_pub_.publish(pose3d_block);
			is_block = false;
		}
	}
};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "block_finder");
	BlockFinder bf;
	ros::spin();
	return 0;
}
