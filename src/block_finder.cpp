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

//using namespace std;
//using namespace cv;

static const std::string WINDOW_O = "Original";
static const std::string WINDOW_T = "Temp";
static const std::string WINDOW_R = "Result";
static int threshold_bin = 160;

class BlockFinder
{
	ros::NodeHandle nh_;
	image_transport::ImageTransport it;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	ros::Subscriber info_sub_;
	ros::Publisher pose_pub_;
	ros::Time begin_;
	tf::TransformListener tf_listener_;
	tf::TransformBroadcaster tf_broadcaster_;

	image_geometry::PinholeCameraModel cam_model_;

	std::string fixed_frame;
	std::string camera_frame;
	std::string target_frame;

	cv::Matx33d K;//3x3の行列で、要素はdouble
	cv::Mat D;
	cv::Mat tvec;//並進ベクトル
	cv::Mat rvec;//回転行列
	
	cv::Mat ideal_points_;
	cv::Point2f area_max;

public:

	void convertCVtoEigen(cv::Mat& tvec, cv::Mat& R, Eigen::Vector3f& translation, Eigen::Quaternionf& orientation)
	{
		// This assumes that cv::Mats are stored as doubles. Is there a way to check this?
		// Since it's templated...
		translation = Eigen::Vector3f(tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));

		Eigen::Matrix3f Rmat;
		Rmat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
				R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
				R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

		orientation = Eigen::Quaternionf(Rmat);
	}


	//正解となるチェッカーボードの３次元位置を計算する。
	cv::Mat calcChessboardCorners(cv::Size boardSize, float squareSize, cv::Point3f offset)
	{
		cv::Mat corners;

		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				corners.push_back(cv::Point3f(float(j * squareSize), float(i * squareSize), 0) + offset);
				area_max = cv::Point2f(float(j * squareSize), float(i * squareSize));
			}
		}
		return corners;
	}


	void infoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg)
	{
		cam_model_.fromCameraInfo(info_msg);

		K = cam_model_.intrinsicMatrix();
		D = cam_model_.distortionCoeffs();
	}


	BlockFinder(): it(nh_)
	{
		info_sub_ = nh_.subscribe("/usb_cam_node/camera_info", 1, &BlockFinder::infoCallback, this);
		image_sub_ = it.subscribe("/usb_cam_node/image_raw", 1, &BlockFinder::imageCb, this);
		image_pub_ = it.advertise("/block_finder/image_block", 1);
		pose_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/block_finder/pose", 1);

		nh_.param<std::string>("fixed_frame", fixed_frame, "/world");
		nh_.param<std::string>("camera_frame", camera_frame, "/camera_link");
		nh_.param<std::string>("target_frame", target_frame, "/pattern_link");

		cv::namedWindow(WINDOW_O);
		cv::namedWindow(WINDOW_T);
		cv::namedWindow(WINDOW_R);
		cv::moveWindow(WINDOW_O, 0, 0);
		cv::moveWindow(WINDOW_T, 640, 0);
		cv::moveWindow(WINDOW_R, 1280, 0);
		cv::createTrackbar("Subtracter", WINDOW_T, &threshold_bin, 255);
		
		//３次元位置を求める。
		ideal_points_ = calcChessboardCorners(cv::Size(8, 6), 0.025, cv::Point3f(0.0,0.0,0.0));
	}

	~BlockFinder()
	{
		cv::destroyWindow(WINDOW_O);
		cv::destroyWindow(WINDOW_T);
		cv::destroyWindow(WINDOW_R);
	}


	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{
		begin_ = ros::Time::now();
		
		geometry_msgs::Pose2D pose_block;
		pose_block.x = 0;
		pose_block.y = 0;
		pose_block.theta = 0;

		cv::Size patternsize(8,6);
		cv::Mat board_corners;

		cv_bridge::CvImagePtr cv_ptr_, cv_ptr_2;
		try
		{
			cv_ptr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		}
		catch (cv_bridge::Exception& ex)
		{
			ROS_ERROR("cv_bridge exception:\n%s", ex.what());
			return;
		}

		cv::Mat img_gray_;
		cvtColor(cv_ptr_->image, img_gray_, CV_BGR2GRAY);
		
		bool patternfound = findChessboardCorners(img_gray_, patternsize, board_corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

		if(patternfound)
		{
			cv::drawChessboardCorners(img_gray_, patternsize, board_corners, patternfound);
			
			try
			{
					tf::Transform target_transform;
					tf::StampedTransform base_transform;
					
					Eigen::Vector3f translation(0.0, 0.0, 0.0);
					Eigen::Quaternionf orientation(0.0, 0.0, 0.0, 0.0);
					cv::Mat observation_points_ = board_corners;

					cv::Mat R;

					//画像上の２次元位置と空間内の３次元位置を対応付ける。
					cv::solvePnP(ideal_points_, observation_points_, K, D, rvec, tvec, false);
					//回転ベクトルを回転行列へ変換する。
					cv::Rodrigues(rvec, R);
					convertCVtoEigen(tvec, R, translation, orientation);
					
					//変換ベクトルを確認する。
					//ROS_INFO("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f", translation.x(), translation.y(), translation.z(), orientation.x(), orientation.y(), orientation.z(), orientation.w());

					//変換ベクトルを登録する。
					target_transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
					target_transform.setRotation(tf::Quaternion(orientation.x(), orientation.y(), orientation.z(), orientation.w()));//順番はx, y, z, wです。

					//usb_camの原点はcamera_link
					tf_broadcaster_.sendTransform(tf::StampedTransform(target_transform, begin_, camera_frame, target_frame));
					
					//カメラは画像面がxy、奥行きがz軸
			}
			catch (tf::TransformException& ex)
			{
				ROS_WARN("TF exception:\n%s", ex.what());
				return;
			}
		}
		else
		{
			//ROS_INFO("Not Found");
		}
		
		//二値化する。
		cv::Mat img_bin_, img_bin_2;
		cv::medianBlur(img_gray_, img_gray_, 3);
		cv::threshold(img_gray_, img_bin_, threshold_bin, 255, cv::THRESH_BINARY);//threshold(img_gray, img_bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

		//輪郭を求める。
		std::vector<std::vector<cv::Point> > contours;//輪郭を表現するベクトル
		img_bin_2 = img_bin_.clone();
		cv::findContours(img_bin_2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		
		//各輪郭の面積を求める。
		if(contours.empty() != true)
		{
			double max_area=0.0;
			int max_area_contour=-1;
			for(int i=0;i<contours.size();i++)
			{
				double area=cv::contourArea(contours.at(i));
				if(max_area<area)
				{
					max_area=area;
					max_area_contour=i;
				}
				polylines(cv_ptr_->image, contours.at(i), true, cv::Scalar(0, 255, 0), 3);
			}
			
			//ROS_INFO("%+d (%.0f)", max_area_contour, max_area);

			//重心を求める。
			if(max_area_contour != -1)//輪郭が点で表現されることがある。
			{
				int count=contours.at(max_area_contour).size();
				double x=0.0;
				double y=0.0;
				for(int i=0;i<count;i++)
				{
					x+=contours.at(max_area_contour).at(i).x;
					y+=contours.at(max_area_contour).at(i).y;
				}
				x/=count;
				y/=count;
				circle(cv_ptr_->image, cv::Point(x,y), 10, cv::Scalar(0,0,255), -1, CV_AA);
				
				pose_block.x = x;
				pose_block.y = y;
			}
		}

		
		
		//Segmentation
		/*
		Mat dist;
		Mat kernel1;
		Mat dist_8u;
		Mat markers;
		vector<vector<Point> > contours2;
		distanceTransform(img_bin, dist, CV_DIST_L2, 3);
		normalize(dist, dist, 0, 1., NORM_MINMAX);
		threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
		kernel1 = cv::Mat::ones(3, 3, CV_8UC1);
		dilate(dist, dist, kernel1);
		dist.convertTo(dist_8u, CV_8U);
		findContours(dist_8u, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		if(!contours2.empty()){
			markers = Mat::zeros(dist.size(), CV_32SC1);
			for (size_t i = 0; i<contours2.size(); i++)
			{
				drawContours(markers, contours2, static_cast<int>(i), Scalar::all(static_cast<int>(i) + 1), -1);
			}
			circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);//Background Marker
		}
		 */

		//重心を三次元空間へ変換する。
		tf::StampedTransform transform;
		try{
			tf_listener_.lookupTransform(fixed_frame, camera_frame, begin_, transform);
		}
		catch (tf::TransformException ex)
		{
			ROS_ERROR("%s",ex.what());
		}

		geometry_msgs::PointStamped pose_block_3d, pose_block_3d2;
		pose_block_3d.header.frame_id = camera_frame;//点の座標系
		pose_block_3d2.header.frame_id = fixed_frame;//点の座標系
		pose_block_3d.header.stamp = ros::Time();
		pose_block_3d2.header.stamp = ros::Time();
		pose_block_3d.point.x = 0.0f;
		pose_block_3d.point.y = 0.0f;
		pose_block_3d.point.z = 0.0f;
		
		std::vector<cv::Point3f> test_point3f;
		std::vector<cv::Point2f> test_point2f;
		
		for (float i = 0; i < area_max.x; i += 0.001)
		{
			for (float j = 0; j < area_max.y; j += 0.001)
			{
				test_point3f.push_back(cv::Point3f(i, j, 0.0f));
			}
		}
		//ROS_INFO("%f %f", area_max.x, area_max.y);
		
		try
		{
		    if(int(test_point3f.size()) > 0)
		    {
			    cv::projectPoints(test_point3f, rvec, tvec, K, D, test_point2f);
		        //ROS_INFO("%f", float(test_point2f.size()));
		    }
		}
		catch(cv::Exception& ex)
		{
			ROS_ERROR("cv exception:\n%s", ex.what());
			return;
		}
		
		float error, error_best = FLT_MAX;
		float i_best = -1;
		for (float i = 0; i < float(test_point2f.size()); i++)
		{
			error = sqrtf(pow(pose_block.x - test_point2f.at(i).x, 2)+pow(pose_block.y - test_point2f.at(i).y, 2));
			if (error < error_best)
			{
				error_best = error;
				i_best = i;
				//ROS_INFO("%f, %f", error, i);
			}
		}
		if(i_best != -1)
		{
			ROS_INFO("%f / %f", i_best, float(test_point2f.size()));
			
			pose_block_3d2.point.x = test_point3f.at(i_best).x;
			pose_block_3d2.point.y = test_point3f.at(i_best).y;
			pose_block_3d2.point.z = 0.0f;
		}
		else
		{
			ROS_WARN("Out of Area");
		}
				
		try
		{
			//tf_listener_.transformPoint(camera_frame, begin_ - ros::Duration(1.0), pose_block_3d, fixed_frame, pose_block_3d2);
			//listener.transformPoint("base_link", laser_point, base_point);
			//tf_listener_.transformPoint(fixed_frame, pose_block_3d, pose_block_3d2);
			
			//ROS_INFO("1: %5.3f %5.3f %5.3f", pose_block_3d.point.x, pose_block_3d.point.y, pose_block_3d.point.z);
			//ROS_INFO("2: %5.3f %5.3f %5.3f", pose_block_3d2.point.x, pose_block_3d2.point.y, pose_block_3d2.point.z);
			//ROS_INFO("1: %s", pose_block_3d.header.frame_id.c_str());
			//ROS_INFO("2: %s", pose_block_3d2.header.frame_id.c_str());
		}
		catch(tf::TransformException& ex)
		{
			ROS_ERROR("tf exception:\n%s", ex.what());
			return;
		}

		
		//geometry_msgs::Pose pose_result;
		//pose_result.position.y = pose_block_3d2.point.y;
		//pose_result.position.z = pose_block_3d2.point.z;
		
		//結果の表示
		cv::imshow(WINDOW_O, cv_ptr_->image);//処理前
		cv::waitKey(10);
		cv::imshow(WINDOW_T, img_gray_);//処理中
		cv::waitKey(10);
		//cv::imshow(WINDOW_R, cv_ptr_->image);//処理後
		//cv::waitKey(10);

		
		//結果の出力
		image_pub_.publish(cv_ptr_->toImageMsg());
		pose_pub_.publish(pose_block_3d2);
	}

};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "block_finder");
	BlockFinder bf;
	ros::spin();
	return 0;
}
