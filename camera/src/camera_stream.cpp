#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>

#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]){
  //Initialization
  ros::init(argc, argv, "node_camera",ros::init_options::AnonymousName);
  ros::NodeHandle nh;

  //Specify which camera is being used
  int camera_index = 0;
  //Create private nodehandle to get private params
  ros::NodeHandle nh_priv("~");
  if(nh_priv.hasParam("camera")){
    nh_priv.getParam("camera", camera_index);
  }

  cv::VideoCapture camera(camera_index);

  std::stringstream ss;
  if(!camera.isOpened()){
    ss << "No camera was found at index " << camera_index << "!";
    //ros::ROS_ERROR("%s\n", ss.str().c_str());
    ros::shutdown();
  }
  ss.str(std::string());
  ss << "camera_" << camera_index;


  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise(ss.str(), 1);

  ros::Rate loop_rate(30);
  //ros::ROS_INFO("%s\n", "Stream Started!");

  while(nh.ok()){
    cv::Mat frame;
    camera >> frame;

    if(!frame.empty()){
      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();

      pub.publish(msg);
    }

    ros::spinOnce();
    loop_rate.sleep();
  }

  //ros::ROS_INFO("%s\n", "Stream Stopped!");
  camera.release();
}
