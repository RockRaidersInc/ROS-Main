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
  ros::init(argc, argv, "camera",ros::init_options::AnonymousName);
  ros::NodeHandle nh;

  //Specify which camera is being used
  int cameraIndex = 0;
  //Create private nodehandle to get private params
  ros::NodeHandle nhPriv("~");
  if(nhPriv.hasParam("camera")){
    nhPriv.getParam("camera", cameraIndex);
  }

  cv::VideoCapture camera(cameraIndex);

  std::stringstream ss;
  if(!camera.isOpened()){
    ss << "No camera was found at index " << cameraIndex << "!";
    //ros::ROS_ERROR("%s\n", ss.str().c_str());
    ros::shutdown();
  }
  ss.str(std::string());
  ss << "camera_" << cameraIndex;


  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise(ss.str(), 1);

  ros::Rate loopRate(30);
  //ros::ROS_INFO("%s\n", "Stream Started!");

  while(nh.ok()){
    cv::Mat frame;
    camera >> frame;

    if(!frame.empty()){
      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();

      pub.publish(msg);
    }

    ros::spinOnce();
    loopRate.sleep();
  }

  //ros::ROS_INFO("%s\n", "Stream Stopped!");
  camera.release();
}
