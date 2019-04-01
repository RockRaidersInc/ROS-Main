
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

#include <sstream>


void chatterCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    static tf::TransformBroadcaster tf_pub;
    
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));
    tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    transform.setRotation(q);
    tf_pub.sendTransform(tf::StampedTransform(transform, msg->header.stamp, "odom", "base_link"));

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);

    ros::spin();

    return 0;
}
