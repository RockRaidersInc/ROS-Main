//TODO: licence

#include <gazebo/common/Plugin.hh>
#include <ros/ros.h>
#include "nav_msgs/Odometry.h"
#include <gazebo/physics/physics.hh>
#include "std_msgs/Int32.h"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/transform_datatypes.h>

namespace gazebo {

class GazeboExactOdomPlugin : public ModelPlugin
{
public:
  GazeboExactOdomPlugin();
  virtual ~GazeboExactOdomPlugin();

protected:
  virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
  virtual void Update();

private:
  /// \brief The parent World
  physics::WorldPtr world;

  /// \brief The link referred to by this plugin
  physics::LinkPtr link;

  std::unique_ptr<ros::NodeHandle> node_handle_;
  ros::Publisher odom_publisher;
  ros::Publisher map_publisher;

  std::string namespace_;
  std::string link_name_;
  std::string frame_id_;

  event::ConnectionPtr updateConnection;

  ros::Time last_time;

  double update_rate = 0.1;
};

GazeboExactOdomPlugin::GazeboExactOdomPlugin() { }

////////////////////////////////////////////////////////////////////////////////
// Destructor
GazeboExactOdomPlugin::~GazeboExactOdomPlugin()
{
  node_handle_->shutdown();
}

////////////////////////////////////////////////////////////////////////////////
// Load the controller
void GazeboExactOdomPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  std::cout << "exact odom plugin loading" << std::endl;
  world = _model->GetWorld();

  // load parameters
  if (!_sdf->HasElement("robotNamespace"))
    namespace_.clear();
  else
    namespace_ = _sdf->GetElement("robotNamespace")->GetValue()->GetAsString();

  if (!_sdf->HasElement("bodyName"))
  {
    link = _model->GetLink();
    link_name_ = link->GetName();
  }
  else {
    link_name_ = _sdf->GetElement("bodyName")->GetValue()->GetAsString();
    link = _model->GetLink(link_name_);
  }

  if (!link)
  {
    ROS_FATAL("GazeboExactOdomPlugin plugin error: bodyName: %s does not exist\n", link_name_.c_str());
    return;
  }

  // default parameters
  frame_id_ = "FRAME_ID_MUST_BE_SET";
  std::string topic = "fix";


  if (_sdf->HasElement("frameId"))
    frame_id_ = _sdf->GetElement("frameId")->GetValue()->GetAsString();

  if (_sdf->HasElement("topicName"))
    topic = _sdf->GetElement("topicName")->GetValue()->GetAsString();


  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
      << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  node_handle_.reset(new ros::NodeHandle("gazebo_perfect_odom_node"));
  odom_publisher = node_handle_->advertise<nav_msgs::Odometry>("/odometry/perfect", 2);
  map_publisher = node_handle_->advertise<nav_msgs::Odometry>("/odometry/perfect_map", 2);

  last_time = ros::Time::now();

  // setup dynamic_reconfigure servers
  this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&GazeboExactOdomPlugin::Update, this));
}


////////////////////////////////////////////////////////////////////////////////
// Update the controller
void GazeboExactOdomPlugin::Update()
{
  auto current_time = ros::Time::now();

  if ((current_time - last_time).toSec() > update_rate) {
    last_time = current_time;

    common::Time sim_time = world->GetSimTime();
    nav_msgs::Odometry msg;
    msg.header.stamp = ros::Time::now();
    msg.child_frame_id = "base_link";

    #if (GAZEBO_MAJOR_VERSION >= 8)
        ignition::math::Pose3d pose = link->WorldPose();
        ignition::math::Vector3d velocity = link->WorldLinearVel();
    #else
        math::Pose pose = link->GetWorldPose();
        tf2::Vector3 pose_pos(pose.pos.x, pose.pos.y, 0);  // pose.pos.z - 1.06985 the offset makes z zero when the rover is on zero-height terrain
        tf2::Quaternion pose_rot(pose.rot.x, pose.rot.y, pose.rot.z, pose.rot.w);

        gazebo::math::Vector3 velocity = link->GetWorldLinearVel();
        tf2::Vector3 lin_vel(velocity.x, velocity.y, velocity.z);

        gazebo::math::Vector3 velocityAngular = link->GetWorldAngularVel();
        tf2::Vector3 ang_vel(velocityAngular.x, velocityAngular.y, velocityAngular.z);
    #endif

    for(int i = 0; i < 36; i++) {
      msg.pose.covariance[i] = 0;
    }
    for(int i = 0; i < 36; i++) {
      msg.twist.covariance[i] = 0;
    }

    // Offset is added here
    tf2::Quaternion pose_rot_offset = tf2::Quaternion(0, 0, 3.14159 / 2) * pose_rot;

    msg.pose.pose.position.x = pose_pos.getX();
    msg.pose.pose.position.y = pose_pos.getY();
    msg.pose.pose.position.z = pose_pos.getZ();
    msg.pose.pose.orientation = tf2::toMsg(pose_rot_offset);

    tf2::Vector3 lin_vel_rotated = lin_vel * tf2::Matrix3x3(pose_rot_offset);
    msg.twist.twist.linear = tf2::toMsg(lin_vel_rotated);
    msg.twist.twist.angular = tf2::toMsg(ang_vel);

    msg.header.frame_id = "map";
    map_publisher.publish(msg);
    msg.header.frame_id = "odom";
    odom_publisher.publish(msg);
  }
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(GazeboExactOdomPlugin)

} // namespace gazebo
