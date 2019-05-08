#include <string>

#include <gazebo/gazebo.hh>
#include <gazebo/sensors/Sensor.hh>
#include <gazebo/sensors/GpsSensor.hh>
#include <ignition/math/Angle.hh>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/NavSatFix.h>


using namespace gazebo;


namespace gazebo
{
  /// \brief An example plugin for a contact sensor.
  class RockGpsPlugin : public SensorPlugin
  {
    /// \brief Constructor.
    public: RockGpsPlugin();

    /// \brief Destructor.
    public: virtual ~RockGpsPlugin();

    /// \brief Load the sensor plugin.
    /// \param[in] _sensor Pointer to the sensor that loaded this plugin.
    /// \param[in] _sdf SDF element that describes the plugin.
    public: virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf);

    /// \brief Callback that receives the contact sensor's update signal.
    private: virtual void OnUpdate();

    /// \brief Pointer to the contact sensor
    private: sensors::GpsSensorPtr parentSensor;

    /// \brief Connection that maintains a link between the contact sensor's
    /// updated signal and the OnUpdate callback.
    private: event::ConnectionPtr updateConnection;

    std::unique_ptr<ros::NodeHandle> rosNodeHandle;
    ros::Publisher gpsPub;
    std::string frameId = "GPS_FRAME_ID_NOT_SET";
    std::string topicName;
  };
}

GZ_REGISTER_SENSOR_PLUGIN(RockGpsPlugin)

/////////////////////////////////////////////////
RockGpsPlugin::RockGpsPlugin() : SensorPlugin()
{
}

/////////////////////////////////////////////////
RockGpsPlugin::~RockGpsPlugin()
{
}

/////////////////////////////////////////////////
void RockGpsPlugin::Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
{
  // Get the parent sensor.
  this->parentSensor =
    std::dynamic_pointer_cast<sensors::GpsSensor>(_sensor);

  // Make sure the parent sensor is valid.
  if (!this->parentSensor)
  {
    gzerr << "RockGpsPlugin requires a GpsSensor.\n";
    ROS_FATAL("RockGpsPlugin requires a GpsSensor.\n");
    return;
  }

  if (_sdf->HasElement("frameId"))
    frameId = _sdf->GetElement("frameId")->GetValue()->GetAsString();
  else {
      gzerr << "RockGpsPlugin needs SDF element <frameId>." << std::endl;
      ROS_ERROR("RockGpsPlugin needs SDF element <frameId>");
  }

    if (_sdf->HasElement("topicName"))
    topicName = _sdf->GetElement("topicName")->GetValue()->GetAsString();
  else {
      gzerr << "RockGpsPlugin needs SDF element <topicName>" << std::endl;
      ROS_ERROR("RockGpsPlugin needs SDF element <topicName>");
  }


  // Connect to the sensor update event.
  this->updateConnection = this->parentSensor->ConnectUpdated(
      std::bind(&RockGpsPlugin::OnUpdate, this));

  // Make sure the parent sensor is active.
  this->parentSensor->SetActive(true);



    if (!ros::isInitialized()) {
        ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load RockGpsPlugin. "
                                    << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
        return;
    }
    rosNodeHandle.reset(new ros::NodeHandle("rock_gps_plugin"));
    gpsPub = rosNodeHandle->advertise<sensor_msgs::NavSatFix>(topicName, 1);

}

/////////////////////////////////////////////////
void RockGpsPlugin::OnUpdate()
{
  sensor_msgs::NavSatFix msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frameId;

  double longitude = parentSensor->Longitude().Degree();
  double latitude = parentSensor->Latitude().Degree();
  double altitude = parentSensor->Altitude();
  double velEast = parentSensor->VelocityEast();
  double velNorth = parentSensor->VelocityNorth();
  double velUp = parentSensor->VelocityUp();

  msg.latitude = latitude;
  msg.longitude = longitude;
  msg.altitude = altitude;

  for (int i = 0; i < 9; i++) {
      msg.position_covariance[i] = 0;
  }

  msg.position_covariance_type = msg.COVARIANCE_TYPE_DIAGONAL_KNOWN;

  gpsPub.publish(msg);

}
