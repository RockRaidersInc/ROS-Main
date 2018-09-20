#include <iostream>
#include <math.h>
#include <sdf/sdf.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Angle.hh>

#include "gazebo/sensors/sensors.hh"
#include "gazebo/sensors/GpsSensor.hh"
#include "gazebo/gazebo.hh"
#include "gazebo/common/common.hh"
#include "gazebo/msgs/msgs.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/transport/transport.hh"

#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Int16.h"

namespace gazebo
{

class SimGpsPlugin : public SensorPlugin
{
    private:
    sensors::GpsSensorPtr gps_sensor_ptr;
    sdf::ElementPtr sdf_elem;

    ros::NodeHandle _nh;
    ros::Publisher _sensorPublisher;

    public:
    SimGpsPlugin() { }

     void OnUpdate() {
        if (gps_sensor_ptr->IsActive()) {
            ignition::math::Angle lat = gps_sensor_ptr->Latitude();
            ignition::math::Angle lon = gps_sensor_ptr->Longitude();
        }
        std::cout << "GPS plugin OnUpdate" << std::endl;      
        std::cerr << "GPS plugin OnUpdate" << std::endl;     
    }

    void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf) {
        std::cout << "GPS plugin Loading" << std::endl;      
        std::cerr << "GPS plugin Loading" << std::endl; 
        // sensors::GpsSensorPtr temp = *std::dynamic_pointer_cast<sensors::GpsSensorPtr>(_sensor);
        // gps_sensor_ptr = temp;
        // sdf_elem = _sdf;
    }
};

// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(SimGpsPlugin)
}