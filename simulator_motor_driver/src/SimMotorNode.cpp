//
// Created by david on 1/24/18.
//

#include <cmath>
#include <ctime>
#include <chrono>

#include <ros/ros.h>
#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>


#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float64.h"


#include "SimMotorNode.h"


// #define DEBUG_PRINT


namespace gazebo {
    /// \brief A plugin to control a Velodyne sensor.
    class MotorNodePlugin : public ModelPlugin {

        /// \brief Constructor
    public:
        MotorNodePlugin() {}

        /// \brief The load function is called by Gazebo when the plugin is
        /// inserted into simulation
        /// \param[in] _model A pointer to the model that this plugin is
        /// attached to.
        /// \param[in] _sdf A pointer to the plugin's SDF element.
    public:
        virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

        void OnUpdate();

    private:
        // Pointer to the model
        physics::ModelPtr model;

        // Pointer to the update event connection
        event::ConnectionPtr updateConnection;

        // a pointer to this ros node
        std::unique_ptr<ros::NodeHandle> rosNodeHandle;


        // Making a seperate subscriber and callback for each motor was intentional, some of the motors might need
        // to be treated differently from the others (like they might need different torque scaling or something)
        ros::Subscriber backLeftMotorSub;
        ros::Subscriber middleLeftMotorSub;
        ros::Subscriber frontLeftMotorSub;
        ros::Subscriber backRightMotorSub;
        ros::Subscriber middleRightMotorSub;
        ros::Subscriber frontRightMotorSub;

        double backLeftMotorVal = 0;
        double middleLeftMotorVal = 0;
        double frontLeftMotorVal = 0;
        double backRightMotorVal = 0;
        double middleRightMotorVal = 0;
        double frontRightMotorVal = 0;

        void backLeftMotorCallback(const std_msgs::Float64::ConstPtr& msg);

        void middleLeftMotorCallback(const std_msgs::Float64::ConstPtr& msg);
        void frontLeftMotorCallback(const std_msgs::Float64::ConstPtr& msg);
        void backRightMotorCallback(const std_msgs::Float64::ConstPtr& msg);
        void middleRightMotorCallback(const std_msgs::Float64::ConstPtr& msg);
        void frontRightMotorCallback(const std_msgs::Float64::ConstPtr& msg);


        double mapMotorTorque(double inval);


    };

    // Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(MotorNodePlugin)




    void MotorNodePlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
        this->model = _model;


        // Just output a message for now
        std::cerr << "\nThe motor controll plugin is attach to model[" <<
                  _model->GetName() << "]\n";

        if (!ros::isInitialized()) {
            ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                                     << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
            return;
        }

        // Listen to the update event. This event is broadcast every simulation iteration.
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                std::bind(&MotorNodePlugin::OnUpdate, this));


        this->rosNodeHandle.reset(new ros::NodeHandle("gazebo_motor_node"));

        backLeftMotorSub = rosNodeHandle->subscribe("backLeft", 1000, &MotorNodePlugin::backLeftMotorCallback, this);
        middleLeftMotorSub = rosNodeHandle->subscribe("middleLeft", 1000, &MotorNodePlugin::middleLeftMotorCallback, this);
        frontLeftMotorSub = rosNodeHandle->subscribe("frontLeft", 1000, &MotorNodePlugin::frontLeftMotorCallback, this);
        backRightMotorSub = rosNodeHandle->subscribe("backRight", 1000, &MotorNodePlugin::backRightMotorCallback, this);
        middleRightMotorSub = rosNodeHandle->subscribe("middleRight", 1000, &MotorNodePlugin::middleRightMotorCallback, this);
        frontRightMotorSub = rosNodeHandle->subscribe("frontRight", 1000, &MotorNodePlugin::frontRightMotorCallback, this);


//            ROS_INFO("Simulator Motor Node Started");
    }



    void MotorNodePlugin::backLeftMotorCallback(const std_msgs::Float64::ConstPtr& msg) {
        backLeftMotorVal = msg->data;
#ifdef DEBUG_PRINT
        ROS_INFO("I heard: [%s]", std::to_string(msg->data).c_str());
#endif
    }

    void MotorNodePlugin::middleLeftMotorCallback(const std_msgs::Float64::ConstPtr& msg) {
        middleLeftMotorVal = msg->data;
#ifdef DEBUG_PRINT
        ROS_INFO("I heard: [%s]", std::to_string(msg->data).c_str());
#endif
    }

    void MotorNodePlugin::frontLeftMotorCallback(const std_msgs::Float64::ConstPtr& msg) {
        frontLeftMotorVal = msg->data;
#ifdef DEBUG_PRINT
        ROS_INFO("I heard: [%s]", std::to_string(msg->data).c_str());
#endif
    }

    void MotorNodePlugin::backRightMotorCallback(const std_msgs::Float64::ConstPtr& msg) {
        backRightMotorVal = msg->data;
#ifdef DEBUG_PRINT
        ROS_INFO("I heard: [%s]", std::to_string(msg->data).c_str());
#endif
    }

    void MotorNodePlugin::middleRightMotorCallback(const std_msgs::Float64::ConstPtr& msg) {
        middleRightMotorVal = msg->data;
#ifdef DEBUG_PRINT
        ROS_INFO("I heard: [%s]", std::to_string(msg->data).c_str());
#endif
    }


    void MotorNodePlugin::frontRightMotorCallback(const std_msgs::Float64::ConstPtr& msg) {
        frontRightMotorVal = msg->data;
#ifdef DEBUG_PRINT
        ROS_INFO("I heard: [%s]", std::to_string(msg->data).c_str());
#endif
    }



    void MotorNodePlugin::OnUpdate() {

        u_int joint_axis = 0;
        this->model->GetJoint("left_back_wheel_hinge")->SetForce(joint_axis, mapMotorTorque(backLeftMotorVal));
        this->model->GetJoint("left_mid_wheel_hinge")->SetForce(joint_axis,  mapMotorTorque(middleLeftMotorVal));
        this->model->GetJoint("left_front_wheel_hinge")->SetForce(joint_axis,  mapMotorTorque(frontLeftMotorVal));
        this->model->GetJoint("right_back_wheel_hinge")->SetForce(joint_axis,  -1 * mapMotorTorque(backRightMotorVal));
        this->model->GetJoint("right_mid_wheel_hinge")->SetForce(joint_axis, -1 * mapMotorTorque(middleRightMotorVal));
        this->model->GetJoint("right_front_wheel_hinge")->SetForce(joint_axis, -1 *  mapMotorTorque(frontRightMotorVal));

    }


    double MotorNodePlugin::mapMotorTorque(double inval) {
        // This line should be some kind of scalar so that the simulated rover accelerates at about the same rate as the actual motor.
        return inval * 1.0;
    }


    void testCallback(const std_msgs::Float64::ConstPtr& msg) {
        std::cout << "test" << std::endl;
    }
}
