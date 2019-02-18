//
// Created by david on 1/24/18.
//

#include <cmath>
#include <ctime>
#include <chrono>
#include <vector>
#include <functional>

#include <ros/ros.h>
#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>

#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Int16.h"
#include "nav_msgs/Odometry.h"


// #include "SimMotorNode.h"


// #define DEBUG_PRINT


namespace gazebo {
    /// \brief A plugin to control a Velodyne sensor.
    class MotorNodePlugin : public ModelPlugin {

        /// \brief Constructor
    public:
        MotorNodePlugin() {
             std::cerr << std::endl << "The motor controll plugin constructor just ran" << std::endl;
        }

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

        // for controlling the arm joint angles
        physics::JointController* j2_controller;

        // Making a seperate subscriber and callback for each motor was intentional, some of the motors might need
        // to be treated differently from the others (like they might need different torque scaling or something)
        ros::Subscriber leftMotorSub;
        ros::Subscriber rightMotorSub;

        ros::Subscriber ArmSub0;
        ros::Subscriber ArmSub1;
        ros::Subscriber ArmSub2;
        ros::Subscriber ArmSub3;
        ros::Subscriber ArmSub4;
        ros::Subscriber ArmSub5;

        ros::Publisher OdomPub;
        ros::Timer OdomPubTimer;

        double leftMotorVel = 0;
        double rightMotorVel = 0;
        std::vector<double> armJointVals;
        std::atomic<long long> odomMessageSequenceNum{0};
        std::atomic<double> leftMotorSpeed{0};
        std::atomic<double> rightMotorSpeed{0};


        void leftMotorCallback(const std_msgs::Int16::ConstPtr& msg);
        void rightMotorCallback(const std_msgs::Int16::ConstPtr& msg);

        void ArmCallback0(const std_msgs::Int16::ConstPtr& msg);
        void ArmCallback1(const std_msgs::Int16::ConstPtr& msg);
        void ArmCallback2(const std_msgs::Int16::ConstPtr& msg);
        void ArmCallback3(const std_msgs::Int16::ConstPtr& msg);
        void ArmCallback4(const std_msgs::Int16::ConstPtr& msg);
        void ArmCallback5(const std_msgs::Int16::ConstPtr& msg);

        int OdomPubCallback(const ros::TimerEvent&);

        double mapMotorVel(double inval);

        double map(double x, double in_min, double in_max, double out_min, double out_max);

    };

    // Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(MotorNodePlugin)


    void MotorNodePlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
         std::cerr << std::endl << "Motor controll plugin is starting to load [" << _model->GetName() << "]" << std::endl;

        for(int i = 0; i < 6; i++) {
                armJointVals.push_back(0);
        }

        this->model = _model;

        // Just output a message for now
        std::cerr << "\nThe motor controll plugin is attach to model[" << _model->GetName() << "]" << std::endl;

        if (!ros::isInitialized()) {
            ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                                     << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
            return;
        }

        // Listen to the update event. This event is broadcast every simulation iteration.
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                std::bind(&MotorNodePlugin::OnUpdate, this));

        this->j2_controller = new physics::JointController(this->model);

        // All ROS stuff
        rosNodeHandle.reset(new ros::NodeHandle("gazebo_motor_node"));

        leftMotorSub = rosNodeHandle->subscribe("/motors/left_vel", 1, &MotorNodePlugin::leftMotorCallback, this);
        rightMotorSub = rosNodeHandle->subscribe("/motors/right_vel", 1, &MotorNodePlugin::rightMotorCallback, this);
       
        ArmSub0 = rosNodeHandle->subscribe("/motors/Arm0", 1, &MotorNodePlugin::ArmCallback0, this);
        ArmSub1 = rosNodeHandle->subscribe("/motors/Arm1", 1, &MotorNodePlugin::ArmCallback1, this);
        ArmSub2 = rosNodeHandle->subscribe("/motors/Arm2", 1, &MotorNodePlugin::ArmCallback2, this);
        ArmSub3 = rosNodeHandle->subscribe("/motors/Arm3", 1, &MotorNodePlugin::ArmCallback3, this);
        ArmSub4 = rosNodeHandle->subscribe("/motors/Arm4", 1, &MotorNodePlugin::ArmCallback4, this);
        ArmSub5 = rosNodeHandle->subscribe("/motors/Arm5", 1, &MotorNodePlugin::ArmCallback5, this);


        this->model->GetJoint("left_back_wheel_hinge")->SetParam("max_force", 0, 10000);
        this->model->GetJoint("left_front_wheel_hinge")->SetParam("max_force", 0, 10000);
        this->model->GetJoint("right_back_wheel_hinge")->SetParam("max_force", 0, 10000);
        this->model->GetJoint("right_front_wheel_hinge")->SetParam("max_force", 0, 10000);

        
        // publish wheel odometry for the localization code
        OdomPub = rosNodeHandle->advertise<nav_msgs::Odometry>("/odometrey/wheel", 1000);
        OdomPubTimer = rosNodeHandle->createTimer(ros::Duration(0.01), boost::bind(&gazebo::MotorNodePlugin::OdomPubCallback, this, _1));


        // ROS_INFO("Simulator Motor Node Started");
    }


    void MotorNodePlugin::leftMotorCallback(const std_msgs::Int16::ConstPtr& msg) {
        leftMotorVel = msg->data;
    }

    void MotorNodePlugin::rightMotorCallback(const std_msgs::Int16::ConstPtr& msg) {
        rightMotorVel = msg->data;
    }

    void MotorNodePlugin::ArmCallback0(const std_msgs::Int16::ConstPtr& msg) {
        armJointVals.at(0) = (double) msg->data;
        this->j2_controller->SetJointPosition(this->model->GetJoint("armbase_armcentershaft"), map(armJointVals[0], -128, 127, -3.14, 3.14));
    }

    void MotorNodePlugin::ArmCallback1(const std_msgs::Int16::ConstPtr& msg) {
        armJointVals.at(1) = (double) msg->data;
        this->j2_controller->SetJointPosition(this->model->GetJoint("armcentershaftoffset_backarm"), map(armJointVals[1], -128, 127, -3.14, 3.14));
    }

    void MotorNodePlugin::ArmCallback2(const std_msgs::Int16::ConstPtr& msg) {
        armJointVals.at(2) = (double) msg->data;
        this->j2_controller->SetJointPosition(this->model->GetJoint("backarm_forearm"), map(armJointVals[2], -128, 127, -3.14, 3.14));
    }

    void MotorNodePlugin::ArmCallback3(const std_msgs::Int16::ConstPtr& msg) {
        armJointVals.at(3) = (double) msg->data;
    }

    void MotorNodePlugin::ArmCallback4(const std_msgs::Int16::ConstPtr& msg) {
        armJointVals.at(4) = (double) msg->data;
    }

    void MotorNodePlugin::ArmCallback5(const std_msgs::Int16::ConstPtr& msg) {
        armJointVals.at(5) = (double) msg->data;
    }


    // std::vector<double> armJointVals;
    // static std::atomic<long long> odomMessageSequenceNum;
    // static std::atomic<double> leftMotorSpeed;
    // static std::atomic<double> rightMotorSpeed;

    int MotorNodePlugin::OdomPubCallback(const ros::TimerEvent& time_obj) {
        double left_vel = leftMotorSpeed;
        double right_vel = rightMotorSpeed;
        double dt = (time_obj.current_real - time_obj.last_real).toSec();

        double linear_vel = (left_vel + right_vel) / 2;

        //TODO: make these not constants, maybe use rosparam?

        double track = 35.485/12;  // Horizontal distance between wheels (ft)
        double diameter = 1.0;  // Wheel diameter (ft)

        double omega = (right_vel - left_vel)*diameter/(2*track);

        auto out_msg = nav_msgs::Odometry();
        out_msg.header.seq = odomMessageSequenceNum;
        out_msg.header.frame_id = "base_link";
        out_msg.twist.twist.linear.x = linear_vel;
        out_msg.twist.twist.angular.z = omega;

        OdomPub.publish(out_msg);
    }


    void MotorNodePlugin::OnUpdate() {
        u_int joint_axis = 0;

        this->model->GetJoint("left_back_wheel_hinge")->SetVelocity(joint_axis, mapMotorVel(leftMotorVel));
        this->model->GetJoint("left_front_wheel_hinge")->SetVelocity(joint_axis, mapMotorVel(leftMotorVel));
        this->model->GetJoint("right_back_wheel_hinge")->SetVelocity(joint_axis, mapMotorVel(rightMotorVel));
        this->model->GetJoint("right_front_wheel_hinge")->SetVelocity(joint_axis, mapMotorVel(rightMotorVel));

        leftMotorSpeed = mapMotorVel(leftMotorVel);
        rightMotorSpeed = mapMotorVel(rightMotorVel);
    }

    double MotorNodePlugin::mapMotorVel(double inval) {
        // This line makes the simulated wheels turn at the same speed as the actual wheels
        return inval * 2 * 3.14 / (12 * 81) * 2;
    }

    // shamelessly coppied from the arduino standard library
    double MotorNodePlugin::map(double x, double in_min, double in_max, double out_min, double out_max)
    {
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }
}
