//
// Created by david on 1/24/18.
//

#include <cmath>
#include <ctime>
#include <chrono>
#include <vector>

#include <ros/ros.h>
#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>


#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Int8.h"


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

        // for controlling the arm joint angles
        physics::JointController* j2_controller;


        // Making a seperate subscriber and callback for each motor was intentional, some of the motors might need
        // to be treated differently from the others (like they might need different torque scaling or something)
        ros::Subscriber backLeftMotorSub;
//        ros::Subscriber middleLeftMotorSub;
        ros::Subscriber frontLeftMotorSub;
        ros::Subscriber backRightMotorSub;
//        ros::Subscriber middleRightMotorSub;
        ros::Subscriber frontRightMotorSub;

        ros::Subscriber ArmSub0;
        ros::Subscriber ArmSub1;
        ros::Subscriber ArmSub2;
        ros::Subscriber ArmSub3;
        ros::Subscriber ArmSub4;
        ros::Subscriber ArmSub5;


        double backLeftMotorVal = 0;
        double middleLeftMotorVal = 0;
        double frontLeftMotorVal = 0;
        double backRightMotorVal = 0;
        double middleRightMotorVal = 0;
        double frontRightMotorVal = 0;
        double armJointVals[6];

        void backLeftMotorCallback(const std_msgs::Int8::ConstPtr& msg);

        void middleLeftMotorCallback(const std_msgs::Int8::ConstPtr& msg);
        void frontLeftMotorCallback(const std_msgs::Int8::ConstPtr& msg);
        void backRightMotorCallback(const std_msgs::Int8::ConstPtr& msg);
        void middleRightMotorCallback(const std_msgs::Int8::ConstPtr& msg);
        void frontRightMotorCallback(const std_msgs::Int8::ConstPtr& msg);
        void ArmCallback0(const std_msgs::Int8::ConstPtr& msg);
        void ArmCallback1(const std_msgs::Int8::ConstPtr& msg);
        void ArmCallback2(const std_msgs::Int8::ConstPtr& msg);
        void ArmCallback3(const std_msgs::Int8::ConstPtr& msg);
        void ArmCallback4(const std_msgs::Int8::ConstPtr& msg);
        void ArmCallback5(const std_msgs::Int8::ConstPtr& msg);


        double mapMotorTorque(double inval);

        double map(double x, double in_min, double in_max, double out_min, double out_max);


    };

    // Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(MotorNodePlugin)




    void MotorNodePlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
        for(int i = 0; i < 6; i++) {
                armJointVals[i] = 0;
        }

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

        /*
        physics::PhysicsEnginePtr physics = world->GetPhysicsEngine();
        const std::string frictionModel = "cone_model";
        physics->SetParam("friction_model", frictionModel);
        */

        this->j2_controller = new physics::JointController(this->model);


        // All ROS stuff
        this->rosNodeHandle.reset(new ros::NodeHandle("gazebo_motor_node"));

        backLeftMotorSub = rosNodeHandle->subscribe("/motors/backLeft", 1, &MotorNodePlugin::backLeftMotorCallback, this);
//        middleLeftMotorSub = rosNodeHandle->subscribe("/motors/middleLeft", 1000, &MotorNodePlugin::middleLeftMotorCallback, this);
        frontLeftMotorSub = rosNodeHandle->subscribe("/motors/frontLeft", 1, &MotorNodePlugin::frontLeftMotorCallback, this);
        backRightMotorSub = rosNodeHandle->subscribe("/motors/backRight", 1, &MotorNodePlugin::backRightMotorCallback, this);
//        middleRightMotorSub = rosNodeHandle->subscribe("/motors/middleRight", 1000, &MotorNodePlugin::middleRightMotorCallback, this);
        frontRightMotorSub = rosNodeHandle->subscribe("/motors/frontRight", 1, &MotorNodePlugin::frontRightMotorCallback, this);
        
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

//            ROS_INFO("Simulator Motor Node Started");
    }



    void MotorNodePlugin::backLeftMotorCallback(const std_msgs::Int8::ConstPtr& msg) {
        backLeftMotorVal = msg->data;
#ifdef DEBUG_PRINT
        ROS_INFO("I heard: [%s]", std::to_string(msg->data).c_str());
#endif
    }

    void MotorNodePlugin::middleLeftMotorCallback(const std_msgs::Int8::ConstPtr& msg) {
        middleLeftMotorVal = msg->data;
    }

    void MotorNodePlugin::frontLeftMotorCallback(const std_msgs::Int8::ConstPtr& msg) {
        frontLeftMotorVal = msg->data;
    }

    void MotorNodePlugin::backRightMotorCallback(const std_msgs::Int8::ConstPtr& msg) {
        backRightMotorVal = msg->data;
    }

    void MotorNodePlugin::middleRightMotorCallback(const std_msgs::Int8::ConstPtr& msg) {
        middleRightMotorVal = msg->data;
    }


    void MotorNodePlugin::frontRightMotorCallback(const std_msgs::Int8::ConstPtr& msg) {
        frontRightMotorVal = msg->data;
    }

    void MotorNodePlugin::ArmCallback0(const std_msgs::Int8::ConstPtr& msg) {
        armJointVals[0] = (double) msg->data;
        this->j2_controller->SetJointPosition(this->model->GetJoint("armbase_armcentershaft"), map(armJointVals[0], -128, 127, -3.14, 3.14));
    }

    void MotorNodePlugin::ArmCallback1(const std_msgs::Int8::ConstPtr& msg) {
        armJointVals[1] = (double) msg->data;
        this->j2_controller->SetJointPosition(this->model->GetJoint("armcentershaftoffset_backarm"), map(armJointVals[1], -128, 127, -3.14, 3.14));
    }

    void MotorNodePlugin::ArmCallback2(const std_msgs::Int8::ConstPtr& msg) {
        armJointVals[2] = (double) msg->data;
        this->j2_controller->SetJointPosition(this->model->GetJoint("backarm_forearm"), map(armJointVals[2], -128, 127, -3.14, 3.14));
    }

    void MotorNodePlugin::ArmCallback3(const std_msgs::Int8::ConstPtr& msg) {
        armJointVals[3] = (double) msg->data;
    }

    void MotorNodePlugin::ArmCallback4(const std_msgs::Int8::ConstPtr& msg) {
        armJointVals[4] = (double) msg->data;
    }

    void MotorNodePlugin::ArmCallback5(const std_msgs::Int8::ConstPtr& msg) {
        armJointVals[5] = (double) msg->data;
    }



    void MotorNodePlugin::OnUpdate() {

        u_int joint_axis = 0;
        this->model->GetJoint("left_back_wheel_hinge")->SetForce(joint_axis, mapMotorTorque(backLeftMotorVal));
        this->model->GetJoint("left_front_wheel_hinge")->SetForce(joint_axis,  mapMotorTorque(frontLeftMotorVal));
        this->model->GetJoint("right_back_wheel_hinge")->SetForce(joint_axis,  mapMotorTorque(backRightMotorVal));
        this->model->GetJoint("right_front_wheel_hinge")->SetForce(joint_axis, mapMotorTorque(frontRightMotorVal));

        // this->model->GetJoint("left_back_wheel_hinge")->SetParam("vel", joint_axis, mapMotorTorque(backLeftMotorVal));
        // this->model->GetJoint("left_front_wheel_hinge")->SetParam("vel", joint_axis,  mapMotorTorque(frontLeftMotorVal));
        // this->model->GetJoint("right_back_wheel_hinge")->SetParam("vel", joint_axis,  mapMotorTorque(backRightMotorVal));
        // this->model->GetJoint("right_front_wheel_hinge")->SetParam("vel", joint_axis, mapMotorTorque(frontRightMotorVal));
        
        // this->j2_controller->SetJointPosition(this->model->GetJoint("armbase_armcentershaft"), map(armJointVals[0], -128, 127, -3.14, 3.14));
        // this->j2_controller->SetJointPosition(this->model->GetJoint("armcentershaftoffset_backarm"), map(armJointVals[1], -128, 127, -3.14, 3.14));
        // this->j2_controller->SetJointPosition(this->model->GetJoint("backarm_forearm"), map(armJointVals[2], -128, 127, -3.14, 3.14));
    }


    double MotorNodePlugin::mapMotorTorque(double inval) {
        // This line should be some kind of scalar so that the simulated rover accelerates at about the same rate as the actual motor.
        return inval * 0.1;
    }


    void testCallback(const std_msgs::Int8::ConstPtr& msg) {
        std::cout << "test" << std::endl;
    }


    double MotorNodePlugin::map(double x, double in_min, double in_max, double out_min, double out_max)
    {
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }
}
