//
// Created by david on 1/24/18.
//

#include <ros/ros.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

#include "SimMotorNode.h"



namespace gazebo
{
    /// \brief A plugin to control a Velodyne sensor.
    class VelodynePlugin : public ModelPlugin
    {
        /// \brief Constructor
    public: VelodynePlugin() {}

        /// \brief The load function is called by Gazebo when the plugin is
        /// inserted into simulation
        /// \param[in] _model A pointer to the model that this plugin is
        /// attached to.
        /// \param[in] _sdf A pointer to the plugin's SDF element.
    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
        {
            // Just output a message for now
            std::cerr << "\nThe velodyne plugin is attach to model[" <<
                      _model->GetName() << "]\n";
        }
    };

    // Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(VelodynePlugin)
}
