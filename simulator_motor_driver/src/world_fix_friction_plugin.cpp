#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo
{
  class FixFrictionPlugin : public WorldPlugin
  {
    public: FixFrictionPlugin() : WorldPlugin()
            {
              printf("Hello World!\n");
            }

    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
            {
                std::cerr << "\nworld_fix_friction_plugin is attached to the simulator" << std::endl;

                physics::PhysicsEnginePtr physics = _world->GetPhysicsEngine();
                const std::string frictionModel = "cone_model";
                physics->SetParam("friction_model", frictionModel);
            }
  };
  GZ_REGISTER_WORLD_PLUGIN(FixFrictionPlugin)
}