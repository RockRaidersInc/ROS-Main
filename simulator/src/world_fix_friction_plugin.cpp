#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo
{
  class FixFrictionPlugin : public WorldPlugin
  {
    public: FixFrictionPlugin() : WorldPlugin() { }

    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
            {
                std::cerr << std::endl << "world_fix_friction_plugin is attached to the simulator" << std::endl;
    #if (GAZEBO_MAJOR_VERSION >= 8)
                physics::PhysicsEnginePtr physics = _world->Physics();
    #else
                physics::PhysicsEnginePtr physics = _world->GetPhysicsEngine();
    #endif
                const std::string frictionModel = "cone_model";
                physics->SetParam("friction_model", frictionModel);
            }
  };
  GZ_REGISTER_WORLD_PLUGIN(FixFrictionPlugin)
}
