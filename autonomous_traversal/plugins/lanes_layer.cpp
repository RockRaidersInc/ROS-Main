#include <lanes_layer/lanes_layer.h>
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(lanes_layer::LanesLayer, costmap_2d::Layer)

using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::NO_INFORMATION;

namespace lanes_layer
{
	LanesLayer::LanesLayer() {}

	void LanesLayer::onInitialize()
	{
		// Initialize
		ros::NodeHandle nh("~/" + name_);
		current_ = true;
		default_value_ = NO_INFORMATION;
		// rolling_window_ = layered_costmap_->isRolling();
		matchSize();

		// Dynamic Reconfig (not fully implemented)
		dsrv_ = new dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>(nh);
		dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>::CallbackType cb = 
			boost::bind( &LanesLayer::reconfigureCB, this, _1, _2);
		dsrv_->setCallback(cb);

		a = true;
	}


	void LanesLayer::matchSize()
	{
		Costmap2D* master = layered_costmap_->getCostmap();
		resizeMap(master->getSizeInCellsX(), 
				  master->getSizeInCellsY(), 
				  master->getResolution(),
				  master->getOriginX(), 
				  master->getOriginY());
	}


	void LanesLayer::reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level)
	{
		enabled_ = config.enabled;
	}

	void LanesLayer::updateBounds(double robot_x, double robot_y, double robot_yaw, 
								  double* min_x, double* min_y, 
								  double* max_x, double* max_y)
	{
		if (!enabled_)
			return;

		// Publish U shape on startup
		// Front left and back right corner in world coodinates
		if (a){
			x1 = robot_x + 1.5;	
			y1 = robot_y + 1.5;
			x2 = robot_x - 1.5;
			y2 = robot_y - 1.5;
			a = false;
		}
		// Front left and back right corner in map coordinates
		ROS_INFO("updateBounds Call");
		ROS_INFO("Test 1");
		unsigned int mx1,my1,mx2,my2; 	
		worldToMap(x1,y1,mx1,my1);
		worldToMap(x2,y2,mx2,my2);
		ROS_INFO("Test 2");
		// Start mark cells as obstacles
		MarkCell marker(costmap_, LETHAL_OBSTACLE); 
		raytraceLine(marker, mx1, my1, mx1, my2);
		raytraceLine(marker, mx1, my2, mx2, my2);
		raytraceLine(marker, mx2, my2, mx2, my1);
		ROS_INFO("Test 3");
		*min_x = std::min(*min_x, x1); 
		*min_x = std::min(*min_x, x2);
		*min_y = std::min(*min_y, y1); 
		*min_y = std::min(*min_y, y2);
		*max_x = std::max(*max_x, x1); 
		*max_x = std::max(*max_x, x1);
		*max_y = std::max(*max_y, y1); 
		*max_y = std::max(*max_y, y2);
		ROS_INFO("Test 4");
		ROS_INFO("(%.2f,%.2f,%.2f,%.2f)", *min_x, *min_y, *max_x, *max_y);
		ROS_INFO("(%.2f,%.2f) (%.2f,%.2f)",x1,y1,x2,y2);

		// double mark_x = robot_x + cos(robot_yaw);
		// double mark_y = robot_y + sin(robot_yaw);
		// unsigned int mx;
		// unsigned int my;
		// if(worldToMap(mark_x, mark_y, mx, my)) { setCost(mx, my, LETHAL_OBSTACLE); }
		
		// *min_x = std::min(*min_x, mark_x);
		// *min_y = std::min(*min_y, mark_y);
		// *max_x = std::max(*max_x, mark_x);
		// *max_y = std::max(*max_y, mark_y);
	}

	void LanesLayer::updateCosts(costmap_2d::Costmap2D& master_grid, 
								 int min_i, int min_j, 
								 int max_i, int max_j)
	{
		if (!enabled_)
			return;

		for (int j = min_j; j < max_j; j++)
		{
			for (int i = min_i; i < max_i; i++)
			{
				int index = getIndex(i, j);
				if (costmap_[index] == NO_INFORMATION)
					continue;
				master_grid.setCost(i, j, costmap_[index]); 
			}
		}

	}

} // end namespace