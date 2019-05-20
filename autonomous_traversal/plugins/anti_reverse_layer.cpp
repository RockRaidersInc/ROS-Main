#include <anti_reverse_layer/anti_reverse_layer.h>
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(anti_reverse_layer::AntiReverseLayer, costmap_2d::Layer)

using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::NO_INFORMATION;

namespace anti_reverse_layer
{
	AntiReverseLayer::AntiReverseLayer() {}

	void AntiReverseLayer::onInitialize()
	{
		// Initialize
		ros::NodeHandle nh("~/" + name_);
		matchSize();
		u_obt_pts_sub = nh.subscribe("/anti_rev_u_pts", 100, &AntiReverseLayer::bufferUPtsMsg, this);
      	ROS_INFO("AntiReverseLayer: subscribed to topic %s", "/anti_rev_u_pts");

      	// Configurations
		current_ = true;
		default_value_ = NO_INFORMATION;
		rolling_window_ = layered_costmap_->isRolling();
		new_U_pts_flag = false;
		reset_costmap_flag = false;

		// Dynamic Reconfig (not fully implemented)
		dsrv_ = new dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>(nh);
		dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>::CallbackType cb = 
			boost::bind( &AntiReverseLayer::reconfigureCB, this, _1, _2);
		dsrv_->setCallback(cb);

	}


	void AntiReverseLayer::matchSize()
	{
		Costmap2D* master = layered_costmap_->getCostmap();
		resizeMap(master->getSizeInCellsX(), 
				  master->getSizeInCellsY(), 
				  master->getResolution(),
				  master->getOriginX(), 
				  master->getOriginY());
	}


	void AntiReverseLayer::reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level)
	{
		enabled_ = config.enabled;
	}

	void AntiReverseLayer::updateBounds(double robot_x, double robot_y, double robot_yaw, 
										double* min_x, double* min_y, 
										double* max_x, double* max_y)
	{
		if (!enabled_)
			return;
  		if (layered_costmap_->isRolling())
    		updateOrigin(robot_x - getSizeInMetersX() / 2, robot_y - getSizeInMetersY() / 2);

		// Publish virtual U shaped obstacle.
		// Takes in 4 points and published obstacle line between pt1-pt2, pt2-pt3, pt3-pt4
		if (new_U_pts_flag){
			ROS_INFO("New U obstacle published with corners at: "
					  "(%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)]",
				      U_pts[0], U_pts[1], U_pts[2], U_pts[3], 
				      U_pts[4], U_pts[5], U_pts[6], U_pts[7]);
			// Calculate U corners transformed
			double U_pts_tf[8];
			calcUCorners(robot_x, robot_y, robot_yaw, U_pts_tf);
			// Convert to map coordinates
			unsigned int U_pts_map[8];
			for (int i=0;i<8;i+=2)
				worldToMap(U_pts_tf[i], U_pts_tf[i+1], U_pts_map[i], U_pts_map[i+1]);
			// Mark cells located on the U using bresenhams line algorithm
			for (int i=0;i<6;i+=2)
			{
				MarkCell marker(costmap_, LETHAL_OBSTACLE); 
				raytraceLine(marker, U_pts_map[i], U_pts_map[i+1], U_pts_map[i+2], U_pts_map[i+3]);
			}
			// Update bounds
			for (int i=0;i<8;i+=2)
			{
				*min_x = std::min(*min_x, U_pts_tf[i]);
				*min_y = std::min(*min_y, U_pts_tf[i+1]);
				*max_x = std::max(*max_x, U_pts_tf[i]);
				*max_y = std::max(*max_y, U_pts_tf[i+1]);
			}
			// Reset flag
			new_U_pts_flag = false;
		}

		if (reset_costmap_flag)
		{
			ROS_INFO("Clearing anti reverse layer");
			// Clear costmap layer
			resetMap(0,0,size_x_,size_y_);
			// Update bounds
			*min_x = -std::numeric_limits<double>::max();  // Really small value to update full map
			*min_y = -std::numeric_limits<double>::max();
			*max_x =  std::numeric_limits<double>::max();  // Really large value to update full map
			*max_y =  std::numeric_limits<double>::max();
			// Reset flag
			reset_costmap_flag = false;
		}
		// ROS_INFO("MIN MAX1: (%.2f,%.2f),(%.2f,%.2f)", *min_x, *min_y, *max_x, *max_y); // Debug123

	}

	void AntiReverseLayer::updateCosts(costmap_2d::Costmap2D& master_grid, 
									   int min_i, int min_j, 
									   int max_i, int max_j)
	{
		if (!enabled_)
			return;

		// ROS_INFO("MIN MAX2: (%d,%d),(%d,%d)", min_i, min_j, max_i, max_j); // Debug123

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

	void AntiReverseLayer::calcUCorners(double robot_x, double robot_y, double robot_yaw, double U_pts_tf[8])
	{
		tf::Transform robot_tf;
		robot_tf.setOrigin(tf::Vector3(robot_x, robot_y, 0));
		robot_tf.setRotation(tf::createQuaternionFromYaw(robot_yaw));
		for (int i=0; i<8; i+=2)
		{
			tf::Vector3 pt(U_pts[i],U_pts[i+1],0);
			tf::Vector3 pt_tf = robot_tf(pt);
			U_pts_tf[i] = pt_tf.getX();
			U_pts_tf[i+1] = pt_tf.getY();
		}
	}

	void AntiReverseLayer::bufferUPtsMsg(const std_msgs::Float64MultiArray::ConstPtr& pts_msg)
	{
		std::vector<double> pts_msg_data = pts_msg->data;
		if (pts_msg_data.size() == 8)
		{
			// If the data field contains all 0's then reset the layer
			int num_zeros = 0;
			for (int i=0;i<pts_msg_data.size();i++)
				if (pts_msg_data[i] == 0) num_zeros++;
			if (num_zeros == 8)
				reset_costmap_flag = true;			
			
			// Buffer the U obst pts for publishing
			std::copy(pts_msg_data.begin(), pts_msg_data.end(), U_pts);
			new_U_pts_flag = true;
		}
		else ROS_WARN("Recieved U_pts message is not of size 8");
	}

} // end namespace