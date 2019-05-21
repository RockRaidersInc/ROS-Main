#include <lanes_layer/lanes_layer.h>
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(lanes_layer::LanesLayer, costmap_2d::Layer)

using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::NO_INFORMATION;
using costmap_2d::FREE_SPACE;

namespace lanes_layer
{
	LanesLayer::LanesLayer() {}

	void LanesLayer::onInitialize()
	{
		// Initialize
		ros::NodeHandle nh("~/" + name_);
		lanes_sub = nh.subscribe("/lanes", 100, &LanesLayer::bufferLanePtsMsg, this);
      	ROS_INFO("LanesLayer: subscribed to topic %s", "/lanes");
  		last_reading_time_ = ros::Time::now();
		min_x_ = min_y_ = -std::numeric_limits<double>::max();
		max_x_ = max_y_ = std::numeric_limits<double>::max();
		matchSize();

		// Configuration
		current_ = true;
		default_value_ = NO_INFORMATION;
		rolling_window_ = layered_costmap_->isRolling();
		reset_costmap_flag = false;

		// Dynamic Reconfig (not fully implemented)
		dsrv_ = new dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>(nh);
		dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>::CallbackType cb = 
			boost::bind( &LanesLayer::reconfigureCB, this, _1, _2);
		dsrv_->setCallback(cb);

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
  		if (layered_costmap_->isRolling())
    		updateOrigin(robot_x - getSizeInMetersX() / 2, robot_y - getSizeInMetersY() / 2);
		if (reset_costmap_flag)
			resetCostmapLayer();

		processLanesMsgs(robot_x, robot_y, robot_yaw);

		*min_x = std::min(*min_x, min_x_);
		*min_y = std::min(*min_y, min_y_);
		*max_x = std::max(*max_x, max_x_);
		*max_y = std::max(*max_y, max_y_);

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

		buffered_readings_ = 0;
		current_ = true;
	}

	void LanesLayer::bufferLanePtsMsg(const autonomous_traversal::LaneConstPtr& lane_pts_msg)
	{
		if (!enabled_)
			return;

		boost::mutex::scoped_lock lock(lanes_msg_mutex_);
  		lanes_msgs_buffer_.push_back(*lane_pts_msg);
	}

	void LanesLayer::processLanesMsgs(double robot_x, double robot_y, double robot_yaw)
	{
		// Load the lanes message data from the buffer
		std::list<autonomous_traversal::Lane> lanes_msgs_buffer_copy;
		lanes_msg_mutex_.lock();
		lanes_msgs_buffer_copy = std::list<autonomous_traversal::Lane>(lanes_msgs_buffer_);
		lanes_msgs_buffer_.clear();
		lanes_msg_mutex_.unlock();

		// Create robot transform
		tf::Transform robot_tf;
		robot_tf.setOrigin(tf::Vector3(robot_x, robot_y, 0));
		robot_tf.setRotation(tf::createQuaternionFromYaw(robot_yaw));

		// Iterate through the buffered Lanes messages
		for (std::list<autonomous_traversal::Lane>::iterator lane_msgs_itr = lanes_msgs_buffer_copy.begin();
       		 lane_msgs_itr != lanes_msgs_buffer_copy.end(); lane_msgs_itr++)
		{
			// // ----- Clear non lane points on the costmap layer -----
			// // Transform polygon coordinates into robot frame abd clear polygon
			// std::vector<geometry_msgs::Vector3> bound_polygon = lane_msgs_itr->bound_polygon;
			// std::vector<geometry_msgs::Point> bound_polygon_tf;
			// for (int i = 0; i < bound_polygon.size(); i++)
			// {
			// 	tf::Vector3 polygon_v3(bound_polygon[i].x, bound_polygon[i].y, 0);
			// 	polygon_v3 = robot_tf(polygon_v3); // Transform into robot frame
			// 	geometry_msgs::Point polygon_pt;
			// 	pointTFToMsg(polygon_v3, polygon_pt);
			// 	bound_polygon_tf.push_back(polygon_pt);
			// }
			// setConvexPolygonCost(bound_polygon_tf, FREE_SPACE);
			// // Touch all points
			// for (int i = 0; i < bound_polygon_tf.size(); i++)
			// {
			// 	min_x_ = std::min(bound_polygon_tf[i].x, min_x_);
			// 	min_y_ = std::min(bound_polygon_tf[i].y, min_y_);
			// 	max_x_ = std::max(bound_polygon_tf[i].x, max_x_);
			// 	max_y_ = std::max(bound_polygon_tf[i].y, max_y_);				
			// }

			// ------------- Add lane points to costmap -------------
			// Step through lane points
			std::vector<geometry_msgs::Vector3> lane_pts = lane_msgs_itr->lane_points;
			for (std::vector<geometry_msgs::Vector3>::iterator lane_pt_itr = lane_pts.begin();
				lane_pt_itr != lane_pts.end(); lane_pt_itr++)
			{
				// Mark points onto costmap
				tf::Vector3 lane_pt(lane_pt_itr->x, lane_pt_itr->y, 0);
				tf::Vector3 lane_pt_tf = robot_tf(lane_pt);
				unsigned int mx, my;
				if(worldToMap(lane_pt_tf.getX(), lane_pt_tf.getY(), mx, my)) 
					setCost(mx, my, LETHAL_OBSTACLE);
				// Touch
				min_x_ = std::min(lane_pt_tf.getX(), min_x_);
				min_y_ = std::min(lane_pt_tf.getY(), min_y_);
				max_x_ = std::max(lane_pt_tf.getX(), max_x_);
				max_y_ = std::max(lane_pt_tf.getY(), max_y_);
			}
			buffered_readings_++;
		}
	}

	void LanesLayer::resetCostmapLayer()
	{
		ROS_INFO("Clearing %s", name_.c_str());
		// Clear costmap layer
		resetMap(0,0,size_x_,size_y_);
		// Update bounds
		min_x_ = -std::numeric_limits<double>::max(); // Really small value to update full map
		min_y_ = -std::numeric_limits<double>::max();
		max_x_ = std::numeric_limits<double>::max();  // Really large value to update full map
		max_y_ = std::numeric_limits<double>::max();
		// Reset flag
		reset_costmap_flag = false;
	}
}