#ifndef LANES_LAYER_H_
#define LANES_LAYER_H_
#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/observation_buffer.h>
#include <costmap_2d/GenericPluginConfig.h>
#include <tf/transform_broadcaster.h>
#include <dynamic_reconfigure/server.h>

namespace lanes_layer
{
	class LanesLayer : public costmap_2d::Layer, public costmap_2d::Costmap2D
	{
	public:
		LanesLayer();
		bool isDiscretized() { return true; }

		virtual void onInitialize();
		virtual void matchSize();
		virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, 
								  double* min_x, double* min_y, 
								  double* max_x, double* max_y);
		virtual void updateCosts(costmap_2d::Costmap2D& master_grid, 
								 int min_i, int min_j, 
								 int max_i, int max_j);

		// void warpedLanesImgCallback(const sensor_msgs::ImageConstPtr& msg,
		// 							const boost::shared_ptr<costmap_2d::ObservationBuffer>& buffer);	

	private:
		void reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level);
		dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig> *dsrv_;
		void calcUCorners(double robot_x, double robot_y, double robot_yaw,
						  double U_pts[8], double tf_U_pts[8]);

		bool a;
		bool rolling_window_;
		double x1, y1, x2, y2, x3, y3, x4, y4;
	};
}
#endif