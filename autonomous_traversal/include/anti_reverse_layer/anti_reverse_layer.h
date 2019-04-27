#ifndef ANTI_REVERSE_LAYER_H_
#define ANTI_REVERSE_LAYER_H_
#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/observation_buffer.h>
#include <costmap_2d/GenericPluginConfig.h>
#include <tf/transform_broadcaster.h>
#include <dynamic_reconfigure/server.h>
#include <std_msgs/Float64MultiArray.h>

namespace anti_reverse_layer
{
	class AntiReverseLayer : public costmap_2d::Layer, public costmap_2d::Costmap2D
	{
	public:
		AntiReverseLayer();
		bool isDiscretized() { return true; }

		virtual void onInitialize();
		virtual void matchSize();
		virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, 
								  double* min_x, double* min_y, 
								  double* max_x, double* max_y);
		virtual void updateCosts(costmap_2d::Costmap2D& master_grid, 
								 int min_i, int min_j, 
								 int max_i, int max_j);

	private:
		void reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level);
		dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig> *dsrv_;
		void calcUCorners(double robot_x, double robot_y, double robot_yaw, double tf_U_pts[8]);
		void bufferUPtsMsg(const std_msgs::Float64MultiArray::ConstPtr& pts_msg);

		ros::Subscriber u_obt_pts_sub;
		bool rolling_window_;
		bool new_U_pts_flag;
		double U_pts[8]; // {x1,y1,x2,y2,x3,y3,x4,y4}
		bool reset_costmap_flag;
	};
}
#endif