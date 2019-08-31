#ifndef LANES_LAYER_H_
#define LANES_LAYER_H_
#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/observation_buffer.h>
#include <costmap_2d/GenericPluginConfig.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/Vector3.h>
#include <autonomous_traversal/Lane.h>
#include <tf/tf.h>

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

	private:
		void reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level);
		dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig> *dsrv_;
		void bufferLanePtsMsg(const autonomous_traversal::Lane::ConstPtr& lanes_msg);
		void processLanesMsgs(double robot_x, double robot_y, double robot_yaw);
		void resetCostmapLayer();

		ros::Subscriber lanes_sub;
		boost::mutex lanes_msg_mutex_;
  		std::list<autonomous_traversal::Lane> lanes_msgs_buffer_;
  		ros::Time last_reading_time_;	
  		double min_x_, min_y_, max_x_, max_y_;
		unsigned int buffered_readings_;
		bool rolling_window_;
		bool reset_costmap_flag;

		static inline void pointsTFToMsgs(std::vector<tf::Vector3>& tf_vect, 
										  std::vector<geometry_msgs::Point>& gm_vect)
		{
			for (std::vector<tf::Vector3>::iterator tf_pt_iter = tf_vect.begin();
				 tf_pt_iter != tf_vect.end(); tf_pt_iter++)
			{
				geometry_msgs::Point gm_pt;
				tf::pointTFToMsg(*tf_pt_iter, gm_pt);
				gm_vect.push_back(gm_pt);
			}
		}
	};
}
#endif
