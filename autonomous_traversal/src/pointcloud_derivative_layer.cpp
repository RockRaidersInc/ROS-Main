#include <math.h>
#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>

ros::Publisher pub;
ros::Publisher pub_unfiltered;
tf::TransformListener* tf_listener;

tf::StampedTransform base_link_to_camera_tf;


// Shamelessly coppied from the arduino standard library
// https://www.arduino.cc/reference/en/language/functions/math/map/
double inline map_range(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


float apply_3x3_kernel(std::vector<float> array, std::vector<float> kernel, int x, int y, int x_len) {
    float temp = 0;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            temp += array.at((y + j - 1) * x_len + (x + i - 1)) * kernel.at(j * 3 + i);
        }
    }
    return temp;
}

float apply_1x1_kernel(std::vector<float> array, std::vector<float> kernel, int x, int y, int x_len) {
    float temp = 0;
    for (int j = 0; j < 1; j++) {
        for (int i = 0; i < 1; i++) {
            temp += array.at((y + j) * x_len + (x + i)) * kernel.at(j * 1 + i);
        }
    }
    return temp;
}


bool is_finite(float num) {
    return (num != std::numeric_limits<float>::infinity()) && (num != -std::numeric_limits<float>::infinity()) && (! std::isnan(num));
}


// void timerCallback(const ros::TimerEvent&) {
void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input_cloud_msg) {
    float resolution = 0.2;

    // Container for original & filtered data
    //TODO: make a shared pointer, this is a memory leak?
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;  
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    pcl::PCLPointCloud2 cloud_filtered_2;

    // Convert to PCL data type
    pcl_conversions::toPCL(*input_cloud_msg, *cloud);

    // Perform the actual filtering
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud (cloudPtr);
    sor.setLeafSize (resolution/1.2, resolution/1.2, resolution/1.2);
    sor.filter (cloud_filtered_2);

    // Convert back to ROS data type
    sensor_msgs::PointCloud2 intermediate_cloud;
    pcl_conversions::fromPCL(cloud_filtered_2, intermediate_cloud);

    // now convert to the normal pointcloud datatype
    pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered_unrotated;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered;
    pcl::fromROSMsg (intermediate_cloud, cloud_filtered_unrotated);
    // pcl::fromROSMsg (*input_cloud_msg, cloud_filtered_unrotated);

    pcl_ros::transformPointCloud(cloud_filtered_unrotated, cloud_filtered, base_link_to_camera_tf.inverse());


    // cloud_filtered.points.clear();
    // cloud_filtered.height = 1;
    // for (float y = 0; y < 5; y = y + 0.1) {
    //     for (float x = 0; x < 5; x = x + 0.1) {
    //         pcl::PointXYZRGB new_point;
    //         new_point.x = x;
    //         new_point.y = y;
    //         if (sqrt((x-2)*(x-2) + (y-2)*(y-2)) < 1) {
    //             new_point.z = 0.25;
    //             new_point.r = 255;
    //             new_point.g = 0;
    //             new_point.b = 0;
    //         }
    //         else {
    //             new_point.z = 0;
    //             new_point.r = 255;
    //             new_point.g = 255;
    //             new_point.b = 255;
    //         }
    //         cloud_filtered.points.push_back(new_point);
    //     }
    // }
    // cloud_filtered.width = cloud_filtered.points.size();


    // sensor_msgs::PointCloud2 output_unfiltered;
    // pcl::toROSMsg(cloud_filtered, output_unfiltered);
    // output_unfiltered.header.frame_id = "/base_link";
    // pub_unfiltered.publish (output_unfiltered);


    float min_x = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    float min_z = std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();

    for (auto b1 = cloud_filtered.points.begin(); b1 < cloud_filtered.points.end(); b1++) {
        if (is_finite(b1->x) && is_finite(b1->y) && is_finite(b1->z)) {
            if (b1->x < min_x) min_x = b1->x;
            if (b1->x > max_x) max_x = b1->x;

            if (b1->y < min_y) min_y = b1->y;
            if (b1->y > max_y) max_y = b1->y;

            if (b1->z < min_z) min_z = b1->z;
            if (b1->z > max_z) max_z = b1->z;
        }
    }
    
    int x_len = ceil((max_x - min_x) / resolution);
    int y_len = ceil((max_y - min_y) / resolution);

    std::vector<float> max_heights(y_len * x_len, -std::numeric_limits<float>::infinity());
    std::vector<float> min_heights(y_len * x_len, std::numeric_limits<float>::infinity());

    for (auto b1 = cloud_filtered.points.begin(); b1 < cloud_filtered.points.end(); b1++) {
        float point_x = b1->x;
        float point_y = b1->y;
        float point_z = b1->z;

        if (std::isnan(point_x)) continue;
        if (std::isnan(point_y)) continue;
        if (std::isnan(point_z)) continue;

        int x_index = std::min(map_range(point_x, min_x, max_x, 0, x_len), x_len - 1.);
        int y_index = std::min(map_range(point_y, min_y, max_y, 0, y_len), y_len - 1.);

        if  (max_heights.at(y_index * x_len + x_index) < point_z) {
            max_heights.at(y_index * x_len + x_index) = point_z;
        }
        if  (min_heights.at(y_index * x_len + x_index) > point_z) {
            min_heights.at(y_index * x_len + x_index) = point_z;
        }
    }

    for (int y = 0; y < y_len; y++) {
        for (int x = 0; x < x_len; x++) {
            if (max_heights.at(y * x_len + x) == -std::numeric_limits<float>::infinity()) {
                max_heights.at(y * x_len + x) = std::numeric_limits<float>::quiet_NaN();
            }
            if (min_heights.at(y * x_len + x) == std::numeric_limits<float>::infinity()) {
                min_heights.at(y * x_len + x) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }


    // smooth the point cloud with an averaging kernel
    std::vector<float> max_heights_smoothed(max_heights);
    std::vector<float> min_heights_smoothed(min_heights);

    std::vector<float> gaussian_weights_3x3 = {0.077847, 0.123317, 0.077847, 0.123317, 0.195346, 0.123317, 0.077847, 0.123317, 0.077847};
    std::vector<float> gaussian_weights_1x1 = {1.0};

//    for (int x = 1; x < x_len - 1; x++) {
//        for (int y = 1; y < y_len - 1; y++) {
//            max_heights_smoothed.at(y * x_len + x) = apply_3x3_kernel(max_heights, gaussian_weights_3x3, x, y, x_len);
//            min_heights_smoothed.at(y * x_len + x) = apply_3x3_kernel(min_heights, gaussian_weights_3x3, x, y, x_len);
//            // max_heights_smoothed.at(y * x_len + x) = apply_1x1_kernel(max_heights, gaussian_weights_1x1, x, y, x_len);
//            // min_heights_smoothed.at(y * x_len + x) = apply_1x1_kernel(min_heights, gaussian_weights_1x1, x, y, x_len);
//        }
//    }

    max_heights_smoothed = max_heights;
    min_heights_smoothed = min_heights;


    std::vector<float> derivatives((y_len - 1) * (x_len - 1), std::numeric_limits<float>::quiet_NaN());
    for (int y = 0; y < y_len - 1; y++) {
        for (int x = 0; x < x_len - 1; x++) {

            float base = max_heights.at((y) * x_len + (x));
            float xp1 = max_heights_smoothed.at((y) * x_len + (x+1));
            // float xm1 = max_heights_smoothed.at((y) * x_len + (x-1));
            float yp1 = max_heights_smoothed.at((y+1) * x_len + (x));
            // float ym1 = max_heights_smoothed.at((y-1) * x_len + (x));

            float x_deriv = (xp1 - base) / (resolution);
            float y_deriv = (yp1 - base) / (resolution);
            
            float deriv_mag = std::min(sqrt(x_deriv * x_deriv + y_deriv * y_deriv) / 2.0, (double) 3);
            float height_diff = max_heights.at((y) * x_len + (x)) - min_heights.at((y) * x_len + (x));

            derivatives.at(y * (x_len - 1) + x) = std::max(deriv_mag, height_diff);
            // derivatives.at(y * (x_len - 1) + x) = deriv_mag;
            // derivatives.at(y * (x_len - 1) + x) = base;
        }
    }


    // make the z valued in the filtered point cloud equal to the derivatives 
    // (so colors are preserved in the output to make visualization better)

    pcl::PointCloud<pcl::PointXYZRGB> output_cloud;
    output_cloud.points.clear();
    output_cloud.height = 1;
    for (auto b1 = cloud_filtered.points.begin(); b1 < cloud_filtered.points.end(); b1++) {
        
        pcl::PointXYZRGB new_point = *b1;

        float point_x = new_point.x;
        float point_y = new_point.y;
        float point_z = new_point.z;

        if (std::isnan(point_x)) continue;
        if (std::isnan(point_y)) continue;
        if (std::isnan(point_z)) continue;

        int x_index = std::min(map_range(point_x, min_x, max_x, 0, x_len - 1) + resolution / 2.0, y_len - 2.);
        int y_index = std::min(map_range(point_y, min_y, max_y, 0, y_len - 1) + resolution / 2.0, y_len - 2.);

        if (0 < x_index && x_index < x_len - 1 &&
            0 < y_index && y_index < y_len - 1) {
            new_point.z = derivatives.at(y_index * (x_len - 1) + x_index);
            if (std::isnan(new_point.z)) continue;
            output_cloud.points.push_back(new_point);
        }

    }
    output_cloud.width = output_cloud.points.size();

    pcl::PointCloud<pcl::PointXYZRGB> cloud_rotated_back;
    pcl_ros::transformPointCloud(output_cloud, cloud_rotated_back, base_link_to_camera_tf);

    // Convert to ROS data type
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(cloud_rotated_back, output);
    // pcl::toROSMsg(output_cloud, output);
    // Publish the data
    output.header.seq = input_cloud_msg->header.seq;
    output.header.stamp = input_cloud_msg->header.stamp;
    output.header.frame_id = input_cloud_msg->header.frame_id;
    // output.header.frame_id = "/base_link";

    pub.publish (output);
    
}


int main (int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "pointcloud_derivative_node");
    ros::NodeHandle nh;

    tf_listener = new tf::TransformListener();
    while (nh.ok()){
        try{
            tf_listener->lookupTransform("/zed_camera_center", "/base_link", ros::Time(0), base_link_to_camera_tf);
            break;
        }
        catch (tf::TransformException &ex) {
            ROS_INFO("could not look up transform /base_link to /zed_camera_center");
            ROS_INFO("%s",ex.what());
            ros::Duration(1.0).sleep();
            continue;
        }
    }
    ROS_INFO("got transform, starting derivative node");


    // // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe("/zed/depth/depth_registered", 1, cloud_cb);
    // Create a ROS publisher for the output point cloud
//    pub_unfiltered = nh.advertise<sensor_msgs::PointCloud2>("/unfiltered", 1);
    pub = nh.advertise<sensor_msgs::PointCloud2>("/zed/depth/depth_registered_filtered", 1);

    // ros::Timer timer = nh.createTimer(ros::Duration(0.25), timerCallback);

    // Spin
    ros::spin();
}
















// void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input_cloud_msg)
// {
    // float resolution = 0.05;

    // // // Container for original & filtered data
    // // //TODO: make a shared pointer, this is a memory leak?
    // // pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;  
    // // pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    // // pcl::PCLPointCloud2 cloud_filtered_2;

    // // // Convert to PCL data type
    // // pcl_conversions::toPCL(*input_cloud_msg, *cloud);

    // // // Perform the actual filtering
    // // pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    // // sor.setInputCloud (cloudPtr);
    // // sor.setLeafSize (resolution, resolution, resolution);
    // // sor.filter (cloud_filtered_2);

    // // // Convert back to ROS data type
    // // sensor_msgs::PointCloud2 intermediate_cloud;
    // // pcl_conversions::fromPCL(cloud_filtered_2, intermediate_cloud);

    // // now convert to the normal pointcloud datatype
    // pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered_unrotated;
    // pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered;
    // // pcl::fromROSMsg (intermediate_cloud, cloud_filtered_unrotated);
    // // pcl::fromROSMsg (*input_cloud_msg, cloud_filtered_unrotated);

    // // pcl_ros::transformPointCloud(cloud_filtered_unrotated, cloud_filtered, base_link_to_camera_tf.inverse());


    // cloud_filtered.points.clear();
    // cloud_filtered.height = 1;
    // for (float y = 0; y < 10; y = y + 0.1) {
    //     for (float x = 0; x < 10; x = x + 0.1) {
    //         pcl::PointXYZRGB new_point;
    //         new_point.x = x;
    //         new_point.y = y;
    //         if (sqrt((x-5)*(x-5) + (y-5)*(y-5)) < 2) {
    //             new_point.z = 1;
    //             new_point.r = 255;
    //             new_point.g = 0;
    //             new_point.b = 0;
    //         }
    //         else {
    //             new_point.z = 0;
    //             new_point.r = 255;
    //             new_point.g = 255;
    //             new_point.b = 255;
    //         }

    //         cloud_filtered.points.push_back(new_point);
    //     }
    // }
    // cloud_filtered.width = cloud_filtered.points.size();


    // float min_x = std::numeric_limits<float>::infinity();
    // float max_x = -std::numeric_limits<float>::infinity();
    // float min_y = std::numeric_limits<float>::infinity();
    // float max_y = -std::numeric_limits<float>::infinity();
    // float min_z = std::numeric_limits<float>::infinity();
    // float max_z = -std::numeric_limits<float>::infinity();

    // for (auto b1 = cloud_filtered.points.begin(); b1 < cloud_filtered.points.end(); b1++) {
    //     if (b1->x < min_x) min_x = b1->x;
    //     if (b1->x > max_x) max_x = b1->x;

    //     if (b1->y < min_y) min_y = b1->y;
    //     if (b1->y > max_y) max_y = b1->y;

    //     if (b1->z < min_z) min_z = b1->z;
    //     if (b1->z > max_z) max_z = b1->z;
    // }
    
    // int x_len = ceil((max_x - min_x) / resolution);
    // int y_len = ceil((max_y - min_y) / resolution);

    // std::vector<float> max_heights(y_len * x_len, -std::numeric_limits<float>::infinity());
    // std::vector<float> min_heights(y_len * x_len, std::numeric_limits<float>::infinity());

    // for (auto b1 = cloud_filtered.points.begin(); b1 < cloud_filtered.points.end(); b1++) {
    //     float point_x = b1->x;
    //     float point_y = b1->y;
    //     float point_z = b1->z;

    //     if (std::isnan(point_x)) continue;
    //     if (std::isnan(point_y)) continue;
    //     if (std::isnan(point_z)) continue;

    //     int x_index = floor(map_range(point_x, min_x, max_x, 0, x_len - 1));
    //     int y_index = floor(map_range(point_y, min_y, max_y, 0, y_len - 1));
    //     if (max_heights.at(y_index * x_len + x_index) == -5.0001) {
    //         max_heights.at(y_index * x_len + x_index) = point_z;
    //     }
    //     if  (max_heights.at(y_index * x_len + x_index) < point_z) {
    //         max_heights.at(y_index * x_len + x_index) = point_z;
    //         // std::cout << "point_replaced, " << x_index << ", " << y_index << ", " << point_z <<  std::endl;
    //     }
    //     if (min_heights.at(y_index * x_len + x_index) == 5.0001) {
    //         min_heights.at(y_index * x_len + x_index) = point_z;
    //     }
    //     if  (min_heights.at(y_index * x_len + x_index) > point_z) {
    //         min_heights.at(y_index * x_len + x_index) = point_z;
    //         // std::cout << "point_replaced, " << x_index << ", " << y_index << ", " << point_z <<  std::endl;
    //     }
    // }

    // for (int y = 0; y < y_len; y++) {
    //     for (int x = 0; x < x_len; x++) {
    //         if (max_heights.at(y * x_len + x) == -std::numeric_limits<float>::infinity()) {
    //             max_heights.at(y * x_len + x) = std::numeric_limits<float>::quiet_NaN();
    //         }
    //         if (min_heights.at(y * x_len + x) == std::numeric_limits<float>::infinity()) {
    //             min_heights.at(y * x_len + x) = std::numeric_limits<float>::quiet_NaN();
    //         }
    //     }
    // }


    // // smooth the point cloud with an averaging kernel
    // std::vector<float> max_heights_smoothed(max_heights);
    // std::vector<float> min_heights_smoothed(min_heights);

    // std::vector<float> gaussian_weights_3x3 = {0.077847, 0.123317, 0.077847, 0.123317, 0.195346, 0.123317, 0.077847, 0.123317, 0.077847};
    // std::vector<float> gaussian_weights_1x1 = {1.0};

    // for (int x = 0; x < x_len; x++) {
    //     for (int y = 0; y < y_len; y++) {
    //         // max_heights_smoothed.at(y * x_len + x) = apply_3x3_kernel(max_heights, gaussian_weights_3x3, x, y, x_len);
    //         // min_heights_smoothed.at(y * x_len + x) = apply_3x3_kernel(min_heights, gaussian_weights_3x3, x, y, x_len);
    //         max_heights_smoothed.at(y * x_len + x) = apply_1x1_kernel(max_heights, gaussian_weights_1x1, x, y, x_len);
    //         min_heights_smoothed.at(y * x_len + x) = apply_1x1_kernel(min_heights, gaussian_weights_1x1, x, y, x_len);
    //     }
    // }


    // std::vector<float> derivatives(y_len * x_len, std::numeric_limits<float>::quiet_NaN());
    // for (int y = 1; y < y_len - 1; y++) {
    //     for (int x = 1; x < x_len - 1; x++) {
    //         if (max_heights_smoothed.at((y) * x_len + (x)) == -5.0001) continue;
    //         if (min_heights_smoothed.at((y) * x_len + (x)) == 5.0001) continue;

    //         float base = max_heights_smoothed.at((y) * x_len + (x));
    //         float xp1 = max_heights_smoothed.at((y) * x_len + (x+1));
    //         float xm1 = max_heights_smoothed.at((y) * x_len + (x-1));
    //         float yp1 = max_heights_smoothed.at((y+1) * x_len + (x));
    //         float ym1 = max_heights_smoothed.at((y-1) * x_len + (x));

    //         float x_deriv = (xp1 - xm1) / (resolution * 2);
    //         float y_deriv = (yp1 - ym1) / (resolution * 2);
            
    //         float deriv_mag = std::min(sqrt(x_deriv * x_deriv + y_deriv * y_deriv) / 10, 3.0);
    //         float height_diff = max_heights.at((y) * x_len + (x)) - min_heights.at((y) * x_len + (x));

    //         // derivatives.at(y * (x_len - 1) + x) = std::max(deriv_mag, height_diff);
    //         derivatives.at(y * (x_len - 1) + x) = base;
    //     }
    // }

    // // make the z valued in the filtered point cloud equal to the derivatives 
    // // (so colors are preserved in the output to make visualization better)
    // for (auto b1 = cloud_filtered.points.begin(); b1 < cloud_filtered.points.end(); b1++) {
    //     float point_x = b1->x;
    //     float point_y = b1->y;

    //     if (std::isnan(point_x)) continue;
    //     if (std::isnan(point_y)) continue;

    //     int x_index = floor(map_range(point_x, min_x, max_x, 0, x_len - 1));
    //     int y_index = floor(map_range(point_y, min_y, max_y, 0, y_len - 1));

    //     b1->z = derivatives.at(y_index * x_len + x_index);
    // }

    // // for (int y = 0; y < y_len; y++) {
    // //     for (int x = 0; x < x_len; x++) {
    // //         pcl::PointXYZRGB new_point;
    // //         new_point.x = map_range(x, 0, x_len - 1, min_x, max_x);
    // //         new_point.y = map_range(y, 0, y_len - 1, min_y, max_y);
    // //         new_point.z = min_heights_smoothed.at(y * x_len + x);
    // //         new_point.r = 255;
    // //         new_point.g = 255;
    // //         new_point.b = 255;
    // //         cloud_filtered.points.push_back(new_point);
    // //     }
    // // }
    // // cloud_filtered.width = cloud_filtered.points.size();


    // // pcl::PointCloud<pcl::PointXYZRGB> cloud_rotated_back;
    // // pcl_ros::transformPointCloud(cloud_filtered, cloud_rotated_back, base_link_to_camera_tf);

    // // Convert to ROS data type
    // sensor_msgs::PointCloud2 output;
    // // pcl::toROSMsg(cloud_rotated_back, output);
    // pcl::toROSMsg(cloud_filtered, output);
    // // Publish the data
    // output.header.seq = input_cloud_msg->header.seq;
    // output.header.stamp = input_cloud_msg->header.stamp;
    // // output.header.frame_id = input_cloud_msg->header.frame_id;
    // output.header.frame_id = "/base_link";

    // pub.publish (output);
// }