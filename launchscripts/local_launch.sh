export ROS_IP=$1
export ROS_MASTER_URI=http://192.168.1.11:11311/

roslaunch launch_files start_rover_local.launch
