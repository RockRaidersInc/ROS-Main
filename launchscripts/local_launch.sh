# for printing in color
green='\e[0;32m'
red='\e[0;31m'
endColor='\e[0m'

echo -e ${red}this file is depreicated, strongly consider using \"source export_remote_ros_vars.sh\" instead ${endcolor}


export ROS_IP=$1
export ROS_MASTER_URI=http://rock-desktop.local:11311/

roslaunch launch_files start_rover_local.launch
