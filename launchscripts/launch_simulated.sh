
#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH
cd $SCRIPTPATH

#source /opt/ros/kinetic/setup.bash
source ../../devel/setup.bash

# add some gazebo enviornment variables
cd ../model_database
source fix_gazebo_paths.sh
cd ../launchscripts

# gazebo caches terrain files here. It does a really bad job of
# keeping different terrain files separate (sometimes the wrong
# maps show up), so delete the cache before every run.
rm -rf ~/.gazebo/paging/

# why do we need the ip address?
text1=$(curl -s http://whatismyip.akamai.com/)

echo "IP address is:"
echo $text1
echo 
echo
roslaunch rover_simulated start_with_simulator.launch

