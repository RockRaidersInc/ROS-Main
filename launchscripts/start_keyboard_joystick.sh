#!/bin/bash

# this changes the current directory to the catkin workspace where rockraiders code resides
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

#source /opt/ros/kinetic/setup.bash
source ../../devel/setup.bash

# go back to the original directory
cd -

rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/cmd_vel_user
