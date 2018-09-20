#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH
cd $SCRIPTPATH

#source /opt/ros/kinetic/setup.bash
source ../../devel/setup.bash

rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/cmd_vel
