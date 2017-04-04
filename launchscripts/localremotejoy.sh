#!/bin/bash
source /opt/ros/kinetic/setup.bash
#set to your workspace setup file
source ~/ws/devel/setup.bash


roslaunch drive joystick.launch

read
