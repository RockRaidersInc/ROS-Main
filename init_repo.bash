#!/bin/bash
echo GETTING USB CAMERA PACKAGE...
git submodule init
git submodule update
cd ..
echo BUILDING PACKAGES...
source /opt/ros/indigo/setup.bash
catkin_make
source ./devel/setup.bash
cd src
echo DONE!