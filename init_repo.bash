#!/bin/bash
echo GETTING USB CAMERA PACKAGE...
git submodule init
git submodule update
cd ..
echo BUILDING PACKAGES...
if [ -d "/opt/ros/kinetic" ]; then
    source /opt/ros/kinetic/setup.bash
else
    source /opt/ros/indigo/setup.bash
fi
catkin_make
source ./devel/setup.bash
cd src
echo DONE!