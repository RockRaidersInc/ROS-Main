# IGVC 2019 Snapshot

This repo contains all code physically on the rover for the IGVC 2019 competition. It exists as a backup and for future reference. A lot of the code here was hacked together and large parts do not work.

Simulation *should* work just fine though (it hasn't been tested). Running the code here requires Ubuntu 16 and ROS Kinetic. Navigate to src then run ./launchscripts/simulator.sh igvc17

The rest of the readme contains install instructions.


# ROS-Main

First install Gazebo. Use these commands:
```
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-latest.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update
sudo apt install -y gazebo7
```

First install ROS Kinetic. Instructions are here:
```
http://wiki.ros.org/kinetic/Installation/Ubuntu
```

Install some more tools:
```
sudo apt install -y cmake python-catkin-pkg python-empy python-nose libgtest-dev python-catkin-tools
sudo apt install -y ros-kinetic-joy ros-kinetic-geographic-msgs ros-kinetic-tf2-geometry-msgs ros-kinetic-move-base ros-kinetic-map-server ros-kinetic-global-planner sudo apt install ros-kinetic-eband-local-planner
sudo apt install -y ros-kinetic-desktop
sudo apt install -y ros-kinetic-gazebo-ros
sudo apt install -y ros-kinetic-pcl-ros
sudo apt install -y ros-kinetic-usb-cam
```

Also install the ZED SDK. Instructions are here:
```
https://www.stereolabs.com/developers/
```

Create a catkin workspace and clone our repository:
```
mkdir ~/URC
cd ~/URC
git clone --recurse-submodules https://github.com/RockRaidersInc/ROS-Main.git src
catkin build
```

While we're here, give yourself permission to open serial ports
```
sudo adduser $(whoami) dialout
```

If catkin build succeeded then test the install by running 
```
./src/launchscripts/simulator.sh igvc17
```
The simulator should start. If it crashes press control-c in the terminal and try it again. Gazebo often crashes on startup for unknonw reasons.


##Starting up the actual rover
On the rover run `roslaunch launch_files launch_with_hardware.launch`. 
Then, on the base station, run `source launchscripts/export_remote_ros_vars.sh` then `roslaunch launch_files base_station.launch` in the same terminal


##Setup and usage for usb_cam submodule

Once you have cloned your repository, if you intend on working with the USB cameras, you will need to initialize and update the usb_cam submodule.  To do so, simply run:
```
git submodule init
git submodule update
```


To start getting your images from the camera, you need to run the usb_cam_node as follows:
```
rosrun usb_cam usb_cam_node _video_device:=[path to your video device, /dev/video0 by default]
```
The full list of parameters appear [here](http://wiki.ros.org/usb_cam), others that may be useful might by image_width and image_height.
This node publishes images to `usb_cam/image_raw` in all supported image_transport formats (raw, compressed, compressedDepth, and theora).

To test that you have it setup correctly, you can view the video stream using the built in image_view node as follows:
```
rosrun image_view image_view image:=/usb_cam/image_raw
```

# RQT Mini Tutorial

To run ROS with UI, you must download and install rqt (http://wiki.ros.org/rqt), which I believe comes with default packages. Launch RQT by typing rqt into shell (after running roscore), then go to perspectives tab, click on import, then locate ui.perspective under ROS-Main/user_interface/config directory


# Other stuff that needs to be installed

```
sudo apt install -y ros-kinetic-joy  ros-kinetic-geographic-msgs
```

## autonomous traversal stuff
```
sudo apt install -y gpsd ros-kinetic-gpsd-client ros-kinetic-nmea-navsat-driver ros-kinetic-gps-common ros-kinetic-mapviz ros-kinetic-mapviz-plugins ros-kinetic-swri-transform-util ros-kinetic-robot-localization
```

## simulation stuff
```
sudo apt install -y libignition-math2-dev ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
```


## xbox controller setup
A driver needs to be running to handle xbox controllers. It can be installed with
```
sudo apt install -y xboxdrv
```
To use an Xbox controller, run this in a separate terminal:

```sudo xboxdrv --detach-kernel-driver```

## Rover specific setup
This section is about setting up the rover to run drive code on startup.
```
# navigate to the source directory and run 
cd extra_files/systemd_services && sudo ./install_services.sh

# make sure the service is running with
sudo systemctl status xboxdrv.service
```

Add the following lines to .bashrc
```
sudo systemctl kill drive_at_startup.service
source /opt/ros/kinetic/setup.bash
source ~/URC/devel/setup.bash
source ~/URC/src/launchscripts/export_remote_ros_vars.sh
```

To remove the xbox controller service 
```
cd extra_files/systemd_services && sudo ./remove_services.sh
```

# Random useful commands


