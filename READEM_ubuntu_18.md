# ROS-Main

First install ROS melodic. Instructions are here:
```
http://wiki.ros.org/melodic/Installation/Ubuntu
```

Install some more tools:
```
sudo apt install -y cmake python-catkin-pkg python-empy python-nose libgtest-dev python-catkin-tools
sudo apt install -y ros-melodic-joy ros-melodic-geographic-msgs ros-melodic-tf2-geometry-msgs ros-melodic-move-base ros-melodic-map-server ros-kinetic-global-planner sudo apt install ros-kinetic-eband-local-planner
sudo apt install -y ros-melodic-desktop
sudo apt install -y ros-melodic-pcl-ros
sudo apt install -y ros-melodic-usb-cam
```

Next install Gazebo. Use this command:
```
sudo apt install -y ros-melodic-gazebo-ros

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
sudo apt install -y ros-melodic-joy  ros-melodic-geographic-msgs
```

## autonomous traversal stuff
```
sudo apt install -y gpsd ros-melodic-gpsd-client ros-melodic-nmea-navsat-driver ros-melodic-gps-common ros-melodic-mapviz ros-melodic-mapviz-plugins ros-melodic-swri-transform-util ros-melodic-robot-localization
```

## simulation stuff
```
sudo apt install -y libignition-math2-dev ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
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
source /opt/ros/melodic/setup.bash
source ~/URC/devel/setup.bash
source ~/URC/src/launchscripts/export_remote_ros_vars.sh
```

To remove the xbox controller service 
```
cd extra_files/systemd_services && sudo ./remove_services.sh
```

# Random useful commands


