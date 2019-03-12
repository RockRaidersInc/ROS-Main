# ROS-Main

This respository should be your src file in your ROS workspace.  So when cloning, call this from your workspace
```
git clone https://github.com/RockRaidersInc/ROS-Main.git src
```

The following commands should be run to install required packages:
```
# installing catkin (the newer version with `catkin build`)
sudo apt install cmake python-catkin-pkg python-empy python-nose libgtest-dev install python-catkin-tools

sudo apt install ros-kinetic-joy
```

##Starting up a simulator
The easiest way is to simply call the bash script `launch_simulated.sh` from the folder launchscripts
Then, in a new terminal, source devel/setup.bash and run `roslaunch launch_files base_station.launch` to get joystick support.

##Starting up the actual rover
On the rover run `roslaunch launch_files launch_with_hardware.launch`. 
Then, on the base station, run `source launchscripts/export_remote_ros_vars.sh` then `roslaunch launch_files base_station.launch` in the same terminal


##Setup and usage for usb_cam submodule

Once you have cloned your repository, if you intend on working with the USB cameras, you will need to initialize and update the usb_cam submodule.  To do so, simply run:
```
git submodule init
git submodule update
```
This will clone the usb_cam pakage into the correct directory.  You will then need to run `catkin_make` to build the new package and it should be ready to go

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
sudo apt install usb_cam
sudo apt install ros-kinetic-joy
```

## autonomous traversal stuff
```
sudo apt install gpsd ros-kinetic-gpsd-client
sudo apt install ros-kinetic-tf2-geometry-msgs ros-kinetic-move-base sudo apt install ros-kinetic-map-server
sudo apt install ros-kinetic-nmea-navsat-driver
```

## simulation stuff
```
sudo apt install libignition-math2-dev ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control
```


## xbox controller setup
A driver needs to be running to handle xbox controllers. It can be installed with
```
sudo apt install xboxdrv
```
To use an Xbox controller, run this in a separate terminal:

```sudo xboxdrv --detach-kernel-driver```

The previous command can be auto-started on boot like so:
```
# navigate to the source directory and run 
cd extra_files/systemd_services && sudo ./install_services.sh

# make sure the service is running with
sudo systemctl status xboxdrv.service
```

To remove the xbox controller service 
```
cd extra_files/systemd_services && sudo ./remove_services.sh
```

# Random useful commands


