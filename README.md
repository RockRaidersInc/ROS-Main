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
sudo apt install -y ros-kinetic-joy ros-kinetic-geographic-msgs ros-kinetic-tf2-geometry-msgs ros-kinetic-move-base ros-kinetic-map-server ros-kinetic-global-planner
sudo apt install -y ros-kinetic-desktop
sudo apt install -y ros-kinetic-gazebo-ros
sudo apt install -y ros-kinetic-pcl-ros
sudo apt install -y ros-kinetic-usb-cam
```

Create a catkin workspace and clone our repository:
```
mkdir ~\URC
cd ~\URC
git clone https://github.com/RockRaidersInc/ROS-Main.git src
catkin build
```

If catkin build succeeded then test the install by running 
```
./src/launchscripts/simulator.sh igvc17
```
The simulator should start. If it crases don't fear, press control-c in the terminal and try it again. Gazebo often crashes on startup for unknonw reasons.


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
sudo apt install ros-kinetic-joy
sudo apt install ros-kinetic-geographic-msgs
```

## autonomous traversal stuff
```
sudo apt install gpsd ros-kinetic-gpsd-client
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


