# ROS-Main

Welcome to Software!
Ask David or Sid about any software-related problems.


# Upgrade Instructions (from ubuntu 16)
Upgrading Ubuntu to from 16 to 18 requires uninstalling all ros-related packages. ```sudo apt remove ros*``` works well. Also remove ROS Kinetic from your sources list (it should have a line in /etc/apt/sources.list.d/ros-latest.list). Then remove gazebo. Run ```sudo apt remove gazebo*``` and ```sudo apt remove libignition*``` and comment out any lines in /etc/apt/sources.list.d/gazebo-latest.list. 
Then upgrade to ubuntu 18 and follow the install instructions to get everything back up and running again.

# Install Instructions:
First install ROS melodic. Instructions are here:
```
http://wiki.ros.org/melodic/Installation/Ubuntu
```

Install some tools:
```
sudo apt install -y cmake python-catkin-pkg python-empy python-nose libgtest-dev python-catkin-tools ros-melodic-desktop ros-melodic-joy ros-melodic-geographic-msgs ros-melodic-tf2-geometry-msgs ros-melodic-move-base ros-melodic-map-server ros-melodic-global-planner ros-melodic-pcl-ros ros-melodic-usb-cam ros-pcl-msgs ros-melodic-key-teleop ros-melodic-joy  ros-melodic-geographic-msgs gpsd ros-melodic-gpsd-client ros-melodic-nmea-navsat-driver ros-melodic-gps-common ros-melodic-swri-transform-util ros-melodic-robot-localization ros-melodic-teb-local-planner ros-melodic-mapviz-plugins ros-melodic-tile-map ros-melodic-multires-image
```

Some installs to make python3 nodes work:
```
pip3 install rospkg
pip install timeout_decorator
```

Next install Gazebo. Use this command:
```
sudo apt install -y ros-melodic-gazebo-ros libignition-math2-dev ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control

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
The simulator should start. If it crashes press control-c in the terminal and try it again. Gazebo often crashes on startup for unknown reasons.


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

# Installing using WSL
The Windows Subsytem for Linux (WSL) allows for running Linux commands on Windows. This is not the ideal environment for ROS, and performance will be hurt, but ROS can be run under WSL.

NOTE: WSL 2 is out/will be out soon. These instructions have not been tested with WSL 2, however performance will likely be better if it works. You may consider installing WSL 2 but you may also have more issues.
Install WSL using the following instructions:

https://docs.microsoft.com/en-us/windows/wsl/install-win10

Use the Ubuntu distribution for best compatability with ROS. Make sure to install version 18.04.

Install a XServer such as XMing. This will allow running graphical linux programs under WSL.

Run the following commands to install ROS

```shell
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -sL "http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654" | sudo apt-key add
sudo apt update
sudo apt install -y ros-melodic-desktop-full
sudo rosdep init
rosdep update
```

Automatiacally source ros files with the following command:
```shell
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Automatically setup the XServer with the following command:
```shell
echo "export DISPLAY=:0" >> ~/.bashrc
echo "export GAZEBO_IP=127.0.0.1" >> ~/.bashrc
source ~/.bashrc
```

## Test your installation
Open 3 bash terminals

In the first run `roscore`

In the second run `rosrun turtlesim turtlesim_node`

In the third run `rosrun turtlesim turtle_teleop_key`

A window with a turtle should pop up. By selecting the window where you ran turtle_teleop_key and pressing arrow keys you should be able to move the turtle around.

Congrats you have installed ROS on WSL!

Talk to Sid if you have issues installing on WSL.
