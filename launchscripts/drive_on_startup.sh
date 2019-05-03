# for printing in color
green='\e[0;32m'
red='\e[0;31m'
endColor='\e[0m'

# This file runs at startup on the rover and turns on the drive code if the wired logitek gamepad is plugged in

source /home/rock/URC/devel/setup.bash

# export ROS_HOSTNAME=$(hostname).local
# export ROS_MASTER_URI=http://rock-desktop.local:11311/


echo "hi"

if [[ $(usb-devices | grep 'S:  Product=Gamepad F310') = 'S:  Product=Gamepad F310' ]];
then
	echo found usb gamepad, runing drive code;
else
	if [[ $(usb-devices | grep 'S:  Product=enCoReII Keyboard RDK') = 'S:  Product=enCoReII Keyboard RDK' ]]
	then
		echo found usb stop button, running drive code;
	else
		echo didn\'t find usb gamepad or usb stop button, not running drive code;
		exit 0
	fi
fi

# roslaunch base_station base_station.launch &

sleep 3

sudo modprobe pcspkr
beep -f 1000 -r 2 -l 250

roslaunch rover start_joystick.launch
