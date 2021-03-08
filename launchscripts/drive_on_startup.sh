# for printing in color
green='\e[0;32m'
red='\e[0;31m'
endColor='\e[0m'

# This file runs at startup on the rover and turns on the drive code if the wired logitek gamepad is plugged in

source /home/rock/URC/devel/setup.bash

# export ROS_HOSTNAME=$(hostname).local
# export ROS_MASTER_URI=http://rock-desktop.local:11311/


sudo modprobe pcspkr
beep -f 1000 -r 2 -l 250

roslaunch rover start_joystick.launch
exit 0



# this code makes sure that the USB stop button is plugged in before starting the drive code
# it was originally used as a saftey measure because the rover used to have a bad habbit of
# driving away on its own.
# We never figured out why, but it hasn't happened for quite a while now and I want the extra USB port

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

sudo modprobe pcspkr
beep -f 1000 -r 2 -l 250

roslaunch rover start_joystick.launch
