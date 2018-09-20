#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH
cd $SCRIPTPATH

#source /opt/ros/kinetic/setup.bash
source ../../devel/setup.bash

gpsd -S 4000 /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A505WWCA-if00-port0
rosrun gpsd_client gpsd_client _host:=localhost _port:=4000
