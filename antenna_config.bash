#!/bin/bash
if [ $# -lt 1 ]; then
    echo "Please specify configuration type (rover or base)"
    exit
fi

# Determine desired static IP
if [ "$1" = "rover" ]; then
    STATIC_IP=192.168.1.1
elif [ "$1" = "base" ]; then
    STATIC_IP=192.168.1.9
else
    echo "Valid configuations are one of 'rover' or 'base'"
    exit
fi

# Check network device
ETH_DEVICE=eth0
if [ $# -ge 2 ]; then
    ETH_DEVICE=$2
fi
ifconfig $ETH_DEVICE &>/dev/null
if [ $? -ne 0 ]; then #Device was not found
    echo "ERROR: Network device "$ETH_DEVICE" could not be found!"
    exit 1
fi

# Configure IP
sudo ifconfig $ETH_DEVICE $STATIC_IP netmask 255.255.255.0

# Print confirmation
echo $ETH_DEVICE" configured for "$1" antenna use."
