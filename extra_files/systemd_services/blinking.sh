#!/bin/bash

source /home/rock/URC/devel/setup.bash

DEVICE='/dev/serial/by-id/usb-Arduino_Srl_Arduino_Mega_7563331313335180E041-if00'
NODE='move_base'
# b for blink? dunno. Echo will add a newline
ARDUINO_COMMAND='a'
BAUD=9600

while :
do

    while [ ! -e $DEVICE ]
    do
        # Loop until the device is found
        sleep 5
        echo 'looking for device'
    done
    echo 'found device'
    
#    stty $BAUD -F $DEVICE 

    while :
    do
        sleep 2
        if [ ! -e $DEVICE ]; then
            break
        fi

        # Check for move base
        if rosnode list | grep $NODE; then
 #           echo $ARDUINO_COMMAND > $DEVICE
            python /home/rock/URC/src/extra_files/systemd_services/blink.py
            echo 'sending arduino command'
        fi

    done

done

