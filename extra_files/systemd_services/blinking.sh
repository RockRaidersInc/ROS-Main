#!/bin/bash

DEVICE='/dev/serial/by-id/***/'
NODE='move_base'
# b for blink? dunno. Echo will add a newline
ARDUINO_COMMAND='b'
BAUD=115200

while :
do

    while [ ! -f $DEVICE ]
    do
        # Loop until the device is found
        sleep 5
        echo 'looking for device'
    done
    echo 'found device'
    
    stty $BAUD -F $DEVICE 

    while :
    do
        sleep 2
        if [ ! -f $DEVICE ]; then
            break
        fi

        # Check for move base
        if rosnode list | grep $NODE; then
            echo $ARDUINO_COMMAND > $DEVICE
            echo 'sending arduino command'
        fi

    done

done
