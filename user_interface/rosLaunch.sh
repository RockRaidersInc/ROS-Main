#!/bin/bash
bash -e roscore
bash -e rosrun turtlesim turtlesim_node
bash -e rqt --perspective-file "$(rospack find user_interface)/config/ui.perspective"
