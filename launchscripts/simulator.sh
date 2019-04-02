#!/bin/bash

green='\e[0;32m'
red='\e[0;31m'
endColor='\e[0m'

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
# echo $SCRIPTPATH
cd $SCRIPTPATH

# gazebo caches terrain files here. It does a really bad job of
# keeping different terrain files separate (sometimes the wrong
# maps show up), so delete the cache before every run.
rm -rf ~/.gazebo/paging/



if [ ${1} == "empty" ]
then
    export map="\$(find model_database)/../model_database/sim.world"

elif [ ${1} == "mdrs" ]
then
    export map="\$(find model_database)/../model_database/mars_desert_research_station_lower.world"

elif [ ${1} == "mountain" ]
then
    export map="\$(find model_database)/../model_database/mountain_populated.world"

elif [ ${1} == "86_field" ]
then
    export map="\$(find model_database)/../model_database/86_field.world"

elif [ ${1} == "maze_simple" ]
then
    export map="\$(find model_database)/../model_database/maze_simple.world"

elif [ ${1} == "maze_medium" ]
then
    export map="\$(find model_database)/../model_database/maze_medium.world"

elif [ ${1} == "maze_ramp" ]
then
    export map="\$(find model_database)/../model_database/maze_ramp.world"

elif [ ${1} == "mars" ]
then
    export map="\$(find model_database)/../model_database/mars_terrain_sample.world"

elif [ ${1} == "straight_empty" ]
then
    export map="\$(find model_database)/../model_database/igvc_straight_empty.world"

elif [ ${1} == "straight_pop" ]
then
    export map="\$(find model_database)/../model_database/igvc_straight_pop.world"

else
    echo -e ${red}The allowed maps are empyt, mdrs, mars, 86_filed, mountain, maze_simple, maze_medium, maze_ramp, straight_pop, or straight_empty ${endColor}
    echo -e ${red}example usage: ./launch_simulated.sh maze_simple ${endColor}
    exit
fi



#***********************************************************************************************************************
# There is a little bit of magic going on here. Basically, gzserver (part of gazebo) does not stop when it is
# sent SIGINT (control-c). This makes roslaunch hang after controll-c is sent and you either have to wait a while
# for roslaunch to forcibly kill gzserver or run "kill gzserver" in another shell. 
# 
# This monsterous snippit of shell code intercepts control-c, forcibly kills gazebo, then sends controll-c to
# roslaunch. I'm not entirely sure how it works, but it does. 
#       - David Michelman, Rock Raiders Vice President and OG Supreme Dictator
# 
#   PS: let me know if you read this (shoot me an email at daweim0@gmail.com). I sorta doubt anybody ever will. 
#***********************************************************************************************************************


(
    #source /opt/ros/kinetic/setup.bash
    source ../../devel/setup.bash

    # add some gazebo enviornment variables
    cd ../model_database
    source fix_gazebo_paths.sh
    cd ../launchscripts


    roslaunch rover_simulated start_with_simulator.launch sim_world_file:="$map" &
    PID=$!

    # catch control-c and kill gzserver (ROS has trouble killing it)
    other_commands() {
        echo -e ${red} "\nSIGTERM or SIGINT caught, killing gzserver and gzclient" ${endColor}
        pkill gzserver
        pkill gzclient
        kill $PID

        wait  # this wait waits for everythign to finish before the shell script exits
              # otherwise the bash prompt doesn't print after this script finishes
    }

    # I don't think both sigterm and sigint are needed
    trap 'other_commands' SIGINT
    trap 'other_commands' SIGTERM

    wait # this effectively pauses the shell script untill roslaunch exits
)

# echo -e ${green}script finished, press enter to see the prompt${endColor}