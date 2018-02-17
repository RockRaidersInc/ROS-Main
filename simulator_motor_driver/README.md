This node acts as a motor driver for use in the gazebo simulator.


The whole simulator can be started by first running `source source fix_gazebo_paths.sh` then gazebo can be started with `roslaunch simulator_motor_driver simulator.launch`. Contact David (miched@rpi.edu) if it doesn't work, the paths are a little finicky and have only ben tested on one laptop. 


The error 
`[gazebo-2] process has died [pid 15828, exit code 255, cmd /opt/ros/kinetic/lib/gazebo_ros/gzserver -e ode /home/david/Documents/14_rockraiders/src/simulator_motor_driver/../model_database/sim.world __name:=gazebo __log:=/home/david/.ros/log/082227f6-06b8-11e8-b3c4-5c514f1cd27f/gazebo-2.log].` 
can be fixed by running the command `killall gzserver`. We don't know what causes the error yet.