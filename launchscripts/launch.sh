#!/bin/bash
source /opt/ros/kinetic/setup.bash
source ~/ws/devel/setup.bash
text1=$(curl -s http://whatismyip.akamai.com/)

echo "IP address is:"
echo $text1
echo 
echo
roslaunch drive startup.launch
q='quit'
while true
do
echo 'enter command'
read text
if text==q
do

done
