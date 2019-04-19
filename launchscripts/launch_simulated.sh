#!/bin/bash

green='\e[0;32m'
red='\e[0;31m'
endColor='\e[0m'

echo -e ${red}
echo \*
echo \*
echo \*
echo \*
echo \*
echo This file is deprecated and will be removed soon. Use simulator.sh to start the simulator
echo \*
echo \*
echo \*
echo \*
echo \*${endColor}

./simulator.sh ${@}
