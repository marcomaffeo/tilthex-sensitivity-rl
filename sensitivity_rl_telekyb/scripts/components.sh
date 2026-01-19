#!/bin/bash

set -e

h2 init

genomixd &

optitrack-pocolibs -f &
qualisys-pocolibs -f &
#vicon-pocolibs -f &
rotorcraft-pocolibs -f &
pom-pocolibs -f &
nhfc-pocolibs -f &
maneuver-pocolibs -f &
#uavatt-pocolibs -f &
#uavpos-pocolibs -f &
dynamixel-pocolibs -f &
