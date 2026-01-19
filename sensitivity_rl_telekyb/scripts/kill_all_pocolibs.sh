#!/bin/bash

### Description:
#
# Kill most common Genom3-modules.

pkill -f genomixd &
pkill pom-pocolibs &
pkill optitrack &
pkill qualisys &
pkill vicon &
pkill nhfc &
pkill maneuver &
pkill rotorcraft &
#pkill uavatt-pocolibs &
#pkill uavpos-pocolibs &
pkill -f dynamixel-pocolibs &
h2 end 


