#!/bin/sh

set -e

cd $CONDA_PREFIX
mkdir src && cd src

# openrobots-idl: common genom IDL
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/openrobots-idl.git

# genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/genom3.git

# genomix
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/genomix.git

# python-genomix
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/genomix/python-genomix.git

# pocolibs middleware
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/pocolibs.git

# pocolibs genom3 template
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/genom3/genom3-pocolibs.git

# libkdtp
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/libkdtp.git

# qualisys SDK
git -c core.eol=lf -c core.autocrlf=false clone https://github.com/qualisys/qualisys_cpp_sdk.git

# ----------------
# telekyb3 modules
# ----------------

# rotorcraft-genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/telekyb3/rotorcraft-genom3

# nhfc-genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/telekyb3/nhfc-genom3.git

# uavatt-genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/telekyb3/uavatt-genom3.git

# uavpos-genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/telekyb3/uavpos-genom3.git

# maneuver-genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/telekyb3/maneuver-genom3.git

# # qualisys-genom3
# TODO checkout PR fix for SDK
#git -c core.eol=lf -c core.autocrlf=false clone --branch fix_qualisys_sdk_new_api git@gitlab.inria.fr:rainbow/software/telekyb3/qualisys-genom3.git

# ----------------
# OpenRobots modules
# ----------------

# pom-genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/pom-genom3.git

# optitrack-genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/optitrack-genom3.git


# dynamixel-genom3
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/dynamixel-genom3.git

# ----------------
# OpenRobots Gazebo plugins
# ----------------

# optitrack-gazebo
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/optitrack-gazebo.git

# libmrsim
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/libmrsim.git

# mrsim-gazebo
git -c core.eol=lf -c core.autocrlf=false clone git://git.openrobots.org/robots/mrsim-gazebo.git

