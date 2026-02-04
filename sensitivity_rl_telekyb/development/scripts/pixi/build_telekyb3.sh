#!/bin/sh

set -e

cd $CONDA_PREFIX
cd src

# TODO:
# compiler options ? will include them in binaries packaques
# ${CFLAGS:=-g -O3 -Wall}
# ${CXXFLAGS:=${CFLAGS}}

# openrobots-idl: common genom IDL
echo "=========================================="
echo "Starting build of openrobots-idl ..."
echo "=========================================="
cd openrobots-idl
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# genom3
echo "=========================================="
echo "Starting build of genom3 ..."
echo "=========================================="
cd ../..
cd genom3
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# genomix
echo "=========================================="
echo "Starting build of genomix ..."
echo "=========================================="
cd ../..
cd genomix
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# python-genomix
echo "=========================================="
echo "Starting build of python-genomix ..."
echo "=========================================="
cd ../..
cd python-genomix
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# pocolibs middleware
echo "=========================================="
echo "Starting build of pocolibs ..."
echo "=========================================="
cd ../..
cd pocolibs
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# pocolibs genom3 template
echo "=========================================="
echo "Starting build of genom3-pocolibs ..."
echo "=========================================="
cd ../..
cd genom3-pocolibs
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# libkdtp
echo "=========================================="
echo "Starting build of libkdtp ..."
echo "=========================================="
cd ../..
cd libkdtp
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# qualisys SDK
echo "=========================================="
echo "Starting build of qualisys SDK ..."
echo "=========================================="
cd ../..
cd qualisys_cpp_sdk && mkdir -p build && cd build
cmake -S .. -B . -Dqualisys_cpp_sdk_OUTPUT_TYPE=SHARED -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release
make install -j${CPU_COUNT}

# ----------------
# telekyb3 modules
# ----------------

# rotorcraft-genom3
echo "=========================================="
echo "Starting build of rotorcraft-genom3 ..."
echo "=========================================="
cd ../..
cd rotorcraft-genom3
autoreconf -vif
mkdir -p build && cd build
../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# nhfc-genom3
echo "=========================================="
echo "Starting build of nhfc-genom3 ..."
echo "=========================================="
cd ../..
cd nhfc-genom3
autoreconf -vif
mkdir -p build && cd build
../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# uavatt-genom3
echo "=========================================="
echo "Starting build of uavatt-genom3 ..."
echo "=========================================="
cd ../..
cd uavatt-genom3
autoreconf -vif
mkdir -p build && cd build
../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# uavpos-genom3
echo "=========================================="
echo "Starting build of uavpos-genom3 ..."
echo "=========================================="
cd ../..
cd uavpos-genom3
autoreconf -vif
mkdir -p build && cd build
../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# maneuver-genom3
echo "=========================================="
echo "Starting build of maneuver-genom3 ..."
echo "=========================================="
cd ../..
cd maneuver-genom3
autoreconf -vif
mkdir -p build && cd build
../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# qualisys-genom3
#echo "=========================================="
#echo "Starting build of qualisys-genom3 ..."
#echo "=========================================="
#cd ../..
#cd qualisys-genom3
#autoreconf -vif
#mkdir -p build && cd build
#../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX CPPFLAGS="$CPPFLAGS -I$CONDA_PREFIX/include/qualisys_cpp_sdk"
#make install -j${CPU_COUNT}

# ----------------
# OpenRobots modules
# ----------------

# pom-genom3
echo "=========================================="
echo "Starting build of pom-genom3 ..."
echo "=========================================="
cd ../..
cd pom-genom3
autoreconf -vif
mkdir -p build && cd build
../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# optitrack-genom3
echo "=========================================="
echo "Starting build of optitrack-genom3 ..."
echo "=========================================="
cd ../..
cd optitrack-genom3
autoreconf -vif
mkdir -p build && cd build
../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}


# dynamixel-genom3
echo "=========================================="
echo "Starting build of dynamixel-genom3 ..."
echo "=========================================="
cd ../..
cd dynamixel-genom3
autoreconf -vif
mkdir -p build && cd build
../configure --with-templates=pocolibs/server,pocolibs/client/c --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# ----------------
# OpenRobots Gazebo plugins
# ----------------

# optitrack-gazebo
echo "=========================================="
echo "Starting build of optitrack-gazebo ..."
echo "=========================================="
cd ../..
cd optitrack-gazebo
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# libmrsim
echo "=========================================="
echo "Starting build of libmrsim ..."
echo "=========================================="
cd ../..
cd libmrsim
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}

# dynamixel-genom3
echo "=========================================="
echo "Starting build of mrsim-gazebo ..."
echo "=========================================="
cd ../..
cd mrsim-gazebo
autoreconf -vif
mkdir -p build && cd build
../configure --prefix=$CONDA_PREFIX
make install -j${CPU_COUNT}
