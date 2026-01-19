#!/bin/bash
# Kill Gazebo simulation processes forcefully
pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "gz-launch" 2>/dev/null
pkill -9 -f "ruby.*gz" 2>/dev/null

# Clean up leftover transport files
rm -rf /tmp/gz-* ~/.gz/transport/ 2>/dev/null

echo "Simulation processes killed and transport cleaned up"
