#!/bin/bash

set -e

# Function to kill all pocolibs components
kill_all_pocolibs() {
    echo "Killing all pocolibs components..."
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
    echo "Done."
}

# Function to create all pocolibs components
create_all_pocolibs() {
    echo "Creating all pocolibs components..."
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
    echo "Done."
}

# Function to restart all pocolibs components
restart_all_pocolibs() {
    kill_all_pocolibs
    sleep 1
    create_all_pocolibs
}

# Display help message
show_help() {
    echo "Usage: $(basename "$0") [OPTION]"
    echo ""
    echo "Manage pocolibs components for TeleKyb3."
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message and exit"
    echo "  -k, --kill      Kill all pocolibs components"
    echo "  -r, --restart   Restart all pocolibs components (kill then create)"
    echo ""
    echo "If no option is provided, components will be created."
}

# Parse arguments
case "$1" in
    --help|-h)
        show_help
        ;;
    --kill|-k)
        kill_all_pocolibs
        ;;
    --restart|-r)
        restart_all_pocolibs
        ;;
    *)
        create_all_pocolibs
        ;;
esac
