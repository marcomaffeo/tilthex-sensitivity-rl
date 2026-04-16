#!/bin/bash

# Function to kill all pocolibs components
kill_all_pocolibs() {
    echo "Killing all pocolibs components and cleaning shared memory..."
    
    pkill -f genomixd 2>/dev/null || true
    pkill -f pom-pocolibs 2>/dev/null || true
    pkill -f optitrack 2>/dev/null || true
    pkill -f qualisys 2>/dev/null || true
    pkill -f vicon 2>/dev/null || true
    pkill -f nhfc 2>/dev/null || true
    pkill -f maneuver 2>/dev/null || true
    pkill -f rotorcraft 2>/dev/null || true
    pkill -f dynamixel-pocolibs 2>/dev/null || true
    
    h2 end 2>/dev/null || true
    h2 clean 2>/dev/null || true
    rm -f ~/.*.pid-* 2>/dev/null || true
    
    echo "Cleanup done."
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
    echo "Creation done."
}

# Function to restart all pocolibs components
restart_all_pocolibs() {
    kill_all_pocolibs
    sleep 2 
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
    echo "  -c, --create    Create all components (without cleaning)"
    echo "  -r, --restart   Restart all pocolibs components (kill then create)"
    echo ""
    echo "If no option is provided, a RESTART will be executed by default."
}

# Parse arguments
case "$1" in
    --help|-h)
        show_help
        ;;
    --kill|-k)
        kill_all_pocolibs
        ;;
    --create|-c)
        create_all_pocolibs
        ;;
    *)
        restart_all_pocolibs
        ;;
esac