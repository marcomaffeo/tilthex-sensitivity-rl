#!/bin/bash

# Kill all pocolibs components
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
    
    h2 end 2>/dev/null || true
    h2 clean 2>/dev/null || true
    rm -f ~/.*.pid-* 2>/dev/null || true
    
    echo "Cleanup done."
}

# Start all pocolibs components
create_all_pocolibs() {
    echo "Creating all pocolibs components..."
    h2 init
    genomixd &

    # Motion capture: qualisys on the vehicle (aarch64), optitrack in simulation
    ARCH=$(uname -m)
    
    if [ "$ARCH" = "aarch64" ]; then
        echo "[Components] ARM64 target detected (Physical Jetson)."
        echo "[Components] Launching Qualisys Mocap..."
        qualisys-pocolibs -f &
    else
        echo "[Components] x86_64 target detected (Simulation PC)."
        echo "[Components] Launching Optitrack Mocap..."
        optitrack-pocolibs -f &
    fi
    
    rotorcraft-pocolibs -f &
    pom-pocolibs -f &
    nhfc-pocolibs -f &
    maneuver-pocolibs -f &
    
    echo "Creation done."
}

# Restart all pocolibs components
restart_all_pocolibs() {
    kill_all_pocolibs
    sleep 2 
    create_all_pocolibs
}

# Help message
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
    echo "Without options the components are restarted."
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