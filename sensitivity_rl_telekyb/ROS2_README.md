# ROS2 Policy Controller

This package provides a ROS2 node for running trained reinforcement learning policies on a quadrotor with control system integration (via genomix components) for both simulation and real hardware.

## Installation

The package uses pixi with robostack to manage ROS2 Jazzy dependencies:

```bash
pixi install
```

## Quick Start

### Run with Simulation

Start the Gazebo simulation in one terminal:
```bash
pixi run sim-server
```

Then launch the policy control node in another terminal:
```bash
pixi run policy-control
```

### Run with Real Hardware

Simply launch the node (assumes hardware is connected and drivers initialized):
```bash
pixi run policy-control
```

The node automatically connects to whatever genomix is configured for (simulation or hardware).

## Control Sequence

Follow this sequence to operate the drone:

1. **Setup control system** (configure genomix components):
   ```bash
   pixi run policy-setup
   ```
   Or manually:
   ```bash
   ros2 service call /control/setup std_srvs/srv/Trigger
   ```

2. **Start control system** (spin up servo and logging):
   ```bash
   pixi run policy-start
   ```
   Or manually:
   ```bash
   ros2 service call /control/start std_srvs/srv/SetBool "{data: true}"
   ```

3. **Set initial setpoint** (optional):
   ```bash
   pixi run policy-setpoint -- --x 0.0 --y 0.0 --z 1.0 --yaw 0.0
   ```
   Or manually:
   ```bash
   ros2 topic pub /policy/setpoint geometry_msgs/msg/PoseStamped "{
     header: {frame_id: 'world'},
     pose: {
       position: {x: 0.0, y: 0.0, z: 1.0},
       orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}
     }
   }"
   ```

4. **Start policy loop**:
   ```bash
   pixi run policy-loop-start
   ```
   Or manually:
   ```bash
   ros2 service call /policy/start std_srvs/srv/SetBool "{data: true}"
   ```

5. **Update setpoint** (while running):
   ```bash
   pixi run policy-setpoint -- --x 1.0 --y 0.5 --z 1.5 --yaw 0.0
   ```
   Or manually:
   ```bash
   ros2 topic pub /policy/setpoint geometry_msgs/msg/PoseStamped "{
     header: {frame_id: 'world'},
     pose: {
       position: {x: 1.0, y: 0.5, z: 1.5},
       orientation: {w: 0.707, x: 0.0, y: 0.0, z: 0.707}
     }
   }"
   ```

6. **Land** (emergency descent with zero thrust):
   ```bash
   pixi run policy-land
   ```
   Or manually:
   ```bash
   ros2 service call /policy/land std_srvs/srv/Trigger
   ```

7. **Stop control system**:
   ```bash
   pixi run policy-stop
   ```
   Or manually:
   ```bash
   ros2 service call /control/stop std_srvs/srv/Trigger
   ```

## Services

| Service | Type | Purpose |
|---------|------|---------|
| `/control/setup` | `std_srvs/Trigger` | Configure control system (genomix components) |
| `/control/start` | `std_srvs/SetBool` | Start control system (spin up servo and logging) |
| `/control/stop` | `std_srvs/Trigger` | Stop control system |
| `/policy/start` | `std_srvs/SetBool` | Start policy execution loop |
| `/policy/stop` | `std_srvs/Trigger` | Stop policy loop |
| `/policy/land` | `std_srvs/Trigger` | Emergency land (zero thrust override) |

## Topics

### Subscribed
- `/policy/setpoint` (geometry_msgs/PoseStamped): Target position (xyz) and yaw from quaternion

## Parameters

- `policy_dir`: Path to trained policy directory (default: `configs/policy/default`)
- `frequency_hz`: Policy loop frequency in Hz (default: `100.0`)

Set parameters when running:
```bash
ros2 run . policy_control_node.py --ros-args -p policy_dir:=/path/to/policy -p frequency_hz:=100.0
```

## Architecture

The policy control system operates with:
- **Control system** (via genomix): Manages low-level servo control and hardware integration
- **Policy loop**: Runs in a separate thread at the specified frequency, allowing:
  - Dynamic setpoint updates without restarting
  - Clean stop/land procedures that override policy commands
  - Non-blocking ROS2 service calls

## Pixi Tasks Reference

### Core System Tasks

| Task | Command | Description |
|------|---------|-------------|
| `components` | `pixi run components` | Start the genomix component server |
| `web-gui` | `pixi run web-gui` | Launch the web-based GUI for visualization |
| `policy-control` | `pixi run policy-control` | Run the policy control node |
| `quadrotor-control` | `pixi run quadrotor-control` | Interactive Python console for quadrotor control |

### Policy Control Tasks (Convenience Wrappers)

| Task | Arguments | Description |
|------|-----------|-------------|
| `policy-setup` | None | Call `/control/setup` service |
| `policy-start` | None | Call `/control/start` service (data: true) |
| `policy-stop` | None | Call `/control/stop` service |
| `policy-loop-start` | None | Call `/policy/start` service (data: true) |
| `policy-loop-stop` | None | Call `/policy/stop` service |
| `policy-land` | None | Call `/policy/land` service |
| `policy-setpoint` | `--x`, `--y`, `--z`, `--yaw` | Publish target setpoint (defaults: 0.0, 0.0, 1.0, 0.0) |

Example:
```bash
pixi run policy-setpoint --x 1.5 --y -0.5 --z 2.0 --yaw 3.14
```

### Simulation Tasks

| Task | Arguments | Description |
|------|-----------|-------------|
| `sim` | `--world_file` | Run Gazebo simulation with visualization (default: `worlds/example.world`) |
| `sim-server` | `--world_file` | Run Gazebo simulation in server mode (default: `worlds/example.world`) |
| `sim-remote` | `--world_file` | Run Gazebo simulation with web interface (default: `worlds/example.world`) |
| `sim-reset` | `--world_name` | Reset the simulation world (default: `mrsim`) |

Examples:
```bash
# Run simulation with custom world
pixi run sim -- --world_file worlds/example_camera.world

# Reset simulation
pixi run sim-reset

# Run remote simulation with web GUI
pixi run sim-remote
```

### Build and Development Tasks

| Task | Description |
|------|-------------|
| `checkout` | Checkout telekyb3 repository |
| `build` | Build telekyb3 components |

## Typical Workflow

### Option 1: Simulation Workflow

```bash
# Terminal 1: Start simulation
pixi run sim-server

# Terminal 2: Start components
pixi run components

# Terminal 3: Run policy control
pixi run policy-control

# Terminal 4: Execute control sequence
pixi run policy-setup
pixi run policy-start
pixi run policy-setpoint --x 0.0 --y 0.0 --z 1.0
pixi run policy-loop-start
# ... update setpoints as needed
pixi run policy-land
pixi run policy-loop-stop
pixi run policy-stop
```

### Option 2: Real Hardware Workflow

```bash
# Terminal 1: Start components (hardware drivers)
pixi run components

# Terminal 2: Run policy control
pixi run policy-control

# Terminal 3: Execute control sequence
pixi run policy-setup
pixi run policy-start
pixi run policy-setpoint --x 0.0 --y 0.0 --z 1.0
pixi run policy-loop-start
# ... update setpoints as needed
pixi run policy-land
pixi run policy-loop-stop
pixi run policy-stop
```
