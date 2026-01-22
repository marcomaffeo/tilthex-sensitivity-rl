
## Installation with pixi

1. Configure pixi following the instructions at [https://pixi.prefix.dev/latest/installation/](https://pixi.prefix.dev/latest/installation/)
2. Run the following commands for creating the environment as defined in `pixi.toml`, pulling the telekyb3 sources and installing them:
    ```
    pixi run checkout

    pixi run build
    ```

The rest of the commands are assumed to be run from a `pixi shell` or prefixed by the `pixi run` command when necessary

## Instructions

You will need at least three separate shells to run the simulation:

1. Run the genom components
    
       pixi run components
       
    or `sh scripts/components.sh`
       
3. (Optional) For remote development, you can launch the websocket server
    
       pixi run web-gui

    or `sh scripts/websocket.sh`

    Then open the browser on your local machine at [https://app.gazebosim.org/visualization](https://app.gazebosim.org/visualization) and connect to the simulation with URL
    
       wss://<REMOTE_IP>:9002

   Note: You might need to open the URL https://<REMOTE_IP>:9002 in a separate tab first to have the browser accept the SSL certificate

2. Launch the gazebo simulation (add -s for remote development)
    
       gz sim -v 4 -r worlds/example.world
    
4. Launch the control script, e.g.
    
       python -i src/tk3-quadrotor-control.py
    
## Available commands

Here is a list of all the pixi commands which can be used to run and manage the simulation, organized by category:

### Installation

- **checkout**: Check out and initialize the telekyb3 sources
  ```
  pixi run checkout
  ```

- **build**: Build the telekyb3 project
  ```
  pixi run build
  ```

### Simulation

- **components**: Launch the genom components
  ```
  pixi run components
  ```

- **web-gui**: Launch the websocket server for remote visualization
  ```
  pixi run web-gui
  ```

- **sim**: Launch the gazebo simulation with a specified world file (default: `worlds/example.world`)
  ```
  pixi run sim
  pixi run sim world_file=worlds/example_camera.world
  ```

- **sim-server**: Launch the gazebo simulation in server mode (headless)
  ```
  pixi run sim-server
  pixi run sim-server world_file=worlds/example_camera.world
  ```

- **sim-remote**: Launch both the websocket server and gazebo simulation in server mode for remote development
  ```
  pixi run sim-remote
  pixi run sim-remote world_file=worlds/example_camera.world
  ```

- **sim-reset**: Reset the gazebo simulation world
  ```
  pixi run sim-reset
  pixi run sim-reset world_name=mrsim
  ```

### Control

- **quadrotor-control**: Launch the interactive quadrotor control script
  ```
  pixi run quadrotor-control
  ```




## Notes
- The latest compatible gazebo version is 9.4.0. This is reflected in the way the websocket server is launched through gz-launch, and will require to be updated for `gz-sim>=10`
    
    
