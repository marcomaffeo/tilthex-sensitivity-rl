
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
    
        sh scripts/components.sh
    
2. Launch the gazebo simulation
    
        gz sim -v 4 -r -s worlds/example.world
    
3. Launch the control script, e.g.
    
        python3 -i ./tk3-quadrotor-control.py
    
4. (Optional) For remote development, you can launch the websocket server
    
        sh scripts/websocket.sh
    
    Then open the browser on your local machine at [https://app.gazebosim.org/visualization](https://app.gazebosim.org/visualization) and connect to the simulation with URL
    
        wss://<REMOTE_IP>:9002
    


## Notes
- Tested on ubuntu 24.04
- The latest compatible gazebo version is 9.4.0. This is reflected in the way the websocket server is launched through gz-launch, and will require to be updated for `gz-sim>=10`
    
    