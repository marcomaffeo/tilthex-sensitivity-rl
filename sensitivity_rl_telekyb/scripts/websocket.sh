#!/bin/bash
# Expand the environment variables in websocket.gzlaunch and launch it

TEMP_LAUNCH=$(mktemp --suffix=.gzlaunch)
envsubst < "$PIXI_PROJECT_ROOT/worlds/websocket.gzlaunch" > "$TEMP_LAUNCH"
gz launch -v 4 "$TEMP_LAUNCH"
rm "$TEMP_LAUNCH"
