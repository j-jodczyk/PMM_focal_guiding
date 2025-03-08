#!/bin/bash

# List of scenes
params=("dining-room " "living-room" "modern-hall")

cd ../mitsuba
source setpath.sh
# Loop through each parameter and execute the command
for param in "${params[@]}"; do
    echo "Executing: mitsuba $param"
    mitsuba ../scenes/$param/$param.xml
    echo "Done with: $param"
    echo "-----------------------------"
done

cd ../scripts
