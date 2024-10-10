#!/bin/bash

# GPU index to target
gpu_index=3

# Get the list of Python processes running on the specified GPU
processes=$(nvidia-smi pmon -c 1 | grep python | awk -v gpu_index="$gpu_index" '$2 == gpu_index {print $3}')

# Loop through each process and kill it
for pid in $processes
do
    echo "Killing Python process with PID: $pid"
    kill $pid
done
