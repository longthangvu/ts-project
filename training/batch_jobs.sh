#!/bin/bash
while IFS= read -r cmd; do
    echo "Running: $cmd"
    eval "$cmd"
    echo "Finished: $cmd"
done < jobs.txt
