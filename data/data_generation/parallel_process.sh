#!/bin/bash

# Set base directory and number of parallel processes (N)
BASE_DIR="/scratch/ssd1/audio_datasets/SpatialSounds/mp3d_reverb/binaural"
N=18

# Get the list of subdirectories and split into N segments
SUBDIRS=($(ls -d $BASE_DIR/*/))
TOTAL_SUBDIRS=${#SUBDIRS[@]}
SPLIT_SIZE=$((TOTAL_SUBDIRS / N))
REMAINDER=$((TOTAL_SUBDIRS % N))

# Create an array of jobs (subdirectory lists) to be processed by synth.py
for ((i=0; i<N; i++)); do
    START=$((i * SPLIT_SIZE))
    END=$((START + SPLIT_SIZE))

    # Distribute the remaining subdirectories
    if [[ $i -eq $((N-1)) ]]; then
        END=$((END + REMAINDER))
    fi

    # Create a temp file with the subdirectories for this segment
    TEMP_FILE="sublist_$i.txt"
    echo "${SUBDIRS[@]:$START:$((END - START))}" > "$TEMP_FILE"
    
    # Run the synth.py script in the background
    python3 synth.py "$TEMP_FILE" &
done

# Wait for all background jobs to complete
wait

echo "All processes completed!"

