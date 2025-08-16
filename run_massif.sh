#!/bin/bash

# List of Docker image tags (customize this list)
DOCKER_TAGS=("pmm_focal_guiding")

# Output directory for massif logs
OUTPUT_DIR="./massif_outputs"
mkdir -p "$OUTPUT_DIR"

for TAG in "${DOCKER_TAGS[@]}"; do
    echo "=== Running Valgrind Massif on container: $TAG ==="

    # Generate a unique container name
    CONTAINER_NAME="massif_$(echo $TAG | tr ':/' '__')"

    # Run container in detached mode with Valgrind Massif
    docker run --name "$CONTAINER_NAME" -d "$TAG" tail -f /dev/null

    # Run the target command with Valgrind Massif
    docker exec "$CONTAINER_NAME" \
        valgrind --tool=massif --massif-out-file=/tmp/massif.out \
        mitsuba ./scene/living-room/living-room.xml

    # Copy the massif output from the container to host
    docker cp "$CONTAINER_NAME":/tmp/massif.out "$OUTPUT_DIR/massif_${CONTAINER_NAME}_lv.out"

    # Stop and remove the container
    docker stop "$CONTAINER_NAME" > /dev/null
    docker rm "$CONTAINER_NAME" > /dev/null

    echo "Output saved to $OUTPUT_DIR/massif_${CONTAINER_NAME}_lv.out"
    echo
done

echo "All profiling runs complete. Dining room."

for TAG in "${DOCKER_TAGS[@]}"; do
    echo "=== Running Valgrind Massif on container: $TAG ==="

    # Generate a unique container name
    CONTAINER_NAME="massif_$(echo $TAG | tr ':/' '__')"

    # Run container in detached mode with Valgrind Massif
    docker run --name "$CONTAINER_NAME" -d "$TAG" tail -f /dev/null

    # Run the target command with Valgrind Massif
    docker exec "$CONTAINER_NAME" \
        valgrind --tool=massif --massif-out-file=/tmp/massif.out \
        mitsuba ./scene/modern-hall/modern-hall.xml

    # Copy the massif output from the container to host
    docker cp "$CONTAINER_NAME":/tmp/massif.out "$OUTPUT_DIR/massif_${CONTAINER_NAME}_mh.out"

    # Stop and remove the container
    docker stop "$CONTAINER_NAME" > /dev/null
    docker rm "$CONTAINER_NAME" > /dev/null

    echo "Output saved to $OUTPUT_DIR/massif_${CONTAINER_NAME}_mh.out"
    echo
done
