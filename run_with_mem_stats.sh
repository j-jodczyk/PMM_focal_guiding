#!/bin/bash

# -------------------- Configuration --------------------

HOST_VOLUME="/home/julka/PMM_focal_guiding"
CONTAINER_VOLUME="/var/lib/focal_vol"
DOCKER_IMAGE_NAME="pmm_focal_guiding"
SCENE_FILE="$1"  # relative to the container mount, e.g., scene/foo.xml
SAVE_DIR="$2"

echo $SCENE_FILE
SCENE_NAME=$(echo "${SCENE_FILE##*/}" | cut -d. -f1)

LOG_DIR="/home/julka/PMM_focal_guiding/$SAVE_DIR/$SCENE_NAME"

if [ -z "$SCENE_FILE" ]; then
  echo "Usage: $0 <scene_file_relative_to_volume>"
  exit 1
fi

echo "🚀 Starting container..."
CID=$(docker run -d \
  -v "${HOST_VOLUME}:${CONTAINER_VOLUME}" \
  -p 127.0.0.1:3002:3002 \
  -e SCENE_FILE="$SCENE_FILE" \
  "$DOCKER_IMAGE_NAME")
SHORT_CID=${CID:0:12}

if [ -z "$CID" ]; then
  echo "❌ Failed to start container"
  exit 1
fi

echo "📦 Container started with ID: $SHORT_CID"

# -------------------- Step 3: Monitor memory --------------------

# MEM_LOG_FILE="${LOG_DIR}/memusage.${SHORT_CID}.log"
RENDER_LOG_FILE="${LOG_DIR}/mitsuba.${SHORT_CID}.log"

# mkdir -p "$LOG_DIR"

# echo "📊 Logging memory usage to $MEM_LOG_FILE"

# # Start logging memory usage in the background
# docker stats $CID --no-stream=false --format "{{.MemUsage}}" | sed 's/\x1b\[[0-9;]*m//g' > "$MEM_LOG_FILE" &
# STATS_PID=$!

# -------------------- Step 4: Wait for container to finish --------------------

echo "⏳ Waiting for container to finish rendering..."
docker wait "$CID" > /dev/null

# Stop memory logging after the container finishes
kill "$STATS_PID"

# -------------------- Step 5: Copy Mitsuba render log --------------------

echo "📜 Copying Mitsuba log from container..."
docker cp "${SHORT_CID}:/mitsuba/mitsuba.${SHORT_CID}.log" "$RENDER_LOG_FILE" 2>/dev/null

if [ $? -ne 0 ]; then
  echo "⚠️ Could not copy render log. Check if it exists inside the container at /mitsuba/mitsuba.${SHORT_CID}.log"
else
  echo "✅ Mitsuba log saved to $RENDER_LOG_FILE"
fi

# -------------------- Step 6: Copy rendered result --------------------

echo "📥 Copying rendered file from container..."

OUTPUT_SCENE_FILE="${SCENE_FILE%.xml}.exr"
# Copy output from container
docker cp $SHORT_CID:/mitsuba/$OUTPUT_SCENE_FILE $LOG_DIR/$(basename "$OUTPUT_SCENE_FILE")

if [ $? -ne 0 ]; then
  echo "⚠️ Could not copy rendered output from container. Check path."
else
  echo "✅ Rendered output saved to $SCENE_FILE"
fi

# -------------------- Step 7: Memory stats summary --------------------

# echo "📈 Peak memory usage:"
# awk '{gsub(/MiB.*/, "", $1); if ($1+0 > max) max=$1+0} END {print max " MiB"}' "$MEM_LOG_FILE"

# -------------------- Done --------------------

echo "✅ Render complete."
echo "🗂 Logs: $RENDER_LOG_FILE"
# echo "🗂 Memory: $MEM_LOG_FILE"
