#!/bin/bash

# Variables
AUDIO_FILE="hot-sugar.webm"
IMAGE_PATTERN="images-high-def/img*.png"
OUTPUT_VIDEO="final_video.mp4"

# 1. Get the duration of the audio file in seconds
AUDIO_DURATION=$(ffmpeg -i $AUDIO_FILE 2>&1 | grep "Duration" | cut -d ' ' -f 4 | sed s/,// | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')

# 2. Calculate the total number of frames required
TOTAL_FRAMES=$(echo "$AUDIO_DURATION * 30" | bc)

# 3. Count the number of images in the directory
NUM_IMAGES=$(ls $IMAGE_PATTERN | wc -l)

# 4. Calculate the number of loops required
LOOPS=$(echo "$TOTAL_FRAMES / $NUM_IMAGES" | bc)

# 5. Create the looping video
ffmpeg -framerate 30 -pattern_type glob -i "$IMAGE_PATTERN" -vf "loop=loop=$LOOPS:size=$NUM_IMAGES" -c:v libx264 -pix_fmt yuv420p -shortest looping_video.mp4

# 6. Combine the audio and video
ffmpeg -i looping_video.mp4 -i $AUDIO_FILE -c:v copy -c:a aac -strict experimental -shortest $OUTPUT_VIDEO

# 7. Clean up the temporary looping video
rm looping_video.mp4

echo "Done! Your video is saved as $OUTPUT_VIDEO"
