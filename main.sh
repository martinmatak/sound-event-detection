#!/bin/bash

mkdir tmp
cp $1 tmp/video.mp4
cp ffmpeg-split.py tmp/ffmpeg-split.py
cd tmp
echo "Preparing a video for processing.."
python ffmpeg-split.py -f video.mp4 -s 3 &> /dev/null
echo "Video prepared."
rm video.mp4
cd ..
echo "Searching for the sound event.."
python evaluate.py
rm -rf tmp
