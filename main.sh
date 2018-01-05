#!/bin/bash

mkdir tmp
cp $1 tmp/video.mp4
cp ffmpeg-split.py tmp/ffmpeg-split.py
cd tmp
python ffmpeg-split.py -f video.mp4 -s 3
rm video.mp4
cd ..
python evaluate.py
rm -rf tmp
