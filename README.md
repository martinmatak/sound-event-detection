# Sound Event Detection

Simple app for sound event detection. 

### Installation

This app requires Python 3 and [FFmpeg](https://www.ffmpeg.org/) to run.

Install the requirements and start `main.sh`.

```sh
$ cd soundEventDetection
$ pip install -r requirements.txt
$ ./main.sh path-to-mp4video
```

### Training

To train a model, start `train.py`. It expects a directory `dataset` in the same directory as `train.py`. Every video in that directory should be around 3 sec long. If in the video is a particular sound which should be detected - *name_1.mp4* , otherwise *name_0.mp4*. 

### Todos

 - Experiment with neural network architecture (reduce number of false positives)
 - Check the case when there are multiple sound event detections one after another

License
----

MIT


**Free Software, Hell Yeah!**
