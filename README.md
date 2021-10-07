# Object-classification-in-depth

This repository uses a Realsense camera for object classification depending the distance from the camera. It is possible to recognize an object only if the distance between camera and object is near enough.

It uses OpenCV, numpy, Tensorflow and the Intel RealSense library pyrealsense2.`

```
pip install python-opencv 
pip install numpy 
pip install tensorflow 
pip install pyrealsense2 
```

The code has been made following the instructions from [TensorFlow (intelrealsense.com)](https://dev.intelrealsense.com/docs/tensorflow-with-intel-realsense-cameras).

Download and extract one of the models from https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#use-existing-config-file-for-your-model page.

We use Faster-RCNN Inception v2 for this example. Extracted ´´´frozen_inference_graph.pb´´´ is expected to be in the working directory when running the script.

It has been adapted for depth filtering.
