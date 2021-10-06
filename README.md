# Object-classification-in-depth

This repository uses a Realsense camera for object classification depending the distance from the camera. It is possible to recognize an object only if the distance between camera and object is near enough.

It uses OpenCV, numpy, Tensorflow and the Intel RealSense library pyrealsense2.`

```
pip install python-opencv 
pip install numpy 
pip install tensorflow 
pip install pyrealsense2 
```

The code has been made following the instructions from [TensorFlow (intelrealsense.com)](https://dev.intelrealsense.com/docs/tensorflow-with-intel-realsense-cameras)

It has been adapted for depth filtering.
