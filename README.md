# Monocular SLAM

Development of monocular SLAM algorithm (work in progress).

## Requirements

* Python 3.5
* OpenCV
* ROS-kinetic

## Installation notes

1) Install ros libraries for python 3.
```
sudo apt-get install python3-catkin-pkg-modules
sudo apt-get install python3-rospkg-modules
```
2) Install openCv and rename cv2.so file in ROS directory for example cv2.so, this step will allow python 3 to import cv2 properly.
```
cd /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so
sudo mv cv2.so cv2_ros.so
```
3) At this point everything should work fine except cv_bridge library, which converts ROS image messages to numpy array, this problem is
related to libboost library, however I solved this problem by placing cv_bridge module in project directory and editing cv_bridge/core.py. All information about edits is written in this file and all rows of code, which caused errors are commented.
