#!/usr/bin/env python
import numpy as np
import cv2

class ARDroneCamera:
    """ An object, that stores all camera parameters. """
    def __init__(self):
        self.fx = 394.83410645
        self.fy = 304.59488242
        self.cx = 394.50994873
        self.cy = 178.66114884
        self.pp = (self.cx, self.cy)
        self.K = np.array([
            [self.fx,   0,          self.cx],
            [0,         self.fy,    self.cy],
            [0,         0,          1]])
        self.dist = np.array([[-0.40538686, 0.18274696, 0.00449549, -0.00054929, 0.06070349]])
        self.roi = (20,25,590,310) #  Needs re-identification

class DatasetCamera:
    """ An object, that stores all camera parameters. """
    def __init__(self):
        self.fx = 718.8560
        self.fy = 718.8560
        self.cx = 607.1928
        self.cy = 185.2157
        self.pp = (self.cx, self.cy)
        self.K = np.array([
            [self.fx,   0,          self.cx],
            [0,         self.fy,    self.cy],
            [0,         0,          1]])
        self.dist = np.array([[0, 0, 0, 0, 0]])
        self.roi = (0,0,1241,376) #  Needs re-identification

def im_to_undistort_roi(im, cam, to_gray=False):
    undist_im = cv2.undistort(im, cameraMatrix=cam.K, distCoeffs=cam.dist)
    x,y,w,h = cam.roi
    cropped_im = undist_im[y:y+h, x:x+w]
    if to_gray:
        cropped_im = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2GRAY)
    return cropped_im

def show_trajectory(R,t):

    pass