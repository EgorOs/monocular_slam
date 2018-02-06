#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from time import sleep
from time import time
import sys

class CameraCalibration:
    """Documentation string"""
    def __init__(self):
        self.img_thread = '/usb_cam/image_raw'
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber(self.img_thread, Image, self.on_image_get)
        self.grid_images = []
        self.gray_images = []
        self.grid_w = 6
        self.grid_h = 9
        self.n_frames = 20
        self.time_since_last_frame = 0
        self.end_frame_time = 0
        #  Time interval allows to reposition chess grid between frames
        self.time_interval = 1.5
        self.calibration_complete = False
        self.initial_time = time()
    
    def start_countdown(self,t):
        print(t)
        for i in range(t)[::-1]:
            sleep(1)
            print(i)

    def on_image_get(self, data):
        self.camera_img = self.bridge.imgmsg_to_cv2(data, "bgr8")

        #  Calibration tutorials:
        #  https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        #  http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        
        if len(self.grid_images) < self.n_frames:
            self.find_corners(self.camera_img)
        elif self.calibration_complete == False:
            params = self.get_calibration_parameters()
        else:
            sys.exit()

    def find_corners(self, img):
        #  Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        #  Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.grid_w*self.grid_h,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.grid_h,0:self.grid_w].T.reshape(-1,2)

        #  Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #  Looking for chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (self.grid_h,self.grid_w),None)

        #  If corners are found append image data to the list
        self.time_since_last_frame += time()-self.end_frame_time
        if (ret == True):
            self.objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            self.imgpoints.append(corners2)
            
            #  Draw and display the corners
            grid_img = cv2.drawChessboardCorners(img, (self.grid_h,self.grid_w), corners2,ret)
            cv2.imshow('Camera image', grid_img)
            cv2.waitKey(1)
            if self.time_interval < self.time_since_last_frame:
                self.grid_images.append(grid_img)
                self.gray_images.append(gray)
                print('Grid found: {}/{}'.format(len(self.grid_images),self.n_frames))
                self.time_since_last_frame = 0
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(gray,'WASTED',(220,300), font, 2,(2,20,20),4,cv2.LINE_AA)
            cv2.imshow('Camera image', gray)
            cv2.waitKey(1)

        self.end_frame_time = time()
        
    def get_calibration_parameters(self):
        cv2.destroyAllWindows()
        for gray in self.gray_images:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints,self.imgpoints, gray.shape[::-1],None,None)
        h,  w = self.gray_images[0].shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        f = open('camera_matirx.txt', 'w')
        f.write(str(newcameramtx))
        f.close()
        self.calibration_complete = True
        dst = cv2.undistort(self.gray_images[0], mtx, dist, None, newcameramtx)
        cv2.imwrite('undistort/dst.png', dst)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('undistort/roi.png', dst)
        return (mtx, dist, None, newcameramtx)


def main():
    calibration = CameraCalibration()
    calibration.start_countdown(3)
    rospy.init_node("Camera_calibration", anonymous = True)
    try:
        rospy.spin()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()