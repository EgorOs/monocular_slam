#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from camera import ARDroneCamera, DatasetCamera, im_to_undistort_roi
from features import *
from views_data import *

class ViewsBuffer:
    """ Processes a chunk of trajectory (5 frames), that is going to be added to the global map """
    def __init__(self):
        self.viewset = ViewSet()
        self.cam = ARDroneCamera()
        self.img_thread = '/usb_cam/image_raw'
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber(self.img_thread, Image, self.update)
        self.cur_t = None
        self.cur_R = None
        self.traj = np.zeros((600,600,3), dtype=np.uint8)

    def use_dataset_imgs(self):
        """ Returns camera images with dataset images and parameters """
        self.cam = DatasetCamera()
        idx = '00000{}'.format(self.viewset.numViews)
        im = cv2.imread('tests/image_0/{}.png'.format(idx[-6:]))
        return im

    def use_drone_imgs(self):
        """ Returns camera images with dataset images and parameters """
        self.cam = DatasetCamera()
        im = cv2.imread('tests/test_imgs/drone_dataset/test{}.png'.format(self.viewset.numViews))
        return im

    def update(self, im):
        cam = self.cam
        viewset = self.viewset
        views = self.viewset.views
        numViews = self.viewset.numViews
        im = self.bridge.imgmsg_to_cv2(im, "bgr8")
        im = self.use_dataset_imgs() #  Replases camera images with images from dataset
        im = im_to_undistort_roi(im, self.cam, to_gray=True)
        if self.viewset.numViews == 0:
            self.viewset.add_view(im)
        elif self.viewset.numViews == 1:
            self.viewset.add_view(im)
            kp_new, kp_old, matches = match_ORB_features(self.viewset.views[1],self.viewset.views[0], show_matches=True)
            E, mask = cv2.findEssentialMat(kp_new, kp_old, focal=cam.fx, pp=cam.pp, method=cv2.RANSAC, prob=0.9999, threshold=1.0)
            points, R, t, mask = cv2.recoverPose(E, kp_new, kp_old, focal=cam.fx, pp=cam.pp)
            self.cur_t = t
            self.cur_R = R
        else:
            self.viewset.add_view(im)
            kp_new, kp_old, matches = match_ORB_features(self.viewset.views[numViews],self.viewset.views[numViews-1], show_matches=True)
            E, mask = cv2.findEssentialMat(kp_new, kp_old, focal=cam.fx, pp=cam.pp, method=cv2.RANSAC, prob=0.9999, threshold=1.0)
            points, R, t, mask = cv2.recoverPose(E, kp_new, kp_old, focal=cam.fx, pp=cam.pp)
            self.cur_t = self.cur_t + np.dot(self.cur_R,t)
            self.cur_R = np.dot(R,self.cur_R)
            x,y,z = self.cur_t
            draw_x, draw_y = int(x)+290, int(z)+90

            cv2.circle(self.traj, (draw_x,draw_y), 1, (10,255,0), 1)
            cv2.imshow('trajectory', self.traj)
            cv2.waitKey(1)


def main(): 
    VO = ViewsBuffer()
    rospy.init_node("Visual_odomentry", anonymous=True)
    try:
        rospy.spin()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()