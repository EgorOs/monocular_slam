#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
# testing
#import tf
#------
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from camera import ARDroneCamera, DatasetCamera, im_to_undistort_roi
from visualization import CloudStream
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
        self.traj = np.zeros((600,700,3), dtype=np.uint8)
        #  Initializes visualization topic
        self.CS = CloudStream()

    def use_dataset_imgs(self):
        """ For the test and demonstration purposes only, replaces
        camera images and parameters with ones from the dataset.  """
        self.cam = DatasetCamera()
        idx = '00000{}'.format(self.viewset.numViews)
        im = cv2.imread('tests/image_0/{}.png'.format(idx[-6:]))
        return im

    def use_drone_imgs(self):
        """ For the test and demonstration purposes only, replaces
        camera images and parameters with ones from the dataset.  """
        self.cam = ARDroneCamera()
        im = cv2.imread('tests/test_imgs/drone_dataset/test{}.png'.format(self.viewset.numViews))
        return im

    def update(self, im):
        cam = self.cam
        numViews = self.viewset.numViews
        im = self.bridge.imgmsg_to_cv2(im, "bgr8")
        im = self.use_dataset_imgs() #  Replases camera images with images from dataset
        im = im_to_undistort_roi(im, self.cam, to_gray=True)
        if self.viewset.numViews == 0:
            self.viewset.add_view(im)
            ###
            #  Initialize matrix P = (R|t), both rotation and translation are zero
            #  thus R becomes an identity matrix I
            P = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1,0]])
            P = np.dot(cam.K,P)
            self.viewset.projections[numViews] = P
        elif self.viewset.numViews == 1:
            self.viewset.add_view(im)
            kp_new, kp_old, matches = match_ORB_features(self.viewset.views[1],self.viewset.views[0], show_matches=False)
            E, mask = cv2.findEssentialMat(kp_new, kp_old, focal=cam.fx, pp=cam.pp, method=cv2.RANSAC, prob=0.9999, threshold=1.0)
            points, R, t, mask = cv2.recoverPose(E, kp_new, kp_old, focal=cam.fx, pp=cam.pp)
            self.cur_t = t
            self.cur_R = R
            P = np.append(self.cur_R,self.cur_t, axis=1)
            P = np.dot(cam.K,P)
            self.viewset.projections[numViews] = P
        else:
            self.viewset.add_view(im)
            kp_new, kp_old, matches = match_ORB_features(self.viewset.views[numViews],self.viewset.views[numViews-1], show_matches=False)
            E, mask = cv2.findEssentialMat(kp_new, kp_old, focal=cam.fx, pp=cam.pp, method=cv2.RANSAC, prob=0.9999, threshold=1.0)
            points, R, t, mask = cv2.recoverPose(E, kp_new, kp_old, focal=cam.fx, pp=cam.pp)
            self.cur_t = self.cur_t + np.dot(self.cur_R,t)
            self.cur_R = np.dot(R,self.cur_R)
            x,y,z = self.cur_t
            draw_x, draw_y = int(x)+90, int(z)+90
            #  Projection matrix
            P = np.append(self.cur_R,self.cur_t, axis=1)
            P = np.dot(cam.K,P)
            self.viewset.projections[numViews] = P
            #  Triangulation to homogeneous coordinates
            #  Ooops, this fucntion is for stereo cameras!
            #  Or maybe not... Anyway we need to check or replace it.
            x_global = cv2.triangulatePoints(self.viewset.projections[numViews], self.viewset.projections[numViews-1],kp_new,kp_old)
            print(x_global.shape)
            #  Draw 2d trajectory
            cv2.circle(self.traj, (draw_x,draw_y), 1, (100,numViews%255,100), 3)
            cv2.imshow('trajectory', self.traj)
            cv2.waitKey(1)
            #  Send PointCloud2 message
            self.CS.update_cloud(x_global)

def main(): 
    VO = ViewsBuffer()
    rospy.init_node("Visual_odomentry", anonymous=True)
    #  Odometry broadcast for RViz
    #br = tf.TransformBroadcaster()
    try:
        rospy.spin()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()