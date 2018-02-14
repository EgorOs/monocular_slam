#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from time import sleep,time
import sys

class Record:
    """Documentation string"""
    def __init__(self):
        self.img_thread = '/usb_cam/image_raw'
        self.img_thread = '/ardrone/front/image_raw'
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber(self.img_thread, Image, self.on_image_get)
        self.last_time = time()
        self.time_interval = 0.1
        self.ctr = 0

    def start_countdown(self,t):
        print(t)
        for i in range(t)[::-1]:
            sleep(1)
            print(i)

    def on_image_get(self, data):
        camera_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if (time() - self.last_time) > self.time_interval:
        	cv2.imwrite('test_imgs/drone_dataset/test{}.png'.format(self.ctr), camera_img)
        	self.ctr += 1
        	self.last_time = time()
        	print('Frame #{} is taken'.format(self.ctr))

def main():
    rc = Record()
    rc.start_countdown(5)
    rospy.init_node("Camera_calibration", anonymous = True, disable_signals = True)
    try:
        rospy.spin()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()