#!/usr/bin/env python

#  import cv2
import numpy as np
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
#  from sensor_msgs.msg import Image
#  from cv_bridge import CvBridge, CvBridgeError

class CloudStream:
	def __init__(self):
		self.cloud_pub = rospy.Publisher('CloudStream', PointCloud, queue_size = 100)
		self.point_cloud = PointCloud()

	def update_cloud(self, pts3d):
		self.h = std_msgs.msg.Header()
		self.h.stamp = rospy.Time.now()
		self.h.frame_id = 'odom'
		#  Simplify it later
		self.point_cloud.header = self.h
		point = Point32()
		for i in range(len(pts3d[0])):
			point.x = pts3d[0][i]
			point.z = pts3d[1][i]
			point.y = pts3d[2][i]
			#  f = pts3d[3][i]???
			#  print('X: {}, Y: {}, Z: {}'.format(x,y,z))
			self.point_cloud.points.append(point)
		self.cloud_pub.publish(self.point_cloud)

		pass


