#!/usr/bin/env python

#  import cv2
import numpy as np
import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
#  from sensor_msgs.msg import Image
#  from cv_bridge import CvBridge, CvBridgeError


# >>> Separate nodes for local and global point clouds required,
# >>> the local one slould be marged with the global later.
class CloudStream:
	''' Generates PointCloud messages from triangulated points
	    and publishes them. '''
	def __init__(self):
		self.cloud_pub = rospy.Publisher('CloudStream', PointCloud2, queue_size = 20)

	def update_cloud(self, pts3d):
		self.h = std_msgs.msg.Header()
		self.h.stamp = rospy.Time.now()
		self.h.frame_id = 'world_coord'
		#  Slows down the overall process unless amount of keypoints is limited,
		#  However it there might be a better solution with the help of numpy
		#  Maybe something like numpy-to-list

		# Swap Y,Z axis, remove 4th row for now
		#    print(pts3d[0][2],pts3d[1][2],pts3d[2][2])
		pts_list = pts3d[:3].T
		pts_list.tolist()
		#    print(pts_list[2])
		point_cloud = pcl2.create_cloud_xyz32(self.h, pts_list)
		self.cloud_pub.publish(point_cloud)

		pass


