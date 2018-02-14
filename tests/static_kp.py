#!/usr/bin/env python
import cv2
import numpy as np
from kp_matcher import *
from itertools import compress
import time
t = time.time()

''' Additional information can be found at 
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html'''

img1 = cv2.imread('test_imgs/1.png', 1)
img2 = cv2.imread('test_imgs/2.png', 1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


surf = cv2.xfeatures2d.SURF_create()
MIN_MATCH_COUNT = 10

kp1, des1 = surf.detectAndCompute(gray1,None)
kp2, des2 = surf.detectAndCompute(gray2,None)
locs1 = [p.pt for p in kp1]
locs2 = [p.pt for p in kp2]

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)


flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:  #  if not enough matches drone should stop
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    good_matches = list(compress(good,matchesMask))

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    F,mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_8POINT)
    
    # Convert to homogenous coordinates (like transposed [x,y,1])
    pt1 = np.array([[dst_pts[0][0][0]], [dst_pts[0][0][1]], [1]])
    pt0 = np.array([[src_pts[0][0][0], src_pts[0][0][1], 1]])
    print("Fundamental matrix error check: %f"%np.dot(np.dot(pt0,F),pt1))
    print('Execution time: {}'.format(time.time()-t))
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = None, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)

cv2.imwrite('matching_test.png', img3)