#!/usr/bin/env python
import cv2
import numpy as np
from itertools import compress
import time
t = time.time()

''' Additional information can be found at 
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html'''

def check_pairwise(img1,img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    dist = np.array([[-0.40538686, 0.18274696, 0.00449549, -0.00054929, 0.06070349]])
    newcameramtx=np.array([[ 394.83410645,    0.,          304.59488242],
                        [   0.,          394.50994873,  178.66114884],
                        [   0.,            0.,            1.        ]])

    mtx = np.array([[ 482.05945726,    0.,          305.34544298],
                    [   0.,          479.77705725,  176.55010834],
                    [   0.,            0.,            1.        ]])
    gray1 = cv2.undistort(gray1, mtx, dist, None, newcameramtx)
    gray2 = cv2.undistort(gray2, mtx, dist, None, newcameramtx)
    x,y,w,h = 20,25,590,310
    gray1 = gray1[y:y+h, x:x+w]
    gray2 = gray2[y:y+h, x:x+w]

    surf = cv2.xfeatures2d.SURF_create()
    MIN_MATCH_COUNT = 10

    kp1, des1 = surf.detectAndCompute(gray1,None)
    kp2, des2 = surf.detectAndCompute(gray2,None)

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
        src_pts = list(compress(src_pts,matchesMask))
        dst_pts = list(compress(dst_pts,matchesMask))

        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        F,mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_8POINT)
        
        # Convert to homogenous coordinates (like transposed [x,y,1])
        pt1 = np.array([[dst_pts[0][0][0]], [dst_pts[0][0][1]], [1]])
        pt0 = np.array([[src_pts[0][0][0], src_pts[0][0][1], 1]])
        print("Fundamental matrix error check: %f"%np.dot(np.dot(pt0,F),pt1))

        E, mask = cv2.findEssentialMat(src_pts,dst_pts, focal=1.0, pp=(0.,0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
        points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)
        print(R)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = None, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(gray1,kp1,gray2,kp2,good_matches,None,**draw_params)

    return img3

for i in range(362):
    print('Testing {}-{}.png'.format(i,i+1))
    img1 = cv2.imread('test_imgs/drone_dataset/test{}.png'.format(i), 1)
    img2 = cv2.imread('test_imgs/drone_dataset/test{}.png'.format(i+1), 1)
    img3 = check_pairwise(img1, img2)
    cv2.imwrite('test_imgs/test_out/test{}_{}.png'.format(i,i+1), img3)