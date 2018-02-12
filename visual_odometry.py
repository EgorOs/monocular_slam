#!/usr/bin/env python
import cv2
import numpy as np
from itertools import compress

INIT_FRAME = 0
SECOND_FRAME = 1
NEXT_FRAME = 2

def match_features(kp1, des1, kp2, des2, cam, min_match_count=10):

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    if len(good) > min_match_count:  #  if not enough matches drone should stop
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        good_matches = list(compress(good,matchesMask))
        # NANI?! len(src_pts)!=len(matches)
        src_pts = list(compress(src_pts,matchesMask))
        dst_pts = list(compress(dst_pts,matchesMask))
        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        return src_pts, dst_pts
    else:
        print("Not enough matches are found - {}/{}".format(len(good),min_match_count))
        return None

def undistort_frame(frame, cam):
    frame = cv2.undistort(frame, cam.K, cam.dist, None, cam.new_K)
    x,y,w,h = 20,25,590,310
    frame = frame[y:y+h, x:x+w]
    return frame

def estimate_pose(src_pts, dst_pts, cam):
    F,mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_8POINT)
    E, mask = cv2.findEssentialMat(src_pts,dst_pts, focal=cam.focal, pp=cam.pp, method=cv2.RANSAC, prob=0.999, threshold=3.0)
    points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, focal=cam.focal, pp=cam.pp)
    return R,t

class Camera:
    """ Camera model parameters """
    def __init__(self):
        #  K = [fx,0,cx,0,fy,cy,0,0,1]
        self.K = np.array([[ 394.83410645,    0.,          304.59488242],
                           [   0.,          394.50994873,  178.66114884],
                           [   0.,            0.,            1.        ]])
        self.focal = self.K[0][0]
        self.pp = (self.K[0][2], self.K[1][2])

        #  dist = [k1,k2,p1,p2,k3]
        self.dist = np.array([[-0.40538686, 0.18274696, 0.00449549, -0.00054929, 0.06070349]])
        self.new_K = np.array([[ 482.05945726,    0.,          305.34544298],
                                [   0.,          479.77705725,  176.55010834],
                                [   0.,            0.,            1.        ]])

class VisualOdometry:
    def __init__(self):
        self.frame_status = 0
        self.cur_frame = None
        self.old_frame = None
        self.cur_R = []
        self.cur_t = None
        self.cam = Camera()
        self.detector = cv2.xfeatures2d.SURF_create()
        self.old_kp = None
        self.old_des = None
        # self.cur_kp = None
        # self.cur_des = None

    def process_init_frame(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = undistort_frame(gray, self.cam)
        kp,des = self.detector.detectAndCompute(gray,None)
        self.frame_status = SECOND_FRAME
        return kp,des

    def process_next_frame(self,frame):
        cam = self.cam
        old_kp = self.old_kp
        old_des = self.old_des
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = undistort_frame(gray, self.cam)
        cur_kp,cur_des = self.detector.detectAndCompute(gray,None)
        src_pts,dst_pts = match_features(old_kp, old_des, cur_kp, cur_des, cam)
        self.old_kp, self.old_des = cur_kp, cur_des
        if self.frame_status == SECOND_FRAME:
            self.cur_R, self.cur_t = estimate_pose(src_pts,dst_pts, self.cam)
            self.frame_status = NEXT_FRAME
        else:
            R, t = estimate_pose(src_pts,dst_pts, self.cam)
            self.cur_t = self.cur_t + np.dot(self.cur_R,t)
            self.cur_R = np.dot(R,self.cur_R)


    def update(self, frame):
        if self.frame_status == INIT_FRAME:
            self.old_kp, self.old_des = self.process_init_frame(frame)
        else:
            self.process_next_frame(frame)

VO = VisualOdometry()
ox,oy = 0, 0
map_2d = np.zeros((600,600,3), dtype=np.float32)
for i in range(362):
    img = cv2.imread('test_imgs/drone_dataset/test{}.png'.format(i), 1)
    VO.update(img)
    if VO.frame_status > SECOND_FRAME:
        print(VO.cur_t)
        s=2
        nx, ny = (VO.cur_t[0]*s, VO.cur_t[1]*s)
        shift = 290
        cv2.line(map_2d,(ox+shift,oy+shift),(nx+shift,ny+shift),(i,255,255 - i),2)
        ox = nx
        oy = ny
        cv2.imshow('2D Map', map_2d)
        cv2.imshow('Frame', img)
        cv2.waitKey(10)


