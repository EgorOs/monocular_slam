#!/usr/bin/env python
import cv2
import numpy as np

orb = cv2.ORB_create(900)

def match_ORB_features(im1, im2, show_matches=False, window_idx=1):
    """ Matches features and removes a number of wrong correspondances,
    allows to display matching results, if multiple instances of this function
    show matches, different window indices should be set in order to avoid flickering. """
    
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    # Taking only relieble matches (better filtration is required)
    matches = [m for m in matches if m.distance < 60]
    # Sort list in a way that the most precise matches were first
    matches = sorted(matches, key=lambda x:x.distance)

    if show_matches:
        img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=2)
        cv2.imshow('Matches {}'.format(window_idx), img3)
        cv2.waitKey(1)


    kp1 = np.float32([kp1[x.queryIdx].pt for x in matches]).reshape(-1,1,2)
    kp2 = np.float32([kp2[x.trainIdx].pt for x in matches]).reshape(-1,1,2)
        #    kp_new = np.float32([kp_new[x.queryIdx].pt for x in matches]).reshape(-1,1,2)
        #    kp_old = np.float32([kp_old[x.trainIdx].pt for x in matches]).reshape(-1,1,2)

    return kp1,kp2,matches

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    im1 = cv2.imread('tests/test_imgs/drone_dataset/test0.png')
    im2 = cv2.imread('tests/test_imgs/drone_dataset/test1.png')
    kp1, kp2, m = match_ORB_features(im1,im2)

    img3 = cv2.drawMatches(im1,kp1,im2,kp2,m,None, flags=2)

    plt.imshow(img3),plt.show()



