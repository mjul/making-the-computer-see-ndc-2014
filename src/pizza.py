#!/usr/bin/env python

'''
Example code for detecting a pizza box in an image.
'''

# OpenCV imports
import numpy as np
import cv2

# Histogram and plotting
import matplotlib.pyplot as plt


# ----------------------------------------------------------------

def draw_keypoints(img, keypoints, colour = (0, 255, 255)):
    '''Return a new image with the keypoints.'''
    result = img.copy()
    for kp in keypoints:
            x, y = kp.pt
            r = kp.size
            cv2.circle(result, (int(x), int(y)), int(r), colour, thickness=10)
    return result

# ----------------------------------------------------------------

# From OpenCV sample, samples/python2/find_obj.py

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

# ----------------------------------------------------------------

def show(title, img):
    small_shape = (480,640) if (img.shape[0] > img.shape[1]) else (640,480)
    small = cv2.resize(img, small_shape)
    cv2.imshow(title, small)
    
# ----------------------------------------------------------------

def read_scene(fname, image_type):
    '''Read a door scene image and crop it to the interior of the door frame.'''
    whole_scene = cv2.imread(fname, image_type)
    upright = cv2.flip(cv2.transpose(whole_scene), 0)
    x,y,w,h = 610,900,3*440,4*440
    cropped = upright[y: y + h, x: x + w]
    return cropped

# ----------------------------------------------------------------

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)

    return vis

# ----------------------------------------------------------------

# For a detailed description of how this works, refer to:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#feature-homography


if  __name__ =='__main__':
    image_type = cv2.CV_LOAD_IMAGE_GRAYSCALE
    pizza_box = cv2.imread('../images/pizza/pizza_box.jpg', image_type)
    chocolate_box = cv2.imread('../images/pizza/chocolate_box.jpg', image_type)
    pizza_delivery = read_scene('../images/pizza/pizza_delivery_3.jpg', image_type)
    chocolate_delivery = read_scene('../images/pizza/chocolate_delivery_1.jpg', image_type)

    # Detection and matching algorithms
    #detector, norm = cv2.SIFT(), cv2.NORM_L2
    detector, norm = cv2.ORB(4000), cv2.NORM_HAMMING
    
    # Brute-force matcher
    #matcher = cv2.BFMatcher(norm)
    if norm == cv2.NORM_L2:
        FLANN_INDEX_KDTREE = 1
        flann_matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = 5), dict(checks = 50))
    else:
        FLANN_INDEX_LSH = 6
        flann_matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1), {})
         
    matcher = flann_matcher


    object_image = chocolate_box
    scene_image = chocolate_delivery
    
    # Detect features in the reference object
    print "Detecting features..."
    object_keypoints, object_descriptors = detector.detectAndCompute(object_image, None)
    print 'Object has %d features' % len(object_keypoints)
    scene_keypoints, scene_descriptors = detector.detectAndCompute(scene_image, None)
    print 'Scene has %d features' % len(scene_keypoints)

    obj_kp_image = draw_keypoints(cv2.cvtColor(object_image, cv2.COLOR_GRAY2BGR), object_keypoints)
    scene_kp_image = draw_keypoints(cv2.cvtColor(scene_image, cv2.COLOR_GRAY2BGR), scene_keypoints)
    
    show('Scene', scene_image)
    show('Object Keypoints', obj_kp_image)
    show('Scene Keypoints', scene_kp_image)
    
    # match object and scene
    print "Matching..."
    raw_matches = matcher.knnMatch(object_descriptors, scene_descriptors, k=2)
    #p1, p2, kp_pairs = filter_matches(object_descriptors, scene_descriptors, raw_matches)
    good = []
    # store all the good matches as per Lowe's ratio test.
    for m,n in raw_matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print "GOOD points: %d" % len(good)

    MIN_MATCH_COUNT = 10

    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ object_keypoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ scene_keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = scene_image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # Draw the matches
        match_image = cv2.cvtColor(scene_image, cv2.COLOR_GRAY2BGR)
        print dst
        cv2.polylines(match_image,[np.int32(dst)], isClosed = True,
                      color=(255,255,0), thickness=10, lineType = cv2.CV_AA)
        cv2.circle(match_image, (100, 100), 100, (255,0,0), thickness=10)

        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #                     singlePointColor = None,
        #                     matchesMask = matchesMask, # draw only inliers
        #                     flags = 2)

        # match_lines_image = cv2.drawMatches(object_image, object_keypoints,
        #                                     scene_image, scene_keypoints, good, None, **draw_params)

        match_lines_image = explore_match("Title", object_image, scene_image, object_keypoints, scene_keypoints, good)

        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

        show('Matches', match_image)
        show('Match lines', match_lines_image)
        
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
        
    #     H, status = c
    #     print '%d / %d  inliers/matched' % (np.sum(status), len(status))
    # else:
    #     H, status = None, None
    #     print '%d matches found, not enough for homography estimation' % len(p1)
    
    
        
    print "Done."
