#!/usr/bin/env python

'''
Example code for detecting a pizza box in an image.
'''

# OpenCV imports
import numpy as np
import cv2

# Histogram and plotting
import matplotlib.pyplot as plt

import re

# ----------------------------------------------------------------

SAVE_SCREENSHOTS = False

# ----------------------------------------------------------------

def screenshot_filename(title):
    return "../screenshots/pizza %s.jpg" % title.replace(':', '')

def save_image(title, img):    
    fname = screenshot_filename(title)
    print "Writing %s..." % fname
    cv2.imwrite(fname, img)

def show(title, img):
    '''Display the image.
       As a side-effect, save it if SAVE_SCREENSHOTS is True.'''
    h,w = img.shape[:2]
    if (h > 480) or (w >1024):
        wh_ratio = 1.0 * w/h
        if (h < w):
            small_shape = (640, 640*h/w)
        else:
            small_shape = (480, 480*h/w)
        small = cv2.resize(img, small_shape)
    else:
        small = img
    cv2.imshow(title, small)
    if SAVE_SCREENSHOTS:
        save_image(title, img)

# ----------------------------------------------------------------

def draw_keypoints(img, keypoints, colour = (0, 255, 255)):
    '''Return a new image with the keypoints.'''
    result = img.copy()
    for kp in keypoints:
            x, y = kp.pt
            r = kp.size
            cv2.circle(result, (int(x), int(y)), int(r), colour, thickness=2)
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


def draw_matches(img1, img2, kp_pairs, status = None, H = None):
    assert img1.ndim == 2, "Expected single-channel img1."
    assert img2.ndim == 2, "Expected single-channel img2."
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # If we found a homography from the object to the
    # scene, highlight the object in the scene
    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        # We consider only convex quads good matches
        is_convex = cv2.isContourConvex(corners)
        if is_convex:
            cv2.polylines(vis, [corners], True, (255, 0, 255), thickness=30)

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
        
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)

    # Highlight keypoints in object and scene
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, thickness=10)
            cv2.circle(vis, (x2, y2), 2, col, thickness=50)
        else:
            col = red
            r = 8
            thickness = 20
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

    # Connect points in object and scene
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green, thickness=20)

    return vis

# ----------------------------------------------------------------

def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        return img.copy()

# ----------------------------------------------------------------

def foreground_mask(scene_image, empty_scene):
    '''Create a mask for the foreground pixels in the scene.
       Works by diffing the scene and the empty scene and
       building a mask from that.'''
    # Find the foreground
    diff = cv2.absdiff(scene_image, empty_scene)
    ret_val, fg_raw = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)
    # remove small noise...
    fg_denoise = cv2.erode(fg_raw, kernel=None, iterations=scene_image.shape[0]/1000)
    # fill the holes...
    iters = scene_image.shape[0] / 15
    fg_fat = cv2.dilate(fg_denoise, kernel=None, iterations=iters)
    # ...and trim down to the contour
    fg_slim = cv2.erode(fg_fat, kernel=None, iterations=int(.8*iters))

    # Alternatively, we could have computed the convex hull
    # of the denoised foreground to get a mask
    # using cv2.findContours and cv2.convexHull
    
    return fg_slim
    
# ----------------------------------------------------------------

def get_name(filename):
    '''Get the interesting name part of the file name of an image.'''
    fname = re.sub(r".*/", '', filename)
    name = re.sub(r"\..*$", '', fname)
    return name

# ----------------------------------------------------------------

def detect_object(label, object_image, scene_image, empty_scene):
    fg_mask = foreground_mask(scene_image, empty_scene)

    #show("%s Foreground mask" % label, fg_mask)
    #show("%s Object" % label, object_image)
    #show("%s Scene" % label, scene_image)
    #show("%s Scene masked" % label, cv2.min(scene_image, fg_mask))
    
    # Detection and matching algorithms

    #detector, norm = cv2.SIFT(), cv2.NORM_L2 # SIFT is scale invariant
    detector, norm = cv2.SURF(), cv2.NORM_L2 # SURF is scale and rotation invariant
    #detector, norm = cv2.ORB(10000), cv2.NORM_HAMMING
    
    bf_matcher = cv2.BFMatcher(norm) # brute-force matcher
    if norm == cv2.NORM_L2:
        FLANN_INDEX_KDTREE = 1
        flann_matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = 5),
                                              dict(checks = 50))
    else:
        FLANN_INDEX_LSH = 6
        flann_matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_LSH,
                                                   table_number = 6, key_size = 12, multi_probe_level = 1), {})
         
    matcher = bf_matcher

    
    # Detect features in the reference object
    print "Detecting features..."
    object_keypoints, object_descriptors = detector.detectAndCompute(object_image, None)
    print 'Object has %d features' % len(object_keypoints)
    scene_keypoints, scene_descriptors = detector.detectAndCompute(scene_image, fg_mask)
    print 'Scene has %d features' % len(scene_keypoints)

    obj_kp_image = draw_keypoints(to_bgr(object_image), object_keypoints)
    scene_kp_image = draw_keypoints(to_bgr(scene_image), scene_keypoints)
    
    #show("%s Object Keypoints" % label, obj_kp_image)
    #show("%s Scene Keypoints" % label, scene_kp_image)
    
    # match object and scene
    print "Matching..."
    raw_matches = matcher.knnMatch(object_descriptors, scene_descriptors, k=2)
    p1, p2, kp_pairs = filter_matches(object_keypoints, scene_keypoints, raw_matches)

    MIN_MATCH_COUNT = 10
    print "KP pairs:", label, len(kp_pairs)
    
    if len(kp_pairs)>MIN_MATCH_COUNT:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        match_lines_image = draw_matches(object_image, scene_image, kp_pairs, status, H)
        show("%s Match Lines" % label, match_lines_image)
    else:
        print "** Not enough matches are found - %d/%d" % (len(kp_pairs), MIN_MATCH_COUNT)
        matchesMask = None

# ----------------------------------------------------------------

def detect_pizza(scene_file):
    object_file = '../images/pizza/pizza_box_logo_abstract_640x480.png'
    object_image = cv2.imread(object_file, image_type)
    empty_scene = cv2.imread('../images/pizza/empty_corridor.jpg', image_type)
    scene_image = cv2.imread(scene_file, image_type)

    object_name = get_name(object_file)
    scene_name = get_name(scene_file)
    label = "%s in %s" % (object_name, scene_name)

    detect_object(label, object_image, scene_image, empty_scene)

# ----------------------------------------------------------------

def abstractify_pizza_box_logo():
    pb = cv2.imread('../images/pizza/pizza_box_logo.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    pb = cv2.medianBlur(pb, 5)
    pb = cv2.resize(pb, (640,480))
    ret_val, pb = cv2.threshold(pb, 70, 255, cv2.THRESH_BINARY)
    cv2.imwrite('../images/pizza/pizza_box_logo_abstract_640x480.png', pb)

# ----------------------------------------------------------------

if  __name__ =='__main__':
    image_type = cv2.CV_LOAD_IMAGE_GRAYSCALE
    SAVE_SCREENSHOTS = True

    detect_pizza('../images/pizza/pizza_delivery_1.jpg')
    detect_pizza('../images/pizza/pizza_delivery_2.jpg')
    detect_pizza('../images/pizza/pizza_delivery_3.jpg')

    detect_pizza('../images/pizza/chocolate_delivery_2.jpg')
    detect_pizza('../images/pizza/visitor_1.jpg')
        
    print "Done."
    cv2.destroyAllWindows()
