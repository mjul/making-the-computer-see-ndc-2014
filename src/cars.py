#!/usr/bin/env python

'''
Example code for detecting car number plates.
'''

import numpy as np
import cv2

def scale_down(img, max_x, max_y):
    blurred = img.copy()
    cv2.GaussianBlur(img, (3,3), .5, blurred)
    height, width, depth = img.shape
    small = None
    if width>height:
        small = cv2.resize(blurred, (640,480))
    else:
        small = cv2.resize(blurred, (480,640))
    return small

# ----------------------------------------------------------------
# Interactive Canny Edge Detection
# ----------------------------------------------------------------

CANNY_WINDOW = 'Canny Edges'

def interactive_canny(img):
    cv2.namedWindow(CANNY_WINDOW)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def update_canny_callback(*arg):
        thrs1 = cv2.getTrackbarPos('threshold1', CANNY_WINDOW)
        thrs2 = cv2.getTrackbarPos('threshold2', CANNY_WINDOW)
        edges = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
        cv2.imshow(CANNY_WINDOW, edges)
        pass
    cv2.createTrackbar('threshold1', CANNY_WINDOW, 2000, 5000, update_canny_callback)
    cv2.createTrackbar('threshold2', CANNY_WINDOW, 4000, 5000, update_canny_callback)
    update_canny_callback()

# ----------------------------------------------------------------
# Interactive Hough Lines
# ----------------------------------------------------------------

HOUGH_WINDOW = 'Hough Lines'

def hough_lines(img, rho, theta, threshold, min_line, max_gap):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 2000, 4000, apertureSize=5)
    ret_val, bw_edges = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated = cv2.dilate(bw_edges, kernel)
    lines = cv2.HoughLinesP(dilated, rho, theta, threshold, minLineLength=min_line, maxLineGap=max_gap)
    hough = img.copy()
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(hough, (x1,y1), (x2,y2), color=(0,255,0), thickness=3)
    return hough

def interactive_hough_lines(img):
    cv2.namedWindow(HOUGH_WINDOW)
    def update_houghlines_callback(*arg):
        rho =  cv2.getTrackbarPos('rho', HOUGH_WINDOW) + 1
        theta = cv2.getTrackbarPos('theta_deg', HOUGH_WINDOW) / 360.0
        threshold = cv2.getTrackbarPos('threshold', HOUGH_WINDOW)
        min_line = cv2.getTrackbarPos('min_line', HOUGH_WINDOW)
        max_gap = cv2.getTrackbarPos('max_gap', HOUGH_WINDOW)
        hough = hough_lines(img, rho, theta, threshold, min_line, max_gap)
        cv2.imshow(HOUGH_WINDOW, hough)
        pass
    cv2.createTrackbar('rho', HOUGH_WINDOW, 10, 100, update_houghlines_callback)
    cv2.createTrackbar('theta_deg', HOUGH_WINDOW, 180, 359, update_houghlines_callback)
    cv2.createTrackbar('threshold', HOUGH_WINDOW, 10, 100, update_houghlines_callback)
    cv2.createTrackbar('min_line', HOUGH_WINDOW, 0, 200, update_houghlines_callback)
    cv2.createTrackbar('max_gap', HOUGH_WINDOW, 0, 100, update_houghlines_callback)
    update_houghlines_callback()

# ----------------------------------------------------------------
# Find Rectangles in image (based on the OpenCV sample squares.py)
# ----------------------------------------------------------------

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_rectangles(img, min_area, cos_epsilon):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    rectangles = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > min_area and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < cos_epsilon:
                        rectangles.append(cnt)
    return rectangles

# ----------------------------------------------------------------

def has_number_plate_shape(r):
    '''Given a rectangle, check if it has the width/heigh ratio of a number plate.'''
    xs = [pt[0] for pt in r]
    ys = [pt[1] for pt in r]
    min_x = np.min(xs)
    max_x = np.max(xs)
    min_y = np.min(ys)
    max_y = np.max(ys)
    width = max_x - min_x
    height = max_y - min_y
    ratio = 1.0 * width / height
    return (ratio > 4 and ratio < 6)
                   
# ----------------------------------------------------------------

def show_plate_shaped_rectangles(image_file):
    raw = cv2.imread(image_file)
    small = scale_down(raw, 640, 480)
    rectangles = find_rectangles(small, 1000, .3)
    plate_shaped_rectangles = [r for r in rectangles if has_number_plate_shape(r)]
    contour = small.copy()
    cv2.drawContours(contour, plate_shaped_rectangles, -1, (0, 255, 0), 3)
    cv2.imshow(str.format('Squares {0}', image_file), contour) 

def show_all_plate_shaped_rectangles():
    from glob import glob
    for image_file in glob('../images/cars/car*.jpg'):
        show_plate_shaped_rectangles(image_file)
    cv2.waitKey()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------

show_all_plate_shaped_rectangles()
    
#raw = cv2.imread('../images/cars/car_AC46749.jpg')
#small = scale_down(raw)
#cv2.imshow('small', small)

#interactive_canny(small)
#interactive_hough_lines(small)
#cv2.waitKey()
#cv2.destroyAllWindows()




