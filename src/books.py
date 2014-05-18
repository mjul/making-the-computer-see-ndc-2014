#!/usr/bin/env python

'''
Example code for converting a book page to a text string.
'''

# OpenCV imports
import numpy as np
import cv2

# Tesseract OCR
from tesserwrap import Tesseract
import PIL as pil



import re
import os

# ----------------------------------------------------------------

def find_page_with_morphology(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(gray, 3)
    cv2.imshow('Median', median)

    dilated = cv2.dilate(median, kernel=None, iterations=2)
    eroded = cv2.erode(median, kernel=None, iterations=2)
    cv2.imshow('Dilated', dilated)
    cv2.imshow('Eroded', eroded)

    gradient = dilated - eroded
    cv2.imshow('Morphological gradient: dilated minus eroded', gradient)

    optimal_threshold, binarized_otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
    ret_val, binarized = cv2.threshold(gradient, optimal_threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binarized', binarized)


# ----------------------------------------------------------------
# Find Rectangles in image (based on the OpenCV sample squares.py)
# ----------------------------------------------------------------

def draw_contours(img, contours):
    '''Draw the contours on the image, returning a new image.'''
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
    return result

def angle_cos(p0, p1, p2):
    '''Given three points, find the cosine of the angle between the edges
       from p0 to p1 and p1 to p2.'''
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_quadrilaterals(img, min_area, cos_epsilon):
    quads = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > min_area and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            if max_cos < cos_epsilon:
                quads.append(cnt)
    return quads


def show(title, img):
    cv2.imshow(title, img)
    fname = "../screenshots/books %s.jpg" % title.replace(':', '')
    print "Writing %s..." % fname
    cv2.imwrite(fname, img)
    
    

def find_page_with_canny(img):
    '''Find the page in the image.'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show('1: Input image', gray)
    # Remove high-frequency noise (e.g. letters)
    median = cv2.medianBlur(gray, 7)
    show('2: Median filtered', median)
    edges = cv2.Canny(median, 10, 20, apertureSize=3)
    show('3: Edges', edges)
    dilated = cv2.dilate(edges, kernel=None, iterations=1)
    show('4: Dilated Edges', dilated)
    min_page_area = img.shape[0] * img.shape[1] / 4
    quads = find_quadrilaterals(dilated, min_area=min_page_area, cos_epsilon=.8)
    quads_found = draw_contours(img, quads)
    show('5: Page found', quads_found)
    


if  __name__ =='__main__':
    raw = cv2.imread('../images/books/Microserfs_p87_2.jpg')
    blurred = cv2.GaussianBlur(raw, (3,3), 0)
    small = cv2.resize(blurred, (480,640))

    #find_page_with_morphology(small)
    find_page_with_canny(small)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
