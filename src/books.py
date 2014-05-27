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

SAVE_SCREENSHOTS = False

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
    # findContours mutates the image, work on a copy
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(poly) == 4 and cv2.contourArea(poly) > min_area and cv2.isContourConvex(poly):
            poly = poly.reshape(-1, 2)
            max_cos = np.max([angle_cos( poly[i], poly[(i+1) % 4], poly[(i+2) % 4] ) for i in xrange(4)])
            if max_cos < cos_epsilon:
                quads.append(poly)
    return quads

# ----------------------------------------------------------------

def show(title, img):
    '''Show the image.
       As a side-effect saves the image for use in the presentation.'''
    h,w = img.shape[:2]
    scale_factor = 640.0 / h
    small = cv2.resize(img, (int(scale_factor*w), int(scale_factor*h)))
    cv2.imshow(title, small)
    fname = "../screenshots/books %s.jpg" % title.replace(':', '')
    if SAVE_SCREENSHOTS:
        print "Writing %s..." % fname
        cv2.imwrite(fname, img)
    
# ----------------------------------------------------------------

def find_page_with_canny(img):
    '''Find the page in the image.
       Returns the largest rectangular contour found.'''
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
    largest = sorted(quads, key=lambda q: cv2.contourArea(q))[-1]
    return largest


def create_mask(img, cnt):
    '''Create a mask of the same size as the image
       based on the interior of the contour.'''
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    return mask


# ----------------------------------------------------------------

def binarize_adaptive(img, mask):
    '''Convert an image to binary form using adaptive thresholding,
       restricting to only the mask area (where the mask is non-zero).'''
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray_full, gray_full, mask=mask)
    
    binarized = cv2.adaptiveThreshold(gray_masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                      blockSize=7, C=8)
    return binarized

 
# ----------------------------------------------------------------

def transform_perspective(img, source_quad, dsize):
    '''
    Transform the perspective so the selected quadrilateral is mapped to a rectangle.
    This maps image regions to how they would have looked photographed straight on.
    '''
    points_by_x = sorted([tuple(pt) for pt in source_quad])
    leftmost = points_by_x[0:2:1]
    rightmost = points_by_x[2:4:1]
    top_left, bottom_left = sorted(leftmost, key=lambda pt: pt[1])
    top_right, bottom_right = sorted(rightmost, key=lambda pt: pt[1])
    corners = np.array([top_left, top_right, bottom_right, bottom_left]).astype('float32')
    width, height = dsize
    target = np.array([(0,0), (width,0), (width, height), (0,height)]).astype('float32')
    mpt = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(img, mpt, (width, height), flags=cv2.INTER_CUBIC)

# ----------------------------------------------------------------

def ocr_text(img):
    '''Perform OCR on the image.'''
    tr = Tesseract(lang='eng')
    tr.clear()
    pil_image = pil.Image.fromarray(img)
    tr.set_image(pil_image)
    utf8_text = tr.get_text()
    return utf8_text
        
# ----------------------------------------------------------------

def clip_contour(img, contour):
    '''Return the maximum rectangular subset of the image
       that fits inside the contour.'''
    xs = sorted([xy[0] for xy in contour])
    ys = sorted([xy[1] for xy in contour])

    # clip to the largest inside rectangle
    contour = np.array([[xs[1], ys[1]], [xs[2], ys[1]],
                        [xs[2], ys[2]], [xs[1], ys[2]]])

    x1, x2 = xs[1], xs[2]
    y1, y2 = ys[1] + 200, ys[2] - 100
    w = x2 - x1
    h = y2 - y1
    
    page_clipped = raw[y1:y2, x1:x2, :]
    return page_clipped

# ----------------------------------------------------------------

if  __name__ =='__main__':
    raw = cv2.imread('../images/books/Microserfs_p87_1.jpg')

    SAVE_SCREENSHOTS = True
    scale_factor = raw.shape[0] / 640    # .shape is (height, width, depth)
    
    # Work on a small image for speed and better screen-size fit while developing
    blurred = cv2.GaussianBlur(raw, (3,3), 0)
    small = cv2.resize(blurred, (480,640)) # size is (width,height)
    
    #find_page_with_morphology(small)
    page_contour_small = find_page_with_canny(small)

    # Go back to full resolution to get full fidelity text for OCR
    page_contour = scale_factor * page_contour_small
    page_mask = create_mask(raw, page_contour)

    # This includes the page border
    binarized = binarize_adaptive(raw, page_mask)
        
    # Get the interior of the page, not the edges

    # Make the non-interior part of the page white
    # (erode grows the black, non-page area)
    interior_mask = cv2.erode(page_mask, kernel=(3,3), iterations=40)
    not_interior = cv2.bitwise_not(interior_mask)
    clipped_binarized = cv2.max(binarized, not_interior)
    show('6: Binarized', clipped_binarized)
    
    # Send high-definition image to Tesseract OCR
    # (it will binarize etc.)
    inner_rect = clip_contour(raw, page_contour)
    show('7: OCR region', inner_rect)
    print "OCR..."
    print ocr_text(inner_rect)
    
    print "Press any key..."
    cv2.waitKey()
    cv2.destroyAllWindows()
    print "Done."
