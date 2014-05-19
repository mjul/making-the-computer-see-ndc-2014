#!/usr/bin/env python

'''
Example code for adaptive filtering and thresholding.
'''

import re
import os

# OpenCV
import numpy as np
import cv2

# Histogram and plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------------------------------------------

def screenshot_filename(title):
    '''Get the filename for the screenshot file based on the title.'''
    return "../screenshots/binarization %s.jpg" % title.replace(':', '')

def save(title, img):
    fname = screenshot_filename(title)
    print "Writing %s..." % fname
    cv2.imwrite(fname, img)

def show(title, img):
    '''Show the image.
       As a side-effect saves the image for use in the presentation.'''    
    cv2.imshow(title, img)
    save(title, img)
    
# ----------------------------------------------------------------
    
def otsu_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    optimal_threshold, binarized_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
    print optimal_threshold
    ret_val, binarized = cv2.threshold(gray, optimal_threshold, 255, cv2.THRESH_BINARY)
    return binarized

# ----------------------------------------------------------------

def adaptive_threshold(img, block_size=7, c=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                      blockSize=block_size, C=c)
    return binarized

# ----------------------------------------------------------------

ADAPTIVE_THRESHOLD_WINDOW = 'Adaptive Threshold'

def put_text(img, text, pos, col=(0,160,0)):
    '''Mutate image, writing the text at the given position.'''
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    cv2.putText(img, text, pos, font_face, font_scale, col)
    return img

def interactive_adaptive_threshold(img):
    cv2.namedWindow(ADAPTIVE_THRESHOLD_WINDOW)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def update_callback(*arg):
        block_size = cv2.getTrackbarPos('Block size', ADAPTIVE_THRESHOLD_WINDOW) + 3
        block_size |= 1 # must be odd
        c = cv2.getTrackbarPos('C', ADAPTIVE_THRESHOLD_WINDOW) - 200
        brightest = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
        coloured = cv2.cvtColor(brightest, cv2.COLOR_GRAY2BGR)
        put_text(coloured, 'Block size = %d' % block_size, (0,50))
        put_text(coloured, 'C          = %d' % c, (0,100))
        cv2.imshow(ADAPTIVE_THRESHOLD_WINDOW, coloured)
        pass
    cv2.createTrackbar('Block size', ADAPTIVE_THRESHOLD_WINDOW, 5-3, 100, update_callback)
    cv2.createTrackbar('C', ADAPTIVE_THRESHOLD_WINDOW, 205, 400, update_callback)
    update_callback()

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

def show_grayscale_histogram(img):
    '''Show a histogram of the grayscale intensity of the image.'''
    fig = plt.figure()
    fig.add_subplot(111, axisbg='#660000')
    n, bins, patches = plt.hist(img.flatten(), 256, normed=True)
    
    # Colour histogram bins according to the grayscale value of the pixel
    cmap = mpl.cm.gray
    b_max = float(max(bins))
    for b,patch in zip(bins, patches):
        # scale bins to 0-1.0 for colour map look-up
        c = cmap(b/b_max) 
        patch.set_color(c)
    plt.title('Histogram of grayscale intensity')
    plt.show(block=False)
    fname = screenshot_filename('Histogram').replace('.jpg', '.png')
    print "Saving %s..." % fname
    plt.savefig(fname)


# ----------------------------------------------------------------

if  __name__ =='__main__':
    raw = cv2.imread('../images/books/Microserfs_p87_2.jpg')
    #page_contour = np.array([[70, 59], [402, 52], [403, 535], [ 66, 526]]) * (3264/640)
    # make it a bit wider to show the gradients
    page_contour = np.array([[40, 59], [402, 52], [403, 535], [ 36, 526]]) * (3264/640)
    page = transform_perspective(raw, page_contour, (480,640))
    cv2.imshow('Page only', page)
    small = page

    show_grayscale_histogram(small)
    interactive_adaptive_threshold(small)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    ret_val, t_mid = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold mid-point', t_mid)
    
    otsu = otsu_threshold(small)
    cv2.imshow('Threshold OTSU', t_mid)

    adaptive = adaptive_threshold(small)
    cv2.imshow('Adaptive', adaptive)
        
    cv2.waitKey()
    cv2.destroyAllWindows()
