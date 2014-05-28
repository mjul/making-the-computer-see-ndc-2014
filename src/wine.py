#!/usr/bin/env python

'''
Example code for wine recognition section.
'''

# OpenCV imports
import numpy as np
import cv2

# Tesseract OCR
from tesserwrap import Tesseract
import PIL as pil

# Histogram and plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

import re
import os

# ----------------------------------------------------------------

SAVE_SCREENSHOTS = False

# ----------------------------------------------------------------

def screenshot_filename(title):
    return "../screenshots/wine %s.jpg" % title.replace(':', '')

def save_image(title, img):    
    fname = screenshot_filename(title)
    print "Writing %s..." % fname
    cv2.imwrite(fname, img)

def save_plot(title, plot):
    fname = screenshot_filename(title).replace('.jpg', '.png')
    print "Writing %s..." % fname
    plot.savefig(fname)


def show(title, img):
    '''Show the image.
       As a side-effect saves the image for use in the presentation.'''
    h,w = img.shape[:2]
    scale_factor = 640.0 / h
    small = cv2.resize(img, (int(scale_factor*w), int(scale_factor*h)))
    cv2.imshow(title, small)
    if SAVE_SCREENSHOTS:
        save_image(title, img)

# ----------------------------------------------------------------

def show_wine_histograms(title, img):
    '''Show a histogram of the Hue values of the HSV representation of the image.'''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v, = cv2.split(hsv)

    fig = plt.figure(figsize=(16,4))
    plt.title(title)
    
    # First the image
    plt.subplot(1, 4, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.axis('off')

    # Histogram of mean colours by hue bin 
    plt.subplot(1, 4, 2)
    plt.title('Mean colour by hue')
    nbins, range_max = 20, 180
    bin_size = range_max/nbins
    plt.xlim(0, range_max)
    n, bins, patches = plt.hist(h.flatten(), bins=nbins, range=(0,range_max), normed=True)
    cmap = mpl.cm.hsv
    b_max = float(max(bins))
    for b,patch in zip(bins, patches):
        bin_low = b
        bin_high = bin_low + bin_size
        in_bin_mask = cv2.inRange(h, bin_low, bin_high)
        mean_colour = cv2.mean(rgb, mask=in_bin_mask)
        # scale bins to 0-1.0 for colour map look-up
        c = (mean_colour[0]/255.0, mean_colour[1]/255.0, mean_colour[2]/255.0)
        patch.set_color(c)
        patch.set_edgecolor('black')

    # Histogram of HUE 
    plt.subplot(1, 4, 3)
    plt.title('H : Hue')
    n, bins, patches = plt.hist(h.flatten(), bins=nbins, range=(0,180), normed=True)
    plt.xlim(0, 180)
    cmap = mpl.cm.hsv
    b_max = float(max(bins))
    for b,patch in zip(bins, patches):
        # scale bins to 0-1.0 for colour map look-up
        c = cmap(b/b_max) 
        patch.set_color(c)

    # Histogram V (intensity)
    plt.subplot(1, 4, 4)
    plt.title('V: Value')
    n, bins, patches = plt.hist(v.flatten(), bins=nbins, range=(0,256), normed=True)
    plt.xlim(0, 256)
    cmap = mpl.cm.gray
    b_max = float(max(bins))
    for b,patch in zip(bins, patches):
        # scale bins to 0-1.0 for colour map look-up
        c = cmap(b/b_max) 
        patch.set_color(c)
        patch.set_edgecolor('black')

    plt.show(block=False)
    if SAVE_SCREENSHOTS:
        save_plot(title, plt)

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

def distinct_colours(img, k=8):
    # Work in HSV colour space for k-means
    Z_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Z_hsv_flat = Z_hsv.reshape((-1,3))
    Z = np.float32(Z_hsv_flat)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z, k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    result_hsv_flat = centers[label.flatten()]
    result_hsv = result_hsv_flat.reshape((img.shape))
    result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    return result

# ----------------------------------------------------------------

if  __name__ =='__main__':
    dom = cv2.imread('../images/wine/dom-p-2004_375x500.tif')
    mf = cv2.imread('../images/wine/mf-pinot-2011_375x500.tif')
    
    SAVE_SCREENSHOTS = True

    for title,img in [["Dom P", dom], ["MF", mf]]:
        show_wine_histograms(title, img)
        distinct = distinct_colours(img, k=12)
        show_wine_histograms("%s distinct" % title, distinct)
        
    #print "Press any key..."
    #cv2.waitKey()
    cv2.destroyAllWindows()
    print "Done."
    
