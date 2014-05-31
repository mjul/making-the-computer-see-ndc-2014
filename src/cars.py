#!/usr/bin/env python

'''
Example code for detecting car number plates.
'''

# OpenCV imports
import numpy as np
import cv2

# Histogram and plotting
import matplotlib.pyplot as plt

# Tesseract OCR
from tesserwrap import Tesseract
import PIL as pil

# Other
import re
import glob
import collections

# ----------------------------------------------------------------

PLOT_MATCHES = False
SAVE_SCREENSHOTS = False

# ----------------------------------------------------------------
# Saving images for the presentation
# ----------------------------------------------------------------

def screenshot_filename(title):
    return "../screenshots/cars %s.jpg" % title.replace(':', '')

def save_image(title, img):    
    fname = screenshot_filename(title)
    print "Writing %s..." % fname
    cv2.imwrite(fname, img)

def save_plot(title, plot):
    fname = screenshot_filename(title).replace('.jpg', '.png')
    print "Writing %s..." % fname
    plot.savefig(fname, bbox_inches='tight', transparent=False, facecolor='wheat')

def show(title, img):
    '''Show the image.
       As a side-effect saves the image for use in the presentation.'''
    h,w = img.shape[:2]
    if h < 640:
        scale_factor = 1
    else:
        scale_factor = 640.0 / h
    small = cv2.resize(img, (int(scale_factor*w), int(scale_factor*h)))
    cv2.imshow(title, small)
    if SAVE_SCREENSHOTS:
        save_image(title, img)

# ----------------------------------------------------------------
# Image manipulation functions
# ----------------------------------------------------------------

def scale_down(img, max_x, max_y):
    blurred = cv2.GaussianBlur(img, (3,3), .5)
    height, width, depth = img.shape
    small = None
    if width>height:
        small = cv2.resize(blurred, (640,480))
    else:
        small = cv2.resize(blurred, (480,640))
    return small

# ----------------------------------------------------------------
# Find Rectangles in image (based on the OpenCV sample squares.py)
# ----------------------------------------------------------------

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_rectangles(img, min_area, cos_epsilon):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    rectangles = []
    for gray in cv2.split(blurred):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                poly = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(poly) == 4 and cv2.contourArea(poly) > min_area and cv2.isContourConvex(poly):
                    poly = poly.reshape(-1, 2)
                    max_cos = np.max([angle_cos(poly[i], poly[(i+1) % 4], poly[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < cos_epsilon:
                        rectangles.append(poly)
    return rectangles

# ----------------------------------------------------------------
# Detect rectangles of number plate shape.
# ----------------------------------------------------------------

def find_car_image_files():
    '''Get the paths to all car images'''
    return glob.glob('../images/cars/car*.jpg')

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

def find_plate_shaped_rectangles(img):
    '''Find the plate-shaped rectangles in the image and return them as a list.'''
    rectangles = find_rectangles(img, 1000, .3)
    plate_shaped_rectangles = [r for r in rectangles if has_number_plate_shape(r)]
    return plate_shaped_rectangles

def draw_contours(img, contours):
    '''Draw the contours on the image, returning a new image.'''
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
    return result

def show_all_plate_shaped_rectangles():
    for image_file in find_car_image_files():
        raw = cv2.imread(image_file)
        small = scale_down(raw, 640, 480)
        contours = find_plate_shaped_rectangles(small)
        image_with_contours = draw_contours(small, contours)
        show(str.format('Plate shaped rectangles {0}', image_file), image_with_contours)
    cv2.waitKey()
    cv2.destroyAllWindows()

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
# Text recognition
# ----------------------------------------------------------------

def ocr_text(img):
    tr = Tesseract(lang='eng')
    tr.clear()
    pil_image = pil.Image.fromarray(img)
    # Turn off OCR word dictionaries
    tr.set_variable('load_system_dawg', "F")
    tr.set_variable('load_freq_dawg', "F")
    tr.set_variable('-psm', "7") # treat image as single line
    tr.set_variable('tessedit_char_whitelist', "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    tr.set_image(pil_image)
    utf8_text = tr.get_text()
    return unicode(utf8_text)

def ocr_plate(img):
    text = ocr_text(img)
    match = re.search(r"[A-Z][A-Z] *\d{2} *\d{3}", text)
    result = None
    if match:
        result = match.group()
        # Canonicalize to no spacing
        result = result.replace(' ', '')
    return result

# ----------------------------------------------------------------

def plot_image_variants_and_matches(title, image_variants, image_matches):
    '''Plot the variants of the image with the corresponding OCR plate matches.'''
    plt.figure(facecolor='wheat')
    plt.subplots_adjust(wspace=0.5, hspace=0.8)
    for img, plate, i in zip(image_variants, image_matches, xrange(len(image_variants))):
        plt.subplot(len(image_variants), 1, i)
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.axis('off')
        plt.title(plate if plate else "(No plate)")
    plt.show(block=False)
    if SAVE_SCREENSHOTS: save_plot(title, plt)

# ----------------------------------------------------------------

def match_plates(title, candidate_plate_images):
    '''Returns a list of the possible matches for the plate.'''
    matches = []
    n = 0 
    for cp in candidate_plate_images:
        # Filter to different representations
        n = n+1
        gray = cv2.cvtColor(cp, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 15)
        #cv2.imshow('Adaptive', adaptive)
        ret_val, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        #cv2.imshow('Thresh', th)
        t_otsu, th_otsu = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret_val, th_otsu = cv2.threshold(gray, t_otsu, 255, cv2.THRESH_BINARY)
        #cv2.imshow('OTSU', th_otsu)
        # thin the black parts (letters and noise)
        dilated = cv2.dilate(th_otsu, kernel=(5,5), iterations=2)
        # thicken the letters
        eroded = cv2.erode(dilated, None)
        #cv2.imshow('Dilated', dilated)
        #cv2.imshow('Eroded', eroded)
        image_variants = [cp, adaptive, th, th_otsu, dilated, eroded]
        image_matches = [m for m in map(ocr_plate, image_variants)]
        if PLOT_MATCHES: plot_image_variants_and_matches("%s matches %d" % (title, n), image_variants, image_matches)
        matches +=  [m for m in image_matches if m]
    return matches

def show_candidate_plates(title, candidate_plate_images):
    if len(candidate_plate_images) > 0:
        all_plates = np.ma.vstack(candidate_plate_images)
    else:
        all_plates = np.zeros((500,100,3))
    show(title, all_plates)

def match_plates_for_file(f):
    match = re.search(r'[A-Z]{2}\d{5}', f)
    plate = ""
    if match:
        plate = match.group()
    window_name = '%s Contours' % plate
    r = cv2.imread(f)
    s = scale_down(r, 640, 480)
    rects = find_plate_shaped_rectangles(s)
    contours = draw_contours(s, rects)
    show(window_name, contours)
    candidate_plate_images = [transform_perspective(s, r, (250,50)) for r in rects]
    show_candidate_plates("%s Candidate Plates" % plate, candidate_plate_images)
    matches = match_plates("%s Match Plates" % plate, candidate_plate_images)
    best = "-"
    if len(matches) > 0:
        best = collections.Counter(matches).most_common(1)[0][0]
    
    cv2.destroyWindow(window_name)
    is_match = "OK" if (best == plate) else "-"
    print "FILE: %40s : %10s   %s" % (f, plate, is_match)
    
def match_all_plates():
    for f in find_car_image_files():
        # Danish car images only
        if re.search(r"car_[A-Z][A-Z]\d\d\d\d\d.jpg", f):
            match_plates_for_file(f)

# ----------------------------------------------------------------

# Note: this is not very good...
def match_plates_simple(img):
    '''Find and match the plates on an image using simple thresholding algorithms.'''
    rects = find_plate_shaped_rectangles(small)
    show('Plate candidates', draw_contours(small, rects))
    candidate_plate_images = [transform_perspective(small, r, (500, 100)) for r in rects]
    for cp in candidate_plate_images:
        gray = cv2.cvtColor(cp, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 15)
        ret_val, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        print "MATCHED cp:", ocr_plate(cp)
        print "MATCHED bw:", ocr_plate(bw)
        print "MATCHED th:", ocr_plate(th)
        print "-------------------------------"

# ----------------------------------------------------------------

if  __name__ =='__main__':
    raw = cv2.imread('../images/cars/car_AC46749.jpg')
    raw = cv2.imread('../images/cars/car_AK62419.jpg')
    #raw = cv2.imread('../images/cars/car_angle_BF27429.jpg')
    #raw = cv2.imread('../images/cars/car_XJ41721.jpg')
    small = scale_down(raw, 640, 480)

    PLOT_MATCHES = True
    SAVE_SCREENSHOTS = True
    
    print "match_plates_for_file..."
    #match_plates_for_file('../images/cars/car_XJ41721.jpg')
    #match_all_plates()
    match_plates_for_file('../images/cars/car_AC46749.jpg')
    match_plates_for_file('../images/cars/car_AK62419.jpg')
    
    print "Press any key..."
    cv2.waitKey()
    cv2.destroyAllWindows()
    plt.close('all')

