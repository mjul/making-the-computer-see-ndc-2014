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

# ----------------------------------------------------------------
# Image manipulation functions
# ----------------------------------------------------------------

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

def half_size(img):
    return cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))

def glue_2x2(a,b,c,d):
    '''Create a new image from four images by placing the images
    in a 2x2 configuration, a,b over c,d.'''
    return np.ma.vstack((np.ma.hstack((a,b)), np.ma.hstack((c,d))))

def put_text(img, text, pos, col=(0,255,0)):
    '''Mutate image, writing the text at the given position.'''
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    cv2.putText(img, text, pos, font_face, font_scale, col)
    return img


# ----------------------------------------------------------------
# Histograms
# Taken from http://opencv-code.com/tutorials/drawing-histogram-in-python-with-matplotlib/
# ----------------------------------------------------------------

def show_histogram(im):
    """ Function to display image histogram. 
        Supports single and three channel images. """

    if im.ndim == 2:
        # Input image is single channel
        plt.hist(im.flatten(), 256, range=(0, 250), fc='k')
        plt.show()

    elif im.ndim == 3:
        # Input image is three channels
        fig = plt.figure()
        fig.add_subplot(311)
        plt.hist(im[...,0].flatten(), 256, range=(0, 250), fc='b')
        fig.add_subplot(312)
        plt.hist(im[...,1].flatten(), 256, range=(0, 250), fc='g')
        fig.add_subplot(313)
        plt.hist(im[...,2].flatten(), 256, range=(0, 250), fc='r')
        plt.show()

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
        cv2.imshow(str.format('Plate shaped rectangles {0}', image_file), image_with_contours)
    cv2.waitKey()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------
# Detect white and black blobs (plate with black text)
# ----------------------------------------------------------------

SATURATION_WINDOW = 'Saturation'

def detect_low_saturation_blobs(img, s_thresh=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ret_val, mask = cv2.threshold(s, s_thresh, 255, cv2.THRESH_BINARY_INV)
    return mask


def detect_black_white_blobs(img, s_low=50, s_high=200, v_low=20, v_high=130):
    blurred = cv2.GaussianBlur(img, (5,5), 5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # low sat: grey shades (white not included)
    ret_val, low_sat_mask = cv2.threshold(s, s_low, 255, cv2.THRESH_BINARY_INV)
    # high sat
    ret_val, high_sat_mask = cv2.threshold(v, s_high, 255, cv2.THRESH_BINARY)
    # low val: dark/black shades
    ret_val, low_val_mask = cv2.threshold(v, v_low, 255, cv2.THRESH_BINARY_INV)
    # high val: includes white
    ret_val, high_val_mask = cv2.threshold(v, v_high, 255, cv2.THRESH_BINARY)
    grey_or_dark = cv2.min(high_sat_mask, low_val_mask) # grey or dark
    grey_and_bright = cv2.min(low_sat_mask, high_val_mask) # grey and bright (white)
    black_or_white = cv2.max(grey_or_dark, grey_and_bright)
    mask = img.copy()
    mask = cv2.merge([grey_or_dark, low_sat_mask, high_val_mask], mask)
    # Debug code here:
    collage = glue_2x2(*(map(half_size, (low_sat_mask, high_sat_mask, low_val_mask, high_val_mask))))
    collage = cv2.cvtColor(collage, cv2.COLOR_GRAY2BGR)
    height, width = h.shape
    put_text(collage, "Low sat", (0,20))
    put_text(collage, "High sat", (width/2,20))
    put_text(collage, "Low val", (0,20 + height/2))
    put_text(collage, "High val", (width/2,20 + height/2))
    return collage


def interactive_detect_black_white_blobs(img):
    cv2.namedWindow(SATURATION_WINDOW)
    def update_image(*args):
        s_low = cv2.getTrackbarPos('s_low', SATURATION_WINDOW)
        s_high = cv2.getTrackbarPos('s_high', SATURATION_WINDOW)
        v_low = cv2.getTrackbarPos('v_low', SATURATION_WINDOW)
        v_high = cv2.getTrackbarPos('v_high', SATURATION_WINDOW)
        filtered = detect_black_white_blobs(img, s_low, s_high, v_low, v_high)
        cv2.imshow(SATURATION_WINDOW, filtered)
        pass
    cv2.createTrackbar('s_low', SATURATION_WINDOW, 50, 255, update_image)
    cv2.createTrackbar('s_high', SATURATION_WINDOW, 200, 255, update_image)
    cv2.createTrackbar('v_low', SATURATION_WINDOW, 50, 255, update_image)
    cv2.createTrackbar('v_high', SATURATION_WINDOW, 150, 255, update_image)
    update_image()
    
# ----------------------------------------------------------------

def show_hsv(img):
    '''Show the Hue, Saturation and Value channels of an image side-by-side.'''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    height, width = h.shape
    side_by_side = np.ma.hstack((h,s,v))
    side_by_side_colour = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)
    put_text(side_by_side_colour, 'Hue', (0,25))
    put_text(side_by_side_colour, 'Saturation', (width,25))
    put_text(side_by_side_colour, 'Value', (2*width,25))
    cv2.imshow('Hue, Saturation, Value', side_by_side_colour)

# ----------------------------------------------------------------
# Histograms
# ----------------------------------------------------------------



# ----------------------------------------------------------------
# Colour filtering
# ----------------------------------------------------------------

def hue(img):
    '''Get the hue layer of a BGR image.'''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    return h

def hue_close_to(img, hue_mid, max_dist):
    '''Extract a mask for the pixels near a given hue (0-180 excl.).'''
    assert ((0 <= hue_mid) and (hue_mid < 180)), "Expected hue in range (0, 180("
    assert ((0 <= max_dist) and (max_dist < 90)), "Expected max_dist in range (0, 90("
    h = hue(img)
    h_min = hue_mid - max_dist
    h_max = hue_mid + max_dist
    # Hues are 0-180 (degrees), so we need to wrap around
    bands = []
    if (h_min < 0):
        bands.append((h_min % 180, 179))
    if (h_max >= 180):
        bands.append((0, h_max % 180))
    bands.append((max(0,h_min), min(h_max,179)))
    print "bands = ", bands
    masks = [cv2.inRange(h, b[0], b[1]) for b in bands]
    # Masks are 255 for in range, 0 for not, so reduce with max to OR them:
    mask = reduce(lambda x, y: cv2.max(x, y), masks) 
    return mask
            
def whiteish_areas(img):
    h,s,v = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
    h_close = hue_close_to(img, 0, 20)
    ret_val, v_high = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)
    whiteish = cv2.min(h_close, v_high)
    dilated = cv2.dilate(whiteish, kernel=None, iterations=7)
    closed = cv2.erode(dilated, kernel=None, iterations=7)
    return closed

# ----------------------------------------------------------------
# Interactive Adaptive Threshold
# ----------------------------------------------------------------

ADAPTIVE_THRESHOLD_WINDOW = 'Adaptive Threshold'

def interactive_adaptive_threshold(img):
    cv2.namedWindow(ADAPTIVE_THRESHOLD_WINDOW)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def update_callback(*arg):
        block_size = cv2.getTrackbarPos('Block size', ADAPTIVE_THRESHOLD_WINDOW) + 3
        block_size |= 1 # must be odd
        c = cv2.getTrackbarPos('C', ADAPTIVE_THRESHOLD_WINDOW) - 200
        brightest = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        coloured = cv2.cvtColor(brightest, cv2.COLOR_GRAY2BGR)
        put_text(coloured, 'Block size = %d' % block_size, (0,50))
        put_text(coloured, 'C          = %d' % c, (0,100))
        cv2.imshow(ADAPTIVE_THRESHOLD_WINDOW, coloured)
        pass
    cv2.createTrackbar('Block size', ADAPTIVE_THRESHOLD_WINDOW, 25, 100, update_callback)
    cv2.createTrackbar('C', ADAPTIVE_THRESHOLD_WINDOW, 150, 400, update_callback)
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
    match = re.search(r"[A-Z][A-Z] \d\d \d\d\d", text)
    result = None
    if match:
        result = match.group()
    return result
        
# ----------------------------------------------------------------

def select_mask_area(img, mask):
    result = img.copy()
    masked = [cv2.bitwise_and(channel, mask) for channel in cv2.split(img)]
    cv2.merge(masked, result)
    return result

def plate_text_image(img):
    '''Extract the plate text coloured area, everything else is set to white.'''
    blackish_mask = cv2.inRange(img, (0,0,0), (50,50,50))
    dilated_black = cv2.dilate(blackish_mask, kernel=None, iterations=3)
    whiteish_mask = cv2.inRange(img, (150,150,150), (255,255,255))
    dilated_white = cv2.dilate(whiteish_mask, kernel=None, iterations=3)
    plate_mask = cv2.bitwise_or(dilated_black, dilated_white)
    result = select_mask_area(img, plate_mask)
    cv2.imshow('plate_text_mask', np.ma.vstack((img, cv2.cvtColor(plate_mask, cv2.COLOR_GRAY2BGR), result)))
    return result

def match_plates(candidate_plate_images):
    matches = []
    n = 1
    for cp in candidate_plate_images:
        n = n+1
        gray = cv2.cvtColor(cp, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 69, -50)
        ret_val, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        ptx = plate_text_image(cp)
        matches += [m for m in map(ocr_plate, [cp, adaptive, th, ptx]) if m]
    uniques = set(matches)
    return [x for x in uniques]

def show_candidate_plates(candidate_plate_images):
    if len(candidate_plate_images) > 0:
        all_plates = np.ma.vstack(candidate_plate_images)
        cv2.imshow('Candidates', all_plates)

def match_plates_for_file(f):
    match = re.search(r'[A-Z]{2}\d{5}', f)
    plate = ""
    if match:
        plate = match.group()
    window_name = 'Processing %s' % plate
    r = cv2.imread(f)
    s = scale_down(r, 640, 480)
    rects = find_plate_shaped_rectangles(s)
    contours = draw_contours(s, rects)
    cv2.imshow(window_name, contours)
    candidate_plate_images = [transform_perspective(s, r, (250,50)) for r in rects]
    show_candidate_plates(candidate_plate_images)
    uniques = match_plates(candidate_plate_images)
    cv2.destroyWindow(window_name)
    print "FILE: %40s : %10s" % (f, plate), uniques
    
def match_all_plates():
    for f in find_car_image_files():
        match_plates_for_file(f)

# ----------------------------------------------------------------


if  __name__ =='__main__':
    raw = cv2.imread('../images/cars/car_AC46749.jpg')
    #raw = cv2.imread('../images/cars/car_AK62419.jpg')
    #raw = cv2.imread('../images/cars/car_angle_BF27429.jpg')
    small = scale_down(raw, 640, 480)

    # Get the whiteish parts by hue and value filtering and erode/dilate
    #whiteish = whiteish_areas(small)
    #cv2.imshow('Whiteish', whiteish)

    # Try adaptive thresholding
    #interactive_adaptive_threshold(small)
    #h,s,v = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
    #brightest = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, -50)
    #cv2.imshow('Adaptive', brightest)

    # Try line and edge detection
    #interactive_canny(small)
    #interactive_hough_lines(small)

    #cv2.imshow('Plate candidates', draw_plate_shaped_rectangles(small))
    #rects = find_plate_shaped_rectangles(small)
    #candidate_plate_images = [transform_perspective(small, r, (500, 100)) for r in rects]
    #for cp in candidate_plate_images:
    #    gray = cv2.cvtColor(cp, cv2.COLOR_BGR2GRAY)
    #    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 69, -50)
    #    ret_val, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    #    print "MATCHED cp:", ocr_plate(cp)
    #    print "MATCHED bw:", ocr_plate(bw)
    #    print "MATCHED th:", ocr_plate(th)
    #    print "-------------------------------"

    #match_all_plates()
    
    # show_all_plate_shaped_rectangles()
    # show_hsv(small)
    # low_sat = detect_low_saturation_blobs(small)
    # cv2.imshow('Low sat', low_sat)
    # interactive_detect_black_white_blobs(small)

    # nbins = [10]
    # hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # h,s,v = cv2.split(hsv)
    
    #histogram = cv2.calcHist([v],[0], None, nbins, [0,256])
    #histogram = cv2.normalize(histogram, 100)
    #show_histogram(small)

    # reference = cv2.imread('/users/mjul/Downloads/1000px-DK_common_license_plate_1976.svg.png')
    # reference = small
    # hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    # h,s,v = cv2.split(hsv)
    # ref_hist = cv2.calcHist([h],[0], None, nbins, [0,256])
    # plt.title('HSV histograms')
    # fig = plt.figure()
    # for lcrt in zip((h,s,v), ['b','g','r'], (1,2,3), ('Hue', 'Saturation', 'Value')):
    #     layer, colour, row, title = lcrt
    #     fig.add_subplot(3,1, row)
    #     hist = cv2.calcHist([layer], [0], None, nbins, [0 ,256])
    #     hist = cv2.normalize(hist, 100)
    #     plt.plot(hist)
    # plt.show()
    
    #ref_hist = cv2.normalize(ref_hist, 100)

    #print cv2.compareHist(histogram, ref_hist, 1)
    #cv2.NormalizeHist(histogram, 1.0)
    #cv2.NormalizeHist(ref_hist, 1.0)

    #plt.plot(histogram, 'b')
    #plt.plot(ref_hist, 'g')
    #plt.xlim([0,256])
    #plt.show()

    #show_histogram(small)


    #print "Press any key..."
    #cv2.waitKey()
    cv2.destroyAllWindows()

