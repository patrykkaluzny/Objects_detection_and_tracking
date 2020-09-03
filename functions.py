import cv2 as cv
import numpy as np


def noise_remove(frame):
    kernel = np.ones((3, 3), np.uint8)
    ret, thresh = cv.threshold(frame, 200, 255, cv.THRESH_BINARY)
    erode = cv.erode(thresh, kernel)
    dilate = cv.dilate(erode, kernel)
    opening = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel, iterations=1)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1)
    return closing


def detect_move(frame, backSub):
    frame_gray = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    fgMask = backSub.apply(frame_gray)
    fgMask = noise_remove(fgMask)
    edges = cv.Canny(fgMask, 50, 100, 3)
    # cv.imshow('edges', edges)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    number_of_filtered_conturs = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 200:
            number_of_filtered_conturs += 1
    if number_of_filtered_conturs > 0:
        return True
    else:
        return False


def check_roi(roi):
    for item in roi:
        if int(item) is 0:
            return False
    return True


