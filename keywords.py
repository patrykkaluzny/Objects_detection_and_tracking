"""This file contain keywords classes used in project"""
import enum
import cv2 as cv


class Shapes(enum.Enum):
    """Variables used in process of drawing shapes on detected or tracked objects"""
    RECTANGLE = 0
    CIRCLE = 1


class Keys:
    """Class to store keys codes"""
    ESC = 27     # end main loop
    Q = 113       # reset tracking


class Colors:
    """Class to store color values used in project"""
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)


class TrackingMethods:
    """Class to store tracker methods and their names"""
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv.TrackerCSRT_create,
        "kcf": cv.TrackerKCF_create,
        "boosting": cv.TrackerBoosting_create,
        "mil": cv.TrackerMIL_create,
        "tld": cv.TrackerTLD_create,
        "medianflow": cv.TrackerMedianFlow_create,
        "mosse": cv.TrackerMOSSE_create
    }
    MOSSE = "mosse"
    MEDIANFLOW = "medianflow"
    TLD = "tld"
    MIL = "mil"
    BOOSTING = "boosting"
    KCF = "kcf"
    CSRT = "csrt"


