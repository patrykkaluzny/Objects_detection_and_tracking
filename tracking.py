import cv2 as cv
from drawing import Drawing
from keywords import Colors as COLOR
from keywords import TrackingMethods as TM
import sys
from scipy.spatial import distance
from keywords import Shapes as SHAPE


class Tracking(Drawing):
    def __init__(self, object_name=None, contour_color=(0, 255, 255), contour_shape=SHAPE.RECTANGLE, tracking_algorithm=TM.MOSSE):
        super().__init__(object_name, contour_color, contour_shape)
        self._tracking_ROI = None
        self._tracker_algorithm = tracking_algorithm
        self._tracker = None

    def get_trackedROI(self):
        return self._tracking_ROI

    def track_and_draw(self, frame, frame_to_draw):
        is_ok, self._tracking_ROI = self._tracker.update(frame)
        # check if tracking works
        if is_ok:
            x, y, width, height = self._tracking_ROI
            self._draw_contour(frame_to_draw, int(x), int(y), int(width), int(height))

        else:
            self._display_tracking_error(frame_to_draw)


    def init_tracker(self, frame, tracking_ROI):
        # initialize tracker
        self._tracker = TM.OPENCV_OBJECT_TRACKERS[self._tracker_algorithm]()
        self._tracking_ROI = tracking_ROI
        is_ok = self._tracker.init(frame, self._tracking_ROI)

        # check if initialization went wrong
        if not is_ok:
            print('Error during tracker initialization')
            sys.exit()

    def _display_tracking_error(self, frame):
        cv.putText(frame, "Tracking failure detected", (20, 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR.RED)

    def find_nearest(self, detected_rois):
        # from given ROIs choose which one is the closest
        result = {"distance": 10000, 'detected_ROI': None, 'center_point': None}  #10000
        tracked_roi_center = self._get_rect_center(self._tracking_ROI)
        for detected_roi in detected_rois:
            detected_roi_center = self._get_rect_center(detected_roi)
            dist = int(distance.euclidean(detected_roi_center, tracked_roi_center))
            if dist < result['distance']:
                result.update({'distance': dist, 'detected_ROI': detected_roi, 'center_point': detected_roi_center})
        return result

    def _check_roi_correction(self, tuple):
        for item in tuple:
            if int(item) is 0:
                return False
        return True









