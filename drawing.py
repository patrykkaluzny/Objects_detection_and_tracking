import cv2 as cv
import numpy as np
from keywords import Shapes as SHAPE

class Drawing:
    """Class to provide drawing methods """
    def __init__(self, object_name=None, contour_color=(255, 0, 255), contour_shape=None):
        self._object_name = object_name
        self._contour_color = contour_color
        if contour_shape is not None:
            self._contour_shape = contour_shape
        else:
            self._contour_shape = SHAPE.RECTANGLE
        self._method_name = None

    def _draw_contour(self, frame, x, y, width, height):
        if self._contour_shape is SHAPE.CIRCLE:
            self._draw_circle(frame, x, y, width, height)
        else:
            self._draw_rectangle(frame, x, y, width, height)
        if self._object_name is not None:
            text_point = (int(x + width * 0.1), int(y + height * 0.9))
            self._put_object_name(frame, text_point)

    def _draw_rectangle(self, frame, x, y, width, height):
        point1 = (x, y)
        point2 = (x+width, y+height)
        cv.rectangle(frame, point1, point2, self._contour_color)

    def _draw_circle(self, frame, x, y, width, height):
        center = (x + width // 2, y + height // 2)
        radius = int(round((width + height) * 0.25))
        cv.circle(frame, center, radius, self._contour_color)

    def _put_object_name(self, frame, text_point):
        font = cv.FONT_HERSHEY_COMPLEX_SMALL
        cv.putText(frame, self._object_name, text_point, font, 1, self._contour_color)

    def _put_signature(self, frame):
        rows, cols = frame.shape[:2]
        font = cv.FONT_HERSHEY_COMPLEX_SMALL
        signature_point = (int(rows * 0.02), int(cols * 0.02))
        cv.putText(frame, self._method_name, signature_point, font, 1, self._contour_color)

    def _get_rect_center(self, roi):
        x, y, width, height = roi
        center_point = np.array(((x + width) / 2, (y + height) / 2), dtype=int)
        return center_point


