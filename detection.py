import cv2 as cv
from drawing import Drawing
from keywords import Shapes as SHAPE


class HAAR_Detection(Drawing):
    def __init__(self, cascade_classifier, object_name=None, contour_color=(255, 0, 255), contour_shape=SHAPE.RECTANGLE):
        super().__init__(object_name, contour_color, contour_shape)
        self._cascade_classifier = cascade_classifier
        self._method_name = 'detection'
        self._detected_objects = None
        self.object_founded = False

    def detect_and_draw(self, frame, frame_to_draw):
        self._detected_objects = self.detect(frame)
        if len(self._detected_objects) == 0:
            self.object_founded = False
        else:
            self.object_founded = True
            for (x, y, width, height) in self._detected_objects:
                self._draw_contour(frame_to_draw, x, y, width, height)

    def _process_frame(self, frame):
        processed_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
        processed_frame = cv.equalizeHist(processed_frame)
        return processed_frame

    def detect(self, frame):
        processed_frame = self._process_frame(frame)
        detected_objects = self._cascade_classifier.detectMultiScale(processed_frame)
        return detected_objects

    def get_detected_objects(self):
        return self._detected_objects









