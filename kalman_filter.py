import cv2 as cv
import numpy as np
from drawing import Drawing
from keywords import Shapes as SHAPE
from functions import check_roi


class KalmanFilter(Drawing):
    def __init__(self, dT, object_name=None, contour_color=(255, 255, 255), contour_shape=SHAPE.RECTANGLE):
        super().__init__(object_name, contour_color, contour_shape)
        self.first_run = True
        self.dynamParams = 6
        self.measureParams = 4
        self.kalman = cv.KalmanFilter(dynamParams=self.dynamParams, measureParams=self.measureParams)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0, 1]], np.float32)
        # Transition matrix ( eg. p(k) = p(k-1) + v(k-1)*dT ), init dT = 1
        self.kalman.transitionMatrix = np.array([[1, 0, dT, 0, 0, 0],
                                                 [0, 1, 0, dT, 0, 0],
                                                 [0, 0, 1, 0, 0, 0],
                                                 [0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 1]], np.float32)
        # process noise covariance matrix (rough values)
        self.kalman.processNoiseCov = np.array([[0.01, 0, 0, 0, 0, 0],
                                                [0, 0.01, 0, 0, 0, 0],
                                                [0, 0, 2.0, 0, 0, 0],
                                                [0, 0, 0, 1.0, 1.0, 0],
                                                [0, 0, 0, 0, 1.0, 0.01],
                                                [0, 0, 0, 0, 0, 0.01]], np.float32)
        # measurement noise covariance matrix (rough values)
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.not_found_count = 0
        self.initial_state = None
        self.velocity = None
        self.velocity_unit_vec = None

    def reset(self):
        self.kalman.__init__(dynamParams=self.dynamParams, measureParams=self.measureParams)
        self.initial_state = None

    def get_predicted_bb(self):
        pred = self.kalman.predict().T[0]
        pred_bb = np.array([pred[0], pred[1], pred[4], pred[5]])

        return pred_bb

    def predict_and_draw(self, frame_to_draw):
        x, y, w, h = tuple(self.get_predicted_bb())
        self._draw_contour(frame_to_draw, x, y, w, h)

    def get_current_velocity(self):
        return self.velocity.copy()

    def get_current_unit_velocity(self):
        return self.velocity_unit_vec.copy()

    def correct(self, bb, is_object_in_sight):
        if is_object_in_sight and check_roi(bb):
            # measurement is numpy array [[x1,y1,x2,y2]]
            measurement = np.array([bb], dtype=np.float32).T
            if self.first_run is True:
                self.kalman.statePre = np.array([measurement[0], measurement[1], [0], [0], measurement[2], measurement[3]],
                                                dtype=np.float32)
                self.first_run = False
            corr_bb = self.kalman.correct(measurement).T[0]
            self.velocity = np.array([corr_bb[2], corr_bb[3]])

            self.velocity_unit_vec = self.velocity / np.linalg.norm(self.velocity + 1e-6)
        else:
            self.not_found_count += 1
            if self.not_found_count >= 100:
                self.first_run = True
                self.not_found_count = 0
