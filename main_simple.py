import cv2 as cv
import numpy as np
from detection import HAAR_Detection
from tracking import Tracking
import sys
from kalman_filter import KalmanFilter
from functions import check_roi, detect_move
from keywords import Shapes as SHAPE
from keywords import Colors as COLOR
from keywords import TrackingMethods as TM









def main():

    # open vid file
    cap = cv.VideoCapture('resources/benchmarkV9.mp4')

    # calculate time between frames of video
    dt = 1 / cap.get(cv.CAP_PROP_FPS)

    # initialize necessary objects
    cascade = cv.CascadeClassifier()
    backSub = cv.createBackgroundSubtractorKNN()

    detection = HAAR_Detection(cascade, 'detection', COLOR.BLUE, SHAPE.RECTANGLE)
    tracking = Tracking('tracking', COLOR.GREEN, SHAPE.RECTANGLE, TM.MOSSE)
    kalman_filter = KalmanFilter(dt, 'kalman_filter', COLOR.WHITE, SHAPE.RECTANGLE)

    # necessary variables
    tracking_counter = 0
    frame_counter = 0
    ticks = 0

    # flags
    is_object_moving = None
    was_object_moving = None
    is_tracking_active = False

    # variables to adjust program
    detection_frame_rate = 8
    tracking_detection_diff = 10

    # check video
    vid_end, frame = cap.read()
    if not vid_end:
        print('Error during loading video')
        sys.exit(0)

    # define window name
    cv.namedWindow('vid_frame')

    # check if cascade is loading properly
    if not cascade.load(cv.samples.findFile('resources/cascades/haarcascade_frontalface_alt.xml')):
        print('Error loading cascade')
        exit(0)

    # main program loop
    while True:

        # read vid frame
        vid_end, frame = cap.read()

        # Check for vid end
        if vid_end:
            frame_to_draw = frame.copy()

            # detect movement on frame for kalman filter porpoise
            is_object_moving = detect_move(frame, backSub)
            if is_object_moving:
                cv.putText(frame_to_draw, "Object is moving", (20, 60), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           COLOR.RED)
            else:
                cv.putText(frame_to_draw, "Object is not moving", (20, 60), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           COLOR.RED)

            if not is_tracking_active:

                # activate tracking if there is object something detected
                detection.detect_and_draw(frame, frame_to_draw)
                detected_objects = detection.get_detected_objects()
                if len(detected_objects) > 0:
                    roi_x, roi_y, roi_width, roi_height = detected_objects[0]
                    tracking.init_tracker(frame, (roi_x, roi_y, roi_width, roi_height))
                    is_tracking_active = True
            else:
                # if tracking is active
                # preforming kalman filter prediction
                kalman_filter.predict_and_draw(frame_to_draw)
                # preforming tracking
                tracking.track_and_draw(frame, frame_to_draw)
                # preforming kalman filter prediction update
                kalman_filter.correct(tracking.get_trackedROI(),is_object_moving)
                # if tracking is active for indicated amount of frames perform detection to correct tracking error
                if tracking_counter is detection_frame_rate-1:
                    detected_ROIs = (detection.detect(frame))
                    # check if there is any object detected, if not pass
                    if len(detected_ROIs) > 0:
                        # to ensure that detected and tracked objects are the same object, choose detected
                        # ROI that is the nearest to the tracked object
                        nearest_ROI_dict = tracking.find_nearest(detected_ROIs)

                        # if error is larger than maximal value reinitialise tracker (because there is no
                        # other option to reset tracker)
                        if nearest_ROI_dict['distance'] > tracking_detection_diff:
                            tracking.init_tracker(frame, tuple(detected_ROIs[0]))

                # count frames without detection
                tracking_counter = (tracking_counter + 1) % detection_frame_rate

            # checking for frame when object not in sight
            if is_object_moving and not was_object_moving:
                if check_roi(tuple(kalman_filter.get_predicted_bb())):
                    tracking.init_tracker(frame, tuple(kalman_filter.get_predicted_bb()))
            was_object_moving = is_object_moving
            frame_counter += 1


            cv.imshow('vid_frame', frame_to_draw)
            key = cv.waitKey(30)
            # press ecc to exit
            if key == 27:
                print("User ended program with ESC button press")
                break
        else:
            print("Video ended")
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()




