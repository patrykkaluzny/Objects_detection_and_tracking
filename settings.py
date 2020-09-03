from keywords import TrackingMethods as TM


class Settings:
    """A class to store all settings of object detection application"""

    def __init__(self):
        self.detection_rate = 10  # determine step of preforming detection during tracking
        self.cascade_path = 'resources/cascades/haarcascade_frontalface_alt.xml'  # 'resources/cascades/haarcascade_frontalface_alt.xml' ||||| 'resources/cascades/haarcascade_cars.xml'
        self.vid_path = 'resources/benchmarkV2.mp4'  # set to 0 for camera output ||||| 'resources/cropped_video.mp4' ||||| 'resources/benchmark.mp4'
        self.save = False  # set to true to save video with detected objects
        self.save_path = 'resources/test_vid.avi'
        self.frame_time = 30  # determine step of showing frames in ms
        self.tracking_algorithm = TM.MOSSE  # determine which tracker algorithm to use
        self.max_error = 10  # determine maximal difference between centers of ROIs of tracked and detected objects

