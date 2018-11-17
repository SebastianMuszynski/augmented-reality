import cv2
import numpy as np
import time

from helpers import exists

VIDEO_ENCODING = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
MIN_FRAMES_FOR_FPS_ESTIMATE = 20


class CaptureController:
    def __init__(self, capture, window_manager=None, mirror_preview=False):
        self.window_manager = window_manager
        self.mirror_preview = mirror_preview

        self._capture = capture
        self._channel = 0
        self._frame_entered = False
        self._frame = None
        self._image_filename = None
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = 0
        self._fps_estimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if exists(self._frame_entered) and not exists(self._frame):
            _, self._frame = self._capture.retrieve()
        return self._frame

    def fps(self):
        captured_fps = self._capture.get(cv2.CAP_PROP_FPS)
        if captured_fps <= 0:
            return self._estimated_fps()
        return captured_fps

    def _estimated_fps(self):
        if self._frames_elapsed < MIN_FRAMES_FOR_FPS_ESTIMATE:
            return None
        return self._fps_estimate

    @property
    def is_writing_image(self):
        return exists(self._image_filename)

    @property
    def is_writing_video(self):
        return exists(self._video_filename)

    def enter_frame(self):
        assert not self._frame_entered

        if exists(self._capture):
            self._frame_entered = self._capture.grab()

    def exit_frame(self):
        if not exists(self.frame):
            self._frame_entered = False
            return

        self._count_frames()
        self.show_preview()
        self._write_image_frame()
        self._write_video_frame()

        self._frame = None
        self._frame_entered = False

    def _count_frames(self):
        if self._frames_elapsed <= 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed

        self._frames_elapsed += 1

    def show_preview(self):
        if not exists(self.window_manager):
            return

        if self.mirror_preview:
            mirrored_frame = np.fliplr(self._frame).copy()
            self.window_manager.show(mirrored_frame)
        else:
            self.window_manager.show(self._frame)

    def write_image(self, filename):
        self._image_filename = filename

    def _write_image_frame(self):
        if self.is_writing_image:
            cv2.imwrite(self._image_filename, self._frame)
            self._image_filename = None

    def start_writing_video(self, filename, encoding=VIDEO_ENCODING):
        self._video_filename = filename
        self._video_encoding = encoding

    def stop_writing_video(self):
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return

        if not exists(self._video_writer):
            video_fps = self.fps()

            if not exists(video_fps):
                return

            video_size = (
                int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            self._video_writer = cv2.VideoWriter(
                self._video_filename,
                self._video_encoding,
                video_fps,
                video_size,
            )
        self._video_writer.write(self._frame)
