import cv2

import filters
import utils
from capture_controller import CaptureController
from constants import KEY_CODE_SPACE, KEY_CODE_TAB, KEY_CODE_ESCAPE, \
    SCREENSHOT_FILENAME, SCREENCAST_FILENAME, WINDOW_NAME, KEY_CODE_0, \
    KEY_CODE_1, KEY_CODE_2, KEY_CODE_3, KEY_CODE_4, KEY_CODE_5, KEY_CODE_6, \
    KEY_CODE_7, KEY_CODE_8, KEY_CODE_9
from helpers import exists
from model_loader import ModelLoader
from window import Window
import numpy as np


class AR:
    MODEL_PATH = './models/fox.obj'

    def __init__(self):
        self._window = Window(WINDOW_NAME, self.on_keypress)
        self._frame = None

        self._capture_controller = CaptureController(
            capture=cv2.VideoCapture(0),
            window_manager=self._window,
            mirror_preview=True,
        )

        self._filter = None
        self._draw_contours = False
        self._draw_lines = False
        self._draw_corners = False
        self._show_marker = False

        self._model = None
        self._homography = None
        self._camera_parameters = np.array([[800, 0, 320],
                                            [0, 800, 240],
                                            [0,   0,   1]])

    def run(self):
        self._model = ModelLoader(self.MODEL_PATH, swap_yz=True)
        self._window.create()

        while self._window.is_created:
            self._capture_controller.enter_frame()
            self._frame = self._capture_controller.frame

            if not exists(self._frame):
                continue

            # Apply a filter if exists
            if exists(self._filter):
                self._filter.apply(self._frame, dst=self._frame)

            self.draw(
                contours=self._draw_contours,
                lines=self._draw_lines,
                corners=self._draw_corners,
                features=True,
            )

            self._capture_controller.exit_frame()
            self._window.process_events()

    def draw(self, contours=False, lines=False, corners=False, features=False, flann=False):

        if contours:
            utils.draw_contours(self._frame)

        if lines:
            utils.detect_lines(self._frame)

        if corners:
            utils.detect_corners(self._frame)

        if features:
            # Feature matching
            utils.match_and_render(
                self._frame,
                self._camera_parameters,
                self._model,
                show_marker=self._show_marker,
            )

        if flann:
            flann_frame = utils.match_and_render(
                self._frame,
                self._camera_parameters,
                self._model,
                show_marker=True,
            )

            if exists(flann_frame):
                self._window.show(flann_frame)

    def on_keypress(self, keycode):
        if keycode == KEY_CODE_SPACE:
            # Take a screenshot
            self._capture_controller.write_image(SCREENSHOT_FILENAME)

        elif keycode == KEY_CODE_TAB:
            # Start/stop recording a video
            if not self._capture_controller.is_writing_video:
                self._capture_controller.start_writing_video(SCREENCAST_FILENAME)
            else:
                self._capture_controller.stop_writing_video()

        elif keycode == KEY_CODE_ESCAPE:
            # Quit
            self._window.destroy()

        elif keycode == KEY_CODE_0:
            # Restore default settings
            self._filter = None
            self._draw_contours = False
            self._draw_lines = False
            self._draw_corners = False
            self._show_marker = False

        elif keycode == KEY_CODE_1:
            self._filter = filters.SharpenFilter() if self._filter is None else None

        elif keycode == KEY_CODE_2:
            self._filter = filters.BlurFilter() if self._filter is None else None

        elif keycode == KEY_CODE_3:
            self._filter = filters.EdgesFilter() if self._filter is None else None

        elif keycode == KEY_CODE_4:
            self._filter = filters.StrokeEdgesFilter() if self._filter is None else None

        elif keycode == KEY_CODE_5:
            self._filter = filters.EmbossFilter() if self._filter is None else None

        elif keycode == KEY_CODE_6:
            self._draw_contours = not self._draw_contours

        elif keycode == KEY_CODE_7:
            self._draw_lines = not self._draw_lines

        elif keycode == KEY_CODE_8:
            self._draw_corners = not self._draw_corners

        elif keycode == KEY_CODE_9:
            self._show_marker = not self._show_marker


if __name__ == "__main__":
    AR().run()
