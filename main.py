import cv2

import filters
import utils
from capture_controller import CaptureController
from constants import KEY_CODE_SPACE, KEY_CODE_TAB, KEY_CODE_ESCAPE, \
    SCREENSHOT_FILENAME, SCREENCAST_FILENAME, WINDOW_NAME, KEY_CODE_0, \
    KEY_CODE_1, KEY_CODE_2, KEY_CODE_3, KEY_CODE_4, KEY_CODE_5, KEY_CODE_6
from helpers import exists
from window import Window


class AR:
    def __init__(self):
        self._window = Window(WINDOW_NAME, self.on_keypress)
        self._capture_controller = CaptureController(
            capture=cv2.VideoCapture(0),
            window_manager=self._window,
            mirror_preview=True,
        )
        self._frame = None
        self._filter = None

    def run(self):
        self._window.create()
        while self._window.is_created:
            self._capture_controller.enter_frame()
            self._frame = self._capture_controller.frame

            if exists(self._frame):

                # Apply a filter
                if exists(self._filter):
                    self._filter.apply(self._frame, dst=self._frame)

                # Draw contours
                # utils.draw_contours(self._frame)

                # Detect lines
                # utils.detect_lines(self._frame)

                # Detect corners (Harris)
                utils.detect_corners(self._frame)

            self._capture_controller.exit_frame()
            self._window.process_events()

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
            self._filter = None

        elif keycode == KEY_CODE_1:
            self._filter = filters.SharpenFilter()

        elif keycode == KEY_CODE_2:
            self._filter = filters.BlurFilter()

        elif keycode == KEY_CODE_3:
            self._filter = filters.EdgesFilter()

        elif keycode == KEY_CODE_4:
            self._filter = filters.StrokeEdgesFilter()

        elif keycode == KEY_CODE_5:
            self._filter = filters.EmbossFilter()


if __name__ == "__main__":
    AR().run()
