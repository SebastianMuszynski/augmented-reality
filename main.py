import cv2

from capture_controller import CaptureController
from constants import KEY_CODE_SPACE, KEY_CODE_TAB, KEY_CODE_ESCAPE, \
    SCREENSHOT_FILENAME, SCREENCAST_FILENAME, WINDOW_NAME
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

    def run(self):
        self._window.create()
        while self._window.is_created:
            self._capture_controller.enter_frame()
            frame = self._capture_controller.frame

            if exists(frame):
                pass

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


if __name__ == "__main__":
    AR().run()
