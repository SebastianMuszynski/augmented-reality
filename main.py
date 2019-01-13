import cv2
import filters
from capture_controller import CaptureController
import constants as const
from drawing import Drawing
from helpers import exists
from model_loader import ModelLoader
from window import Window


class AR:
    def __init__(self, method=const.METHOD_ORB, use_video=False):
        self._window = Window(const.WINDOW_NAME, self.on_keypress)
        self._frame = None
        self._model = ModelLoader(const.MODEL_PATH, swap_yz=True)
        self._img_marker = cv2.imread(const.MARKER_PATH, cv2.IMREAD_GRAYSCALE)
        self._img_marker_2 = cv2.imread(const.MARKER_PATH_HIRO, cv2.IMREAD_GRAYSCALE)
        self._method = method

        self._capture_controller = CaptureController(
            capture=self._get_capture_source(use_video),
            window_manager=self._window,
            mirror_preview=True,
        )

        self._filter = None
        self._draw_contours = False
        self._draw_lines = False
        self._draw_corners = False

    @staticmethod
    def _get_capture_source(use_video=False):
        source = const.VIDEO_PATH if use_video else const.WEB_CAMERA_SOURCE
        return cv2.VideoCapture(source)

    def run(self):
        self._window.create()

        while self._window.is_created:
            self._capture_controller.enter_frame()
            self._frame = self._capture_controller.frame

            if not exists(self._frame):
                continue

            if exists(self._filter):
                self._filter.apply(self._frame, dst=self._frame)

            self.draw(
                contours=self._draw_contours,
                lines=self._draw_lines,
                corners=self._draw_corners,
            )

            self._capture_controller.exit_frame()
            self._window.process_events()

    def draw(self, contours=False, lines=False, corners=False):

        if contours:
            Drawing.draw_contours(self._frame)

        if lines:
            Drawing.detect_lines(self._frame)

        if corners:
            Drawing.detect_corners(self._frame)

        Drawing.match_and_render(self._frame, self._img_marker, self._model, const.METHOD_ORB)

        # Detect two markers at the same time
        # Drawing.match_and_render(self._frame, self._img_marker_2, self._model, const.METHOD_ORB)

    def on_keypress(self, keycode):
        if keycode == const.KEY_CODE_SPACE:
            # Take a screenshot
            self._capture_controller.write_image(const.SCREENSHOT_FILENAME)

        elif keycode == const.KEY_CODE_TAB:
            # Start/stop recording a video
            if not self._capture_controller.is_writing_video:
                self._capture_controller.start_writing_video(const.SCREENCAST_FILENAME)
            else:
                self._capture_controller.stop_writing_video()

        elif keycode == const.KEY_CODE_ESCAPE:
            # Quit
            self._window.destroy()

        elif keycode == const.KEY_CODE_0:
            # Restore default settings
            self._filter = None
            self._draw_contours = False
            self._draw_lines = False
            self._draw_corners = False

        elif keycode == const.KEY_CODE_1:
            self._filter = filters.SharpenFilter() if self._filter is None else None

        elif keycode == const.KEY_CODE_2:
            self._filter = filters.BlurFilter() if self._filter is None else None

        elif keycode == const.KEY_CODE_3:
            self._filter = filters.EdgesFilter() if self._filter is None else None

        elif keycode == const.KEY_CODE_4:
            self._filter = filters.StrokeEdgesFilter() if self._filter is None else None

        elif keycode == const.KEY_CODE_5:
            self._filter = filters.EmbossFilter() if self._filter is None else None

        elif keycode == const.KEY_CODE_6:
            self._draw_contours = not self._draw_contours

        elif keycode == const.KEY_CODE_7:
            self._draw_lines = not self._draw_lines

        elif keycode == const.KEY_CODE_8:
            self._draw_corners = not self._draw_corners

        elif keycode == const.KEY_CODE_9:
            pass


if __name__ == "__main__":
    AR(method=const.METHOD_ORB, use_video=False).run()
