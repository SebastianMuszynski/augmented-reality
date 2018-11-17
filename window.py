import cv2

from helpers import exists


class Window:
    def __init__(self, name, keypress_callback=None):
        self._name = name
        self._is_created = False
        self.keypress_callback = keypress_callback

    @property
    def is_created(self):
        return self._is_created

    def create(self):
        cv2.namedWindow(self._name)
        self._is_created = True

    def show(self, frame):
        cv2.imshow(self._name, frame)

    def destroy(self):
        cv2.destroyWindow(self._name)
        self._is_created = False

    def process_events(self):
        keycode = cv2.waitKey(1)
        if exists(self.keypress_callback) and keycode != -1:
            keycode &= 0xFF
            self.keypress_callback(keycode)
