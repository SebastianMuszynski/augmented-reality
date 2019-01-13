import numpy as np

WINDOW_NAME = 'AR'

SCREENSHOT_FILENAME = 'screenshot.png'
SCREENCAST_FILENAME = 'screencast.avi'

WEB_CAMERA_SOURCE = 0

KEY_CODE_0 = 48
KEY_CODE_1 = 49
KEY_CODE_2 = 50
KEY_CODE_3 = 51
KEY_CODE_4 = 52
KEY_CODE_5 = 53
KEY_CODE_6 = 54
KEY_CODE_7 = 55
KEY_CODE_8 = 56
KEY_CODE_9 = 57
KEY_CODE_SPACE = 32
KEY_CODE_TAB = 9
KEY_CODE_ESCAPE = 27

METHOD_ORB = 'orb'
METHOD_SIFT = 'sift'

MARKER_PATH = './markers/marker_hiro.png'
MODEL_PATH = './models/fox.obj'
VIDEO_PATH = './videos/vid2-high.mov'

MODEL_COLOR = (183, 28, 28)

CAMERA_PARAMETERS = np.array([[1000, 0, 640],
                              [0, 1000, 360],
                              [0, 0, 1]])
