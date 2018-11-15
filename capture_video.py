import cv2

WEBCAMERA_INDEX = 0
QUIT_SYMBOL = 'q'

videoCapture = cv2.VideoCapture(WEBCAMERA_INDEX)

while videoCapture.isOpened():
    is_frame_captured, frame = videoCapture.read()
    is_quit_key_pressed = cv2.waitKey(1) & 0xFF is ord(QUIT_SYMBOL)

    if not is_frame_captured or is_quit_key_pressed:
        break

    cv2.imshow('AR', frame)

videoCapture.release()
cv2.destroyAllWindows()
