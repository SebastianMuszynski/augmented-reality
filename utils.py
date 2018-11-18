import cv2
import numpy as np


def draw_contours(src,
                  draw_all=True,
                  draw_rectangles=False,
                  draw_circles=False,
                  draw_bounding_boxes=False):

    img_copy = src.copy()
    img_binary = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    result, img_threshold = cv2.threshold(
        src=img_binary,
        thresh=127,
        maxval=255,
        type=cv2.THRESH_BINARY,
    )

    img_contours, contours, hierarchy = cv2.findContours(
        image=img_threshold,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        if draw_rectangles:
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(src, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=1)

        if draw_bounding_boxes:
            rect = cv2.minAreaRect(contour)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(image=src, contours=[box], contourIdx=0, color=(0, 255, 0), thickness=1)

        if draw_circles:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(src, center, radius, color=(0, 0, 255), thickness=1)

    if draw_all:
        cv2.drawContours(src, contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
