import cv2
import numpy as np
from matplotlib import pyplot as plt

from helpers import exists
from model_loader import ModelLoader
import math

def draw_contours(src,
                  draw_all=True,
                  draw_rectangles=False,
                  draw_circles=False,
                  draw_bounding_boxes=False):

    img = src.copy()
    img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


def detect_lines(src):
    img = src.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, threshold1=50, threshold2=120)
    lines = cv2.HoughLinesP(
        image=img_edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=20,
        maxLineGap=5,
    )

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(src, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)


def detect_corners(src):
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)
    img_corners = cv2.cornerHarris(src=img_gray, blockSize=2, ksize=23, k=0.04)
    src[img_corners > 0.01 * img_corners.max()] = [0, 0, 255]


def flann(src, camera_parameters, obj):
    img_marker = cv2.imread('marker_hiro.png', cv2.IMREAD_GRAYSCALE)
    img_src = src

    # sift = cv2.xfeatures2d.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(img_marker, None)
    # kp2, des2 = sift.detectAndCompute(img_src, None)
    #
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img_marker, None)
    kp2, des2 = orb.detectAndCompute(img_src, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    # search_params = dict(checks=50)  # or pass empty dictionary

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)

    # # prepare an empty mask to draw good matches
    # matches_mask = [[0, 0] for i in range(len(matches))]
    #
    # # David G. Lowe's ratio test, populate the mask
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.7 * n.distance:
    #         matches_mask[i] = [1, 0]
    #
    # draw_params = dict(matchColor=(0, 255, 0),
    #                   singlePointColor=(255, 0, 0),
    #                   matchesMask=matches_mask,
    #                   flags=0)
    #
    # img_result = cv2.drawMatchesKnn(img_marker, kp1, img_src, kp2, matches, None, **draw_params)

###

    MIN_MATCHES = 10

    # sort them in the order of their distance
    # the lower the distance, the better the match
    matches = sorted(matches, key=lambda x: x.distance)

    # compute Homography if enough matches are found
    if len(matches) > MIN_MATCHES:
        # differenciate between source points and destination points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # if args.rectangle:
        #     # Draw a rectangle that marks the found model in the frame
        #     h, w = model.shape
        #     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #     # project corners into frame
        #     dst = cv2.perspectiveTransform(pts, homography)
        #     # connect them with lines
        #     frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # if a valid homography matrix was found render cube on model plane
        if homography is not None:
            try:
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(camera_parameters, homography)
                # project cube or model
                img_result = render(img_src, obj, projection, img_marker, False)
                #frame = render(frame, model, projection)
            except:
                pass

        # draw first 10 matches.
        # if args.matches:
        img_result = cv2.drawMatches(img_marker, kp1, img_src, kp2, matches[:10], 0, flags=2)

        # show result
        cv2.imshow('matches preview', img_result)

    else:
        print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

###

    # cv2.imshow('matches preview', img_result)


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

