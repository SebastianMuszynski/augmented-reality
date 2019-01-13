import cv2
import numpy as np
import math

import constants as const
from helpers import exists


class Drawing:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def detect_corners(src):
        img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img_gray = np.float32(img_gray)
        img_corners = cv2.cornerHarris(src=img_gray, blockSize=2, ksize=23, k=0.04)
        src[img_corners > 0.01 * img_corners.max()] = [0, 0, 255]

    @staticmethod
    def get_matches_using_orb(img_src, img_marker):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_marker, None)
        kp2, des2 = orb.detectAndCompute(img_src, None)

        if not exists(des1) or not exists(des2):
            return None, None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = list(filter(lambda m: m.distance < 40, matches))
        return matches, kp1, kp2

    @staticmethod
    def get_matches_using_sift(img_src, img_marker):
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_marker, None)
        kp2, des2 = sift.detectAndCompute(img_src, None)

        if not exists(des1) or not exists(des2):
            return None, None, None

        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        all_matches = flann.knnMatch(des1, des2, k=2)

        matches = []
        for m, n in all_matches:
            if m.distance < 0.75 * n.distance:
                matches.append(m)

        return matches, kp1, kp2

    @staticmethod
    def draw_matches_preview(img_src, img_marker, kp1, kp2, matches):
        img_result = cv2.drawMatches(img_marker, kp1, img_src, kp2, matches[:10], 0, flags=2)
        cv2.imshow('Matches preview', img_result)

    @staticmethod
    def draw_marker_border(img_src, img_marker, homography):
        h, w = img_marker.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, homography)
        cv2.polylines(img_src, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    @staticmethod
    def draw_projection(img_src, img_marker, homography, obj):
        try:
            projection = Drawing.get_projection(homography)
            Drawing.render(img_src, obj, projection, img_marker)
        except:
            pass

    @staticmethod
    def get_src_points(matches, kp1):
        return np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)

    @staticmethod
    def get_dst_points(matches, kp2):
        return np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    @staticmethod
    def find_homography(matches, kp1, kp2):
        homography, mask = cv2.findHomography(
            Drawing.get_src_points(matches, kp1),
            Drawing.get_dst_points(matches, kp2),
            cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        return homography, mask

    @staticmethod
    def get_projection(homography):
        """Calculate 3D projection matrix based on camera params and homography"""
        # Compute rotation (x, y) & translation
        homography = homography * (-1)

        H = np.dot(np.linalg.inv(const.CAMERA_PARAMETERS), homography)
        col_1 = H[:, 0]
        col_2 = H[:, 1]
        col_3 = H[:, 2]

        # Normalise vectors
        alpha = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / alpha
        rot_2 = col_2 / alpha
        translation = col_3 / alpha

        # Orthonormal basis
        a = rot_1 + rot_2
        b = np.cross(rot_1, rot_2)
        c = np.cross(a, b)

        d = a / np.linalg.norm(a, 2)
        e = c / np.linalg.norm(c, 2)
        f = 1 / math.sqrt(2)

        rot_1 = np.dot(d + e, f)
        rot_2 = np.dot(d - e, f)
        rot_3 = np.cross(rot_1, rot_2)

        projection = np.stack((rot_1, rot_2, rot_3, translation)).T
        return np.dot(const.CAMERA_PARAMETERS, projection)

    @staticmethod
    def render(img, obj, projection, model):
        """Render a model into the current frame"""
        vertices = obj.vertices
        scale_matrix = np.eye(3) * 3
        h, w = model.shape

        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
            img_points = np.int32(cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection))
            color = const.MODEL_COLOR[::-1]
            cv2.fillConvexPoly(img, img_points, color)

        return img

    @staticmethod
    def match_and_render(img_src, img_marker, obj, method=const.METHOD_ORB):

        assert method in [const.METHOD_ORB, const.METHOD_SIFT]

        min_matches = {
            const.METHOD_ORB: 20,
            const.METHOD_SIFT: 20,
        }[method]

        matches, kp1, kp2 = {
            const.METHOD_ORB: Drawing.get_matches_using_orb,
            const.METHOD_SIFT: Drawing.get_matches_using_sift,
        }[method](img_src, img_marker)

        if not matches:
            return

        if len(matches) >= min_matches:
            homography, mask = Drawing.find_homography(matches, kp1, kp2)

            if exists(homography):
                # Drawing.draw_marker_border(img_src, img_marker, homography)
                Drawing.draw_projection(img_src, img_marker, homography, obj)

            # Drawing.draw_matches_preview(img_src, img_marker, kp1, kp2, matches)
