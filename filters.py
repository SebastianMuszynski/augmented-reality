import cv2
import numpy as np


class ConvolutionFilter:
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, ddepth=-1, kernel=self._kernel, dst=dst)


class SharpenFilter(ConvolutionFilter):
    """Sharpen filter with 1px radius"""
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

        ConvolutionFilter.__init__(self, kernel)


class EdgesFilter(ConvolutionFilter):
    """Edges filter with 1px radius"""
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])

        ConvolutionFilter.__init__(self, kernel)


class BlurFilter(ConvolutionFilter):
    """Blur filter with 2px radius"""
    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])

        ConvolutionFilter.__init__(self, kernel)


class EmbossFilter(ConvolutionFilter):
    """Emboss filter with 1px radius"""
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [-0,  1, 2]])

        ConvolutionFilter.__init__(self, kernel)


class StrokeEdgesFilter:
    def __init__(self, blur_ksize=7, edge_ksize=5):
        self._blur_ksize = blur_ksize
        self._edge_ksize = edge_ksize

    def apply(self, src, dst):
        if self._blur_ksize >= 3:
            src_blur = cv2.medianBlur(src, self._blur_ksize)
            src_gray = cv2.cvtColor(src_blur, cv2.COLOR_BGR2GRAY)
        else:
            src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        cv2.Laplacian(
            src=src_gray,
            ddepth=cv2.CV_8U,
            dst=src_gray,
            ksize=self._edge_ksize,
        )

        normalised_inverse_alpha = (255 - src_gray) * (1.0 / 255)

        # split an image into channels
        channels = cv2.split(src)

        for channel in channels:
            channel[:] = channel * normalised_inverse_alpha

        # merge channels into an image
        cv2.merge(channels, dst)
