import cv2
import numpy as np
from skimage.transform import (hough_line, hough_line_peaks)

img1 = cv2.imread(r'Images/IntersectingLines.jpeg')
img2 = cv2.imread(r'Images/grid.png')
img3 = cv2.imread(r'Images/sudoku.png')
img4 = cv2.imread(r'Images/building.jpeg')


def houghTransform(img):
    # Convert to GrayScale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Edges In Image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Specify Theta Range and number of lines
    tested_angles = np.linspace(-np.pi, np.pi, 200)
    # Apply Hough Transform
    hspace, theta, dist = hough_line(edges, tested_angles)
    # Output Hough Space
    return hspace


def houghLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for i in range(0, len(lines)):
        a = np.cos(lines[i, 0, 1])
        b = np.sin(lines[i, 0, 1])
        x0 = a * lines[i, 0, 0]
        y0 = b * lines[i, 0, 0]
        x1 = int(x0 + img.shape[0] * (-b))
        y1 = int(y0 + img.shape[1] * a)
        x2 = int(x0 - img.shape[0] * (-b))
        y2 = int(y0 - img.shape[1] * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img


HoughTransfrom = houghTransform(img1)
cv2.imwrite('Outputs/HoughTransfrom.jpg', HoughTransfrom)

LinesDetected = houghLines(img2)
cv2.imwrite('Outputs/HoughLines.jpg', LinesDetected)

LinesDetected2 = houghLines(img3)
cv2.imwrite('Outputs/HoughLines2.jpg', LinesDetected2)

