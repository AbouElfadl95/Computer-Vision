import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Images/StopSign.jpeg', 0)
template = cv.imread('Images/Template1.jpg', 0)

img2 = cv.imread('Images/Cents.jpeg', 0)
template2 = cv.imread('Images/OneCent.jpg', 0)

img3 = cv.imread('Images/ManyStopsSigns.jpeg', 0)
template3 = cv.imread('Images/Template2.jpg', 0)


def CrossCorelate(img, template):
    w, h = template.shape[::-1]
    # Using Template Matching To crosscorelate
    res = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    #Drawing a rectangle
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('cv.TM_CCORR_NORMED')
    plt.savefig('Outputs/Template_Detected.jpg')


def multiObj(img, template, thresh):
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= thresh)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv.imwrite('Outputs/CCorelation_MultiObj.png', img)


CrossCorelate(img, template)
multiObj(img2, template2, 0.7)






