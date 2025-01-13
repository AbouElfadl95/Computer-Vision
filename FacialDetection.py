from PIL import Image
import numpy as np
from numpy import save
from numpy import load

girl1 = Image.open(r"Images/f2.png")
girl2 = Image.open(r"Images/f6.jpg")
boy = Image.open(r"Images/f3.jpg")

girl1gray = girl1.convert('L')
girl2gray = girl2.convert('L')
boygray = boy.convert('L')


def IntegralImage(img_arr):
    img_arr = np.array(img_arr)
    img_arr = img_arr.astype(np.uint64)
    output = np.zeros_like(img_arr)
    output = output.astype(np.uint64)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            output[i, j] = (img_arr[0: i + 1, 0: j + 1]).sum()
    return output


def Loc_Sum(img_arr, x, k):
    y = (k[0], x[1])
    z = (x[0], k[1])
    regionA = (img_arr[x[0], x[1]])
    regionB = (img_arr[y[0], y[1]] - regionA)
    regionC = (img_arr[z[0], z[1]] - regionA)
    regionD = (img_arr[k[0], k[1]] - (regionA + regionB + regionC))
    return regionD


def calc_score(int_img_arr, n, m, i, j):
    P1 = (int(-0.5 * n), int(-0.5 * m))
    P2 = (int(-0.05 * n), 0)
    P3 = (int(-0.5 * n), 0)
    P4 = (int(-0.05 * n), int(0.5 * m))
    P5 = (int(0.05 * n), int(-0.5 * m))
    P6 = (int(0.5 * n), 0)
    P7 = (int(0.05 * n), 0)
    P8 = (int(0.5 * n), int(0.5 * m))
    P9 = (int(-0.325 * n), int(0.833 * m))
    P10 = (int(-0.225 * n), int(2 * m))
    P11 = (int(-0.1 * n), int(0.833 * m))
    P12 = (int(0.1 * n), int(2 * m))
    P13 = (int(0.225 * n), int(0.833 * m))
    P14 = (int(0.325 * n), int(2 * m))
    LS1 = (Loc_Sum(int_img_arr, ((i + P1[0]), (j + P1[1])), ((i + P2[0]), (j + P2[1]))))
    LS2 = (Loc_Sum(int_img_arr, ((i + P3[0]), (j + P3[1])), ((i + P4[0]), (j + P4[1]))))
    LS3 = (Loc_Sum(int_img_arr, ((i + P5[0]), (j + P5[1])), ((i + P6[0]), (j + P6[1]))))
    LS4 = (Loc_Sum(int_img_arr, ((i + P7[0]), (j + P7[1])), ((i + P8[0]), (j + P8[1]))))
    LS5 = (Loc_Sum(int_img_arr, ((i + P9[0]), (j + P9[1])), ((i + P10[0]), (j + P10[1]))))
    LS6 = (Loc_Sum(int_img_arr, ((i + P11[0]), (j + P11[1])), ((i + P12[0]), (j + P12[1]))))
    LS7 = (Loc_Sum(int_img_arr, ((i + P13[0]), (j + P13[1])), ((i + P14[0]), (j + P14[1]))))
    result = ((LS1 + LS3 + LS6) - (LS2 + LS4 + LS5 + LS7))
    return result


def detectEye(int_img_arr, width):
    height = int(0.15 * width)
    maxscore = 0
    coordinates = (0, 0)
    for i in range(int_img_arr.shape[0] - int(width / 2)):
        for j in range(int_img_arr.shape[1] - int(width / 3)):
            score = calc_score(int_img_arr, width, height, i, j)
            print(score)
            if score > maxscore:
                maxscore = score
                coordinates = (i, j)
    # print (maxscore)
    return coordinates


def ExtractDetectedEye(img_arr, width, coordinates):
    img_arr = np.array(img_arr)
    output = np.zeros_like(img_arr)
    height = int(0.15 * width)
    (x, y) = coordinates
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if (j >= x + int(-0.5 * width)) and (j <= x + int(0.5 * width)) and (i >= y + int(-0.5 * height)) and (
                    i <= y + int(0.5 * height)):
                output[i, j] = img_arr[i, j]
    return output


# IntegralImagegirl1 = IntegralImage(girl1gray)
# save('Outputs/IntegralImageirl1.npy', IntegralImagegirl1)
#
# IntegralImagegirl2 = IntegralImage(girl2gray)
# save('Outputs/IntegralImagegirl2.npy', IntegralImagegirl2)
#
# IntegralImageboy = IntegralImage(boygray)
# save('Outputs/IntegralImageboy.npy', IntegralImageboy)


IntegralImage = load('Outputs/IntegralImageirl1.npy')
coordinates = detectEye(IntegralImage, 330)
print(coordinates)
result = ExtractDetectedEye(girl1gray, 330, coordinates)
Image.fromarray(result).convert('L').save('Outputs/detectedEye.jpg')

# print(IntegralImage.shape[0])
# 82675623
# 18446744073709551615
# 124918571

