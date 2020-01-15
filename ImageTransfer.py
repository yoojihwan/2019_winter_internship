import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import random
from datetime import datetime

# 이미지의 위치를 shift시키는 함수
def transLocation(img):
    img = cv2.resize(img, dsize=(224, 224))
    h, w = img.shape[:2]

    x = random.randint(-30, 30)
    y = random.randint(-30, 30)

    M = np.float32([[1, 0, x], [0, 1, y]])
    newImg = cv2.warpAffine(img, M, (w, h))

    # print('x = '+ str(x), 'y = ' + str(y))
    return newImg
# 이미지의 각도를 변화시키는 함수
def transAngle(img):
    img = cv2.resize(img, dsize=(224, 224))
    h, w = img.shape[:2]

    d = random.randint(-45, 45)

    M = cv2.getRotationMatrix2D((w/2, h/2), d, 1)
    newImg = cv2.warpAffine(img, M, (w, h))

    # print('d = ' + str(d))
    return newImg

# 이미지의 원근을 변화시키는 함수
def transPerspective(img):
    img = cv2.resize(img, dsize=(224, 224))
    h, w = img.shape[:2]

    L1 = []
    L2 = []
    for i in range(4):
        L1.append(random.randint(10, 40))
        L2.append(random.randint(194, 254))

    pts1 = np.float32([[0, 0], [224, 0], [0, 224], [224, 224]])
    pts2 = np.float32([[L1[0], L1[1]], [L2[0], L1[2]], [L1[3], L2[1]], [L2[2], L2[3]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    newImg = cv2.warpPerspective(img, M, (w, h))

    # print('L1 = ' + str(L1) + ',\t', 'L2 = ' + str(L2))
    return newImg
# 이미지의 밝기를 변화시키는 함수
def adjust_gamma(img):
    g = [0.5, 1.5, 2, 2.5, 3]
    gamma = random.choice(g)

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # print('gamma = ' + str(gamma))
    return cv2.LUT(img, table)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

original = cv2.imread(args["image"])

transImage = transLocation(original)
transImage = transAngle(transImage)
transImage = transPerspective(transImage)
transImage = adjust_gamma(transImage)

# 이미지 저장
cv2.imwrite('data/train/Oface/' + str(datetime.now()) + '.jpg', transImage)
