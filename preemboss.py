# python3 preemboss.py test.png

import numpy as np
import cv2
import sys

def RBGA2RBG(img):
	height, width, channels = src.shape

	if channels == 3:
		return img

	imgRBG = img[:, :, :3]

	whiteBg = np.zeros((height, width, 3), np.uint8)
	whiteBg[:] = (255, 255, 255)

	imgAlpha = cv2.cvtColor(img[:, :, -1:], cv2.COLOR_GRAY2RGB)

	imgRBG = imgRBG.astype(float)
	whiteBg = whiteBg.astype(float)
	imgAlpha = imgAlpha.astype(float) / 255

	imgRBG = cv2.multiply(imgAlpha, imgRBG)
	whiteBg = cv2.multiply(1.0 - imgAlpha, whiteBg)
	imgRBG = cv2.add(imgRBG, whiteBg)
	
	return imgRBG

def cm2pixels600dpi(cm):
	return int(cm * 0.393701 * 600)

def addPadding(img, size):
	return cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value = [255, 255, 255])

src = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)

dst = RBGA2RBG(src)
dst = addPadding(dst, cm2pixels600dpi(0.5))

cv2.imwrite('preemboss_res.png', dst)















