from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
#from ocr.helpers import implt
import numpy as np
import cv2
import imutils

img = cv2.imread('images/page.jpg')
ratio = img.shape[0] / 500.0
orig = img.copy()
img = imutils.resize(img, height = 500)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

invGamma = 1.0 / 0.3
table = np.array([((i / 255.0) ** invGamma) * 255
for i in np.arange(0, 256)]).astype("uint8")

gray = cv2.LUT(gray, table)

ret,thresh1 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def biggestRectangle(contours):
    biggest = None
    max_area = 0
    indexReturn = -1
    for index in range(len(contours)):
            i = contours[index]
            area = cv2.contourArea(i)
            if area > 100:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.1*peri,True)
                
                if area > max_area: #and len(approx)==4:
                        biggest = approx
                        max_area = area
                        indexReturn = index
    return indexReturn, biggest

indexReturn, biggest = biggestRectangle(contours)

x,y,w,h = cv2.boundingRect(contours[indexReturn])

cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#implt(img, t='Result')

cv2.imwrite('test1.jpg',img)

warped = four_point_transform(orig, biggest.reshape(4, 2) * ratio)

#implt(warped, t='Result')

cv2.imwrite('test.jpg',warped)
