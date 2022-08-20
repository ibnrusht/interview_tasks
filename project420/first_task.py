import cv2
import numpy

im_path = input("Path to the original image:")
temp_path = input("path to the subimage:")
imgc = cv2.imread(im_path)
tempc = cv2.imread(temp_path)

img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
temp = cv2.cvtColor(tempc, cv2.COLOR_BGR2GRAY)

w, h = temp.shape[::-1]

res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)

threshold = 0.8

loc = numpy.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(imgc, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

cv2.imshow('Detected', imgc)
cv2.waitKey(0)
cv2.destroyAllWindows()

