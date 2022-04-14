import cv2
import numpy as np
import imutils
import easyocr

image = cv2.imread('dataset/5.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY", gray)

bilateralFilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bilateralFilter, 30, 200)
cv2.imshow("EDGE", edged)

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None

for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)

    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
masked_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("MASKED", masked_image)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2, y1:y2]

cv2.imshow("CROPPED", cropped_image)

cv2.imwrite(f'results/plate.jpg', cropped_image)

cv2.waitKey(0)
