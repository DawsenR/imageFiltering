import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import match_template

cap = cv2.VideoCapture(0)
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
small = mpimg.imread("face3.jpg")
small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
(tH, tW) = small.shape[:2]

while(True):
	# Capture frame-by-frame
	ret, cameraPic = cap.read()
	frame = cv2.cvtColor(cameraPic, cv2.COLOR_BGR2GRAY)
	found = None
#####

	for scale in np.linspace(0.20, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(frame, width = int(frame.shape[1] * scale))
		r = frame.shape[1] / float(resized.shape[1])

		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break
		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		# edged = cv2.Canny(resized, 50, 200)
		res = cv2.matchTemplate(resized,small,cv2.TM_CCOEFF_NORMED)
		threshold = .5
		loc = np.where(res >= threshold)
		for pt in zip(*loc[::-1]):
			print(pt)
			cv2.rectangle(cameraPic,pt,(pt[0]+tW,pt[1]+tH),(0, 255, 0), 2)
			#cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)

		# if we have found a new maximum correlation value, then ipdate
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)

	# unpack the bookkeeping varaible and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))


#####

	cv2.rectangle(cameraPic, (startX, startY), (endX, endY), (0, 0, 255), 2)

	cv2.imshow('frame',cameraPic)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
