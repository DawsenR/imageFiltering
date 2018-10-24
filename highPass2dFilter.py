import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
h = [1,-1]
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rgb2gray(frame)


    result_array = np.empty((0,frame.shape[1]+1))
    for line in frame:
        result = np.convolve(line,h)
        result_array = np.append(result_array,[result],axis = 0)
    np.transpose(result_array)
    frame = result_array
    result_array = np.empty((0,frame.shape[1]+1))
    for line in frame:
        result = np.convolve(line,h)
        result_array = np.append(result_array,[result],axis = 0)
    frame = result_array
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
