import cv2 as cv
import numpy as np
from cv2 import aruco

# dictionary to specify type of marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

marker_size = 400

# generating unique IDs using the loop
for id in range(20):
    
    # draw marker
    marker_image = aruco.generateImageMarker(aruco_dict, id, marker_size)
    cv.imshow("img", marker_image)
    cv.imwrite(f"markers/marker_{id+1}.png", marker_image)