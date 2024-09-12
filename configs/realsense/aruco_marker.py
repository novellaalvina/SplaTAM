import cv2
from cv2 import aruco

# dictionary to specify type of the marker (DICT_6X6_250)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# marker_id = 0
marker_size = 200 # pixels

# draw the marker
# generateImageMarker(defined marker dictionary, id you want to set on the marker, size you want for the output marker image, [output image, width of the marker black border])
marker_image = aruco.generateImageMarker(marker_dict, 0, marker_size)
cv2.imshow("traj_marker", marker_image)
cv2.imwrite("traj_marker.png", marker_image)
cv2.waitKey(0)