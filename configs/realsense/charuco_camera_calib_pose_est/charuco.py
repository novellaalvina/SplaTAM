import cv2 
from cv2 import aruco

# aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# create chArUco board size of width 5 and height 7
board = aruco.CharucoBoard(size=(6,8), squareLength=0.04, markerLength=0.02, dictionary=aruco_dict)

# create an image from the board
img = board.generateImage(outSize=(988, 1400))
cv2.imwrite('charuco_board.png', img)

# display the image
cv2.imshow('charuco board', img)

# exit
cv2.waitKey(0)
cv2.destroyAllWindows()