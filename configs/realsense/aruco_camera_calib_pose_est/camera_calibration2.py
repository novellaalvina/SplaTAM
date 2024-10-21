import cv2 
import numpy as np 
import os
import json
import glob
import pyrealsense2 as rs 

# print aruco board
ARUCO_DICT = cv2.aruco.DICT_6X6_250  # Dictionary ID
SQUARES_VERTICALLY = 7               # Number of squares vertically
SQUARES_HORIZONTALLY = 5             # Number of squares horizontally
SQUARE_LENGTH = 30                   # Square side length (in pixels)
MARKER_LENGTH = 15                   # ArUco marker side length (in pixels)
MARGIN_PX = 20                       # Margins size (in pixels)

IMG_SIZE = tuple(i * SQUARE_LENGTH + 2 * MARGIN_PX for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY))
OUTPUT_NAME = 'ChArUco_Marker.png'
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, IMG_SIZE, marginSize=MARGIN_PX)
    cv2.imwrite(OUTPUT_NAME, img)

# create_and_save_new_board()

def get_calibration_parameters(img_dir):
   # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
     
    images = glob.glob(os.path.join(img_dir, '*.png'))
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    # Calibrate camera with extracted information
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
    return mtx, dist, rvecs, tvecs, objpoints

OUTPUT_JSON = 'camera_calibration.json'
mtx, dist, rvecs, tvecs, objpoints = get_calibration_parameters(img_dir='images')

data = {"mtx": mtx.tolist(), "dist": dist.tolist(), 'rvecs':np.asanyarray(rvecs).tolist(), 'tvecs':np.asarray(tvecs).tolist(), 'objpoints': np.asanyarray(objpoints).tolist()}

with open(OUTPUT_JSON, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f'Data has been saved to {OUTPUT_JSON}')

