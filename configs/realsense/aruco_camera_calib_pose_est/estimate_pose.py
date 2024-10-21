import json
import cv2 as cv 
import numpy as np 
from cv2 import aruco 
import glob
import os

# print aruco board
ARUCO_DICT = cv.aruco.DICT_6X6_250  # Dictionary ID
SQUARES_VERTICALLY = 7               # Number of squares vertically
SQUARES_HORIZONTALLY = 5             # Number of squares horizontally
SQUARE_LENGTH = 30                   # Square side length (in pixels)
MARKER_LENGTH = 15                   # ArUco marker side length (in pixels)
MARGIN_PX = 20                       # Margins size (in pixels)

IMG_SIZE = tuple(i * SQUARE_LENGTH + 2 * MARGIN_PX for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY))
OUTPUT_NAME = 'ChArUco_Marker.png'

json_file_path = 'camera_calibration.json'

with open(json_file_path, 'r') as file: # Read the JSON file
    json_data = json.load(file)

mtx = np.array(json_data['mtx'])
dst = np.array(json_data['dist'])
# rvecs = np.array(json_data['rvecs'])
# tvecs = np.array(json_data['tvecs'])
objpoints = np.array(json_data['objpoints'])

MARKER_SIZE = 10  # centimeters (measure your printed marker size)

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
param_markers = aruco.DetectorParameters()

img_dir = "images"

def my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv.solvePnP(
            marker_points, c, camera_matrix, distortion, False, cv.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

try:
      
    images = glob.glob(os.path.join(img_dir, '*.png'))
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        h,  w = gray.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
        gray = cv.undistort(gray, mtx, dst, None, newcameramtx)

        #  # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())
    
        aruco_detector = aruco.ArucoDetector(marker_dict, param_markers)
        marker_corners, ids, rejected = aruco_detector.detectMarkers(gray)
        if marker_corners:
            rVec, tVec, _ = my_estimatePoseSingleMarkers(
                marker_corners, MARKER_SIZE, newcameramtx, dst
            )
            print(np.shape(tVec))   ##########Problem###################
            total_markers = range(0, ids.size)
            for ids, corners, i in zip(ids, marker_corners, total_markers):
                cv.polylines(
                    img, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                )
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                # Calculating the distance
                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                # Draw the pose of the marker
                point = cv.drawFrameAxes(img, newcameramtx, dst, rVec[i], tVec[i], 4, 4)

finally:
    # Stop streaming
    cv.destroyAllWindows()