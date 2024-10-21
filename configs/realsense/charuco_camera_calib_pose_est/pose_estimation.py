import cv2 
from cv2 import aruco
import pyrealsense2 as rs 
import os
import numpy as np
import glob
from natsort import natsorted

# Load ArUco marker dictionaries and params
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# create chArUco board size of width 5 and height 7 and defined parameters
board = aruco.CharucoBoard(size=(6,8), squareLength=0.04, markerLength=0.02, dictionary=aruco_dict)
detector_param = cv2.aruco.DetectorParameters()
charuco_param = aruco.CharucoParameters()
refine_param = aruco.RefineParameters(minRepDistance = 10.0, errorCorrectionRate = 3.0, checkAllOrders = True)
charuco_detector = aruco.CharucoDetector(board=board, charucoParams=charuco_param, detectorParams=detector_param, refineParams=refine_param)

# Create directories for saving RGB and depth images
rgb_dir = "../../data/realsense/attempt6/"
depth_dir = "../../data/realsense/attempt6/"
pose_dir = "../../data/realsense/attempt6/poses/"

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)

# camera matrix and coefficient
camera_matrix = np.array([[576.24724099,   0.0,         271.32821338],
                          [  0.0,         541.60133522, 244.86054452],
                          [  0.0,           0.0,           1.0        ]])
dist_coeff = np.array([[-8.20169047e-02,  1.15370456e+00,  2.39575069e-03, -1.67432023e-02,-2.87339441e+00]])

def detect_pose(image, camera_matrix, dist_coeff):

    # detect markers and charuco corners in the image using charuco detector 
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(image)

    valid_pose = False
    if len(camera_matrix) > 0 and len(dist_coeff) > 0 and len(charuco_ids) >= 4:
        obj_point, img_point = board.matchImagePoints(charuco_corners, charuco_ids)
        valid_pose, rvec, tvec = cv2.solvePnP(obj_point, img_point, camera_matrix, dist_coeff)

        # if valid_pose:
        #     cv2.drawFrameAxes(image, camera_matrix, dist_coeff, rvec, tvec, length=0.1, thickness=15)

    return image, rvec, tvec

# all charuco corners and ids
all_charuco_corners = []       # 2D points in image space
all_charuco_ids = []           # ids of detected charuco corners
all_img_points = []            # detected 2D points
all_obj_points = []            # 3D points in real world space
all_images = []                # store the image for visualization (optional)

# image files
image_files = natsorted(glob.glob(os.path.join("../../data/realsense/attempt6/", "rgb_*.png")))


for img_file in image_files:
    image = cv2.imread(img_file)
    
    # detect pose and draw axis
    pose_image,rvec, tvec = detect_pose(image, camera_matrix, dist_coeff)

    print(f'{img_file}: \n rvec: {rvec}, tvec: {tvec} \ncamera_matrix:{camera_matrix}')

cv2.waitKey(0)
cv2.destroyAllWindows()