import pyrealsense2 as rs
import cv2
from cv2 import aruco
import numpy as np

# TO DO #
# print the marker 
# put the marker at a specified location where you want it to be the origin point trajectory of the robot
# then that marker will be detected by the camera
# you can generate the marker from aruco predefined dictionary (aruco_marker.py) or get them online from either of these websites:
    # chev.me/arucogen/
    # fodi.github.io/arucosheetgen/

# Load the ArUco marker dictionary and parameters
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Start the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Color stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start the pipeline
profile = pipeline.start(config)

# get camera intrinsics
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx = intr.fx
fy = intr.fy
cx = intr.ppx
cy = intr.ppy

print("cx is: ",cx)
print("cy is: ",cy)
print("fx is: ",fx)
print("fy is: ",fy)

# Camera intrinsic parameters (you can get these from RealSense calibration or the Realsense API)
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Replace with actual values
dist_coeffs = np.zeros((5, 1))  # Assuming no distortion, or replace with actual
marker_length = 200 # pixels
   
# replacing estimate pose single markers function from aruco since it is deprecated
def my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, camera_matrix, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

# # for transform.json -> pose -> c2w
# transform = {
#     "fl_x": 593.8,
#     "fl_y": 593.8,
#     "cx": 314.7,
#     "cy": 243.0,
#     "w": 640,
#     "h": 480,
#     "camera_model": "OPENCV",
#     "k1": dist_coeffs[0],
#     "k2": dist_coeffs[1],
#     "p1": dist_coeffs[2],
#     "p2": dist_coeffs[3],
#     "k3": dist_coeffs[4],
#     "frames": []
# }

# Loop to continuously get frames
try:
    c = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the image
        aruco_detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = aruco_detector.detectMarkers(gray)

        if len(corners) > 0:
            frame = {}
            # Estimate pose of each marker
            # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            # storing the rotation and translation vectors as transformation matrix of each frame into frames in transform
            frame["file_path"] = "./data/realsense/attempt2"
            
            # this transformation matrix will be the camera pose 
            """ transformation matrix = [
                [x_0, y_0, z_0, x_tr], 
                [x_1, y_1, z_1, y_tr], 
                [x_2, y_2, z_2, z_tr], 
                [0, 0, 0, 1] # constant
            ]"""

            # converting rvecs and tvecs from list to numpy array
            rvecs = np.array(rvecs)
            tvecs = np.array(tvecs)

            # converting the 3x1 rotational vector to 3x3 matrix 
            rodrigues = cv2.Rodrigues(rvecs)
            rvecs_rod = rodrigues[0]
            
            # adding the translation x, y, z to each row of rotational matrix
            tmp = np.concatenate((rvecs_rod, np.reshape(tvecs.T, (3,1))), axis=1)
            constant_matrix = np.reshape(np.array([0, 0, 0, 1]), (1,4))

            # adding the last row of constant vector to make transformation matrix    
            transformation_matrix = np.concatenate((tmp, constant_matrix), axis=0)
            
            # saving the transformation matrix into npy file. 
            transformation = np.save("./data/realsense/attempt2/poses/" + f'pose_{c+1:03}.npy', transformation_matrix)
            
            # Draw detected markers and their axes
            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(color_image, corners)
                # cv2.aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
            
            c +=1
        # Show the image
        cv2.imshow('RealSense', color_image)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()