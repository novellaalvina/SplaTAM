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

# Loop to continuously get frames
try:
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
            # Estimate pose of each marker
            # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            # Draw detected markers and their axes
            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(color_image, corners)
                # cv2.aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

        # Show the image
        cv2.imshow('RealSense', color_image)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()