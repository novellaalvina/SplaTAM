import cv2 
from cv2 import aruco
import pyrealsense2 as rs 
import os
import numpy as np


# Load ArUco marker dictionaries and params
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# create chArUco board size of width 5 and height 7 and defined parameters
board = aruco.CharucoBoard(size=(6,8), squareLength=0.04, markerLength=0.02, dictionary=aruco_dict)
parameters = cv2.aruco.DetectorParameters()
charuco_param = aruco.CharucoParameters()
refine_param = aruco.RefineParameters(minRepDistance = 10.0, errorCorrectionRate = 3.0, checkAllOrders = True)

# charuco detector
charuco_detector = aruco.CharucoDetector(board=board, charucoParams=charuco_param, detectorParams=parameters, refineParams=refine_param)

# Create directories for saving RGB and depth images
rgb_dir = "../../data/realsense/attempt8/"
depth_dir = "../../data/realsense/attempt8/"
pose_dir = "../../data/realsense/attempt8/poses/"

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different formats (resolution, format, frames per second)
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start streaming
profile = pipeline.start(config)

# The more valid captures, the better the calibration
validCaptures = 0

# create filters for depth preprocessing
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
hole_filling_filter = rs.hole_filling_filter()

# all charuco corners and ids
all_charuco_corners = []       # 2D points in image space
all_charuco_ids = []           # ids of detected charuco corners
all_img_points = []            # detected 2D points
all_obj_points = []            # 3D points in real world space
all_images = []                # store the image for visualization (optional)

try:
    for i in range(200):
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        
        # apply filters for depth processing
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filling_filter.process(depth_frame)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (for better visualization)
        depth_colormap = np.asanyarray(rs.colorizer(2).colorize(depth_frame).get_data())

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Save the RGB and depth image (as 16-bit PNG)
        rgb_filename = os.path.join(rgb_dir, f"rgb_{i+1:03}.png")
        depth_filename = os.path.join(depth_dir, f"depth_{i+1:03}.png")
        cv2.imwrite(rgb_filename, color_image)
        cv2.imwrite(depth_filename, depth_image)

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        # detect markers and charuco corners using charuco detector
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        # iterate through all detected charuco corners and ids
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
           
            # 3D object points corresponding to the Charuco board corners
            chessboard_corners = board.getChessboardCorners()
            obj_points = chessboard_corners[charuco_ids.flatten()]

            # Store detected corners and ids for calibration
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            all_obj_points.append(obj_points)
            all_img_points.append(charuco_corners)
            all_images.append(color_image)
        
        # Display the results
        image_copy = color_image.copy()
        if len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)

        if len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids)

        if i <=30 or i >=170:
            print(i)

        cv2.imshow("Charuco Board Detection", image_copy)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()


# Perform camera calibration if enough data has been collected
if len(all_charuco_corners) > 0:
    print("Performing camera calibration...")

    # Calibrate the camera using the Charuco corners
    # camera_matrix = np.zeros((3, 3))
    # dist_coeffs = np.zeros((5, 1))

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=all_obj_points,
        imagePoints=all_img_points,
        imageSize=image_size,
        cameraMatrix=None, 
        distCoeffs=None,
        flags=0
    )

    if ret:
        print(f"Calibration successful. Reprojection error: {ret}")
        print(f"Camera matrix: {camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs}")
        print(f"rotational vector: {rvecs}")
        print(f"translation vector: {tvecs}")

else:
    print("Not enough data for calibration")

# Release the video capture
# inputVideo.release()
# cv2.destroyAllWindows()