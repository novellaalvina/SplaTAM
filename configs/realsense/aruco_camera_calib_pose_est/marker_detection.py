import os
import cv2 
import numpy as np
from cv2 import aruco
import pyrealsense2 as rs

# dictionary to specify type of marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# detect the marker
parameters = aruco.DetectorParameters()

# Create directories for saving RGB and depth images
rgb_dir = "../../data/realsense/attempt6/"
depth_dir = "../../data/realsense/attempt6/"
pose_dir = "../../data/realsense/attempt6/poses/"

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

try:
    for i in range(100):
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # turning the frame to grayscale 
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        marker_corners, marker_ids, reject = detector.detectMarkers(gray_frame)

        # getting marker corners
        if marker_corners:
            for ids, corners in zip(marker_ids, marker_corners):
                cv2.polylines(color_image, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)

                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()
                cv2.putText(
                    color_image,
                    f"id: {ids[0]}",
                    top_right,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.3,
                    (200, 100, 0),
                    2,

                    cv2.LINE_AA,
                )
        cv2.imshow("frame", color_image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()