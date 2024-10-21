import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco
import os
import json

# Load ArUco marker dictionaries and params
def load_aruco():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    detect_params = cv2.aruco.DetectorParameters()

    return aruco_dict, detect_params

def initialize_camera():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different formats (resolution, format, frames per second)
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
    config.enable_stream(rs.stream.accel)   # accelerometer stream
    config.enable_stream(rs.stream.gyro)    # gyrcoscope strem

    # Start streaming
    profile = pipeline.start(config)

    return pipeline, profile

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

def intrinsics(profile, streaming_intrinsic=True):
    if streaming_intrinsic:
        intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        fx = intr.fx
        fy = intr.fy
        cx = intr.ppx
        cy = intr.ppy

        print("cx is: ", cx)
        print("cy is: ", cy)
        print("fx is: ", fx)
        print("fy is: ", fy)

        # Camera intrinsic parameters (you can get these from RealSense calibration or the Realsense API)
        camera_matrix = np.array( [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Replace with actual values
        dist_coeffs = np.zeros((5, 1))  # Assuming no distortion, or replace with actual

        return camera_matrix, dist_coeffs

def rgb_depth_pre_processing(depth_frame, color_frame, i):
    # convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # apply colormap to depth image for better visualization
    depth_colormap = np.asanyarray(rs.colorizer(2).colorize(depth_frame).get_data())

    # stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    # save RGB and depth images
    rgb_filename = os.path.join(rgb_dir, f"rgb_{i+1:03}.png")
    depth_filename = os.path.join(depth_dir, f"depth_{i+1:03}.png")
    cv2.imwrite(rgb_filename, color_image)
    cv2.imwrite(depth_filename, depth_image)

    return color_image, depth_image, images

def generate_pose_matrix(rvecs, tvecs):
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

    return transformation_matrix
    
# replacing estimate pose single markers function from aruco since it is deprecated
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
        nada, R, t = cv2.solvePnP(
            marker_points, c, camera_matrix, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

# Create directories for saving RGB and depth images
rgb_dir = "../../data/realsense/attempt12/"
depth_dir = "../../data/realsense/attempt12/"
pose_dir = "../../data/realsense/attempt12/poses/"

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)

marker_length = 200  # pixels

# load aruco marker and parameters
aruco_dict, detect_params = load_aruco()

# initialize camera streaming
pipeline, profile = initialize_camera()

# get camera intrinsics (camera matrix and distance coefficients)
camera_matrix, dist_coeffs = intrinsics(profile)

try:
    for i in range(100):

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if accel_frame and gyro_frame:
            accel_data2 = accel_frame.as_motion_frame().get_motion_data()
            gyro_data2 = gyro_frame.as_motion_frame().get_motion_data()

        if not depth_frame or not color_frame:
            continue
        
        # rgb and depth image pre processing and save the images
        color_image, depth_image, images = rgb_depth_pre_processing(depth_frame, color_frame, i)

        # convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # detect aruco markers in the image
        aruco_detector = aruco.ArucoDetector(aruco_dict, detect_params)
        marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(gray)

        if len(marker_corners) > 0:

            # estimate pose of each marker
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(marker_corners, marker_length, camera_matrix, dist_coeffs)

            # draw the detected markers and their axes    
            for j in range(len(marker_ids)):
                cv2.aruco.drawDetectedMarkers(color_image, marker_corners)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[j], tvecs[j], 0.1)
            
            # this transformation matrix will be the camera pose
            """ transformation matrix = [
                [x_0, y_0, z_0, x_tr], 
                [x_1, y_1, z_1, y_tr], 
                [x_2, y_2, z_2, z_tr], 
                [0, 0, 0, 1] # constant
            ]"""

            # get transformation matrix from rvecs and tvecs
            transformation_matrix = generate_pose_matrix(rvecs, tvecs)
            
            # saving the transformation matrix into npy file.
            pose_filename = os.path.join(pose_dir, f"pose_{i+1:03}.npy")
            np.save(pose_filename, transformation_matrix)

            # Show images
            cv2.imshow("RGB and Depth", images) 

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()