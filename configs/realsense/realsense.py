import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco
import os

# Create directories for saving RGB and depth images
rgb_dir = "../../data/realsense/attempt2/"
depth_dir = "../../data/realsense/attempt2/"
pose_dir = "../../data/realsense/attempt2/poses/"

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# Load ArUco marker dictionaries and params
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different formats (resolution, format, frames per second)
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start streaming
profile = pipeline.start(config)

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
marker_length = 200  # pixels


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

# create filters for depth preprocessing
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
hole_filling_filter = rs.hole_filling_filter()

try:
    for i in range(100):
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
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = np.asanyarray(rs.colorizer(2).colorize(depth_frame).get_data())
        print("depth image", depth_image)
        print("depth colormap", depth_colormap)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Save the RGB image
        rgb_filename = os.path.join(rgb_dir, f"rgb_{i+1:03}.png")
        cv2.imwrite(rgb_filename, color_image)

        # Save the depth image (as 16-bit PNG)
        depth_filename = os.path.join(depth_dir, f"depth_{i+1:03}.png")
        cv2.imwrite(depth_filename, depth_image)

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the image
        aruco_detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = aruco_detector.detectMarkers(gray)
        print(i + 1, "'s corners length", len(corners))

        if len(corners) > 0:
            # Estimate pose of each marker
            # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )
            # cv2.polylines(
            #         color_frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
            #     )

            # draw detected markers and their axes
            for i in range(len(ids)):
                print(corners)
                cv2.aruco.drawDetectedMarkers(color_image, corners)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

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
            pose_filename = os.path.join(pose_dir, f"pose_{i+1:03}.npy")

            np.save(pose_filename, transformation_matrix)

            # Show images
            cv2.imshow("RGB and Depth", images) 

            # cv2.imshow("realsense color image", color_image)
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()


# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import os

# # Create directories for saving RGB, depth images, and trajectory
# rgb_dir = '../data/realsense/attempt1/results'
# depth_dir = '../data/realsense/attempt1/results'
# trajectory_file = '../data/realsense/attempt1/traj.txt'

# os.makedirs(rgb_dir, exist_ok=True)
# os.makedirs(depth_dir, exist_ok=True)

# # Configure depth, color, and pose streams
# pipeline = rs.pipeline()
# config = rs.config()

# # Start streaming from depth, color, and pose
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.pose)  # Enable pose stream

# # Start the pipeline
# pipeline.start(config)

# # Initialize frame count
# frame_count = 0
# max_frames = 100  # Total frames to capture

# # Open a file to save the trajectory data
# with open(trajectory_file, 'w') as traj_file:
#     traj_file.write('frame_id, x, y, z, qw, qx, qy, qz\n')  # Header for the trajectory file

#     try:
#         while frame_count < max_frames:
#             # Wait for a coherent pair of frames: depth, color, and pose
#             frames = pipeline.wait_for_frames()
#             depth_frame = frames.get_depth_frame()
#             color_frame = frames.get_color_frame()
#             pose_frame = frames.get_pose_frame()  # Pose frame

#             if not depth_frame or not color_frame or not pose_frame:
#                 continue

#             # Convert images to numpy arrays
#             depth_image = np.asanyarray(depth_frame.get_data())
#             color_image = np.asanyarray(color_frame.get_data())

#             # Save the RGB image
#             rgb_filename = os.path.join(rgb_dir, f'rgb_{frame_count:03}.png')
#             cv2.imwrite(rgb_filename, color_image)

#             # Save the depth image (as 16-bit PNG)
#             depth_filename = os.path.join(depth_dir, f'depth_{frame_count:03}.png')
#             cv2.imwrite(depth_filename, depth_image)

#             # Get the pose data (position and orientation)
#             pose_data = pose_frame.get_pose_data()

#             # Extract position and orientation (quaternion)
#             translation = pose_data.translation
#             rotation = pose_data.rotation
#             position = (translation.x, translation.y, translation.z)
#             orientation = (rotation.w, rotation.x, rotation.y, rotation.z)

#             # Write the pose data to the trajectory file
#             traj_file.write(f'{frame_count}, {position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}, '
#                             f'{orientation[0]:.6f}, {orientation[1]:.6f}, {orientation[2]:.6f}, {orientation[3]:.6f}\n')

#             print(f'Saved frame {frame_count} and trajectory.')

#             # Increment frame count
#             frame_count += 1

#     finally:
#         # Stop streaming
#         pipeline.stop()

# print('Finished capturing 100 frames and trajectory.')
