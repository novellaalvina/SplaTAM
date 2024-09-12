import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create directories for saving RGB and depth images
rgb_dir = '../../data/realsense'
depth_dir = '../../data/realsense'

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different formats (resolution, format, frames per second)
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start streaming
pipeline.start(config)

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

        # Apply colormap on depth image (for better visualization)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow('RGB and Depth', images)
       
        # Save the RGB image
        rgb_filename = os.path.join(rgb_dir, f'rgb_{i+1:03}.png')
        cv2.imwrite(rgb_filename, color_image)

        # Save the depth image (as 16-bit PNG)
        depth_filename = os.path.join(depth_dir, f'depth_{i+1:03}.png')
        cv2.imwrite(depth_filename, depth_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
