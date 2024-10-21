import numpy as np
import cv2 as cv
import glob
import pyrealsense2 as rs 
import json

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
# images = glob.glob('*.jpg')
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

        gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv.drawChessboardCorners(color_image, (7,6), corners2, ret)
            cv.imshow('img', color_image)
            cv.waitKey(500)
    
finally:
    # Stop streaming
    pipeline.stop()
    cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("camera matrix", mtx)
print("dist coeff", dist)

data = {"mtx": mtx.tolist(), "dist": dist.tolist()}

with open("calibration.json", 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f'Data has been saved to calibration.json')

# #  Python code to write the image (OpenCV 3.2)
# fs = cv.FileStorage('calibration.yml', cv.FILE_STORAGE_WRITE)
# fs.write('camera_matrix', mtx)
# fs.write('dist_coeff', dist)
# fs.release()



# If you want to use PyYAML to read and write yaml files,
# try the following part
# It's very important to transform the matrix to list.
# data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

# with open("calibration.yaml", "w") as f:
#    yaml.dump(data, f)

# You can use the following 4 lines of code to load the data in file "calibration.yaml"
# Read YAML file
#with open(calibrationFile, 'r') as stream:
#    dictionary = yaml.safe_load(stream)
#camera_matrix = dictionary.get("camera_matrix")
#dist_coeffs = dictionary.get("dist_coeff")
