
import cv2 as cv
import os
import numpy as np 
import pyrealsense2 as rs

Chess_Board_Dimensions = (9, 6)

n = 0  # image counter

# checks images dir is exist or not
image_dir = "images"

os.makedirs(image_dir, exist_ok=True)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret


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

        copyFrame = color_image.copy()
        gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

        image, board_detected = detect_checker_board(
            color_image, gray, criteria, Chess_Board_Dimensions
        )

        # print(ret)
        cv.putText(
            color_image,
            f"saved_img : {n}",
            (30, 40),
            cv.FONT_HERSHEY_PLAIN,
            1.4,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.imshow("frame", color_image)
        # copyframe; without augmentation
        cv.imshow("copyFrame", copyFrame)

        key = cv.waitKey(1)

        if key == ord("q"):
            break
        # if board_detected:
        # the checker board image gets stored
        cv.imwrite(f"{image_dir}/image{n}.png", copyFrame)

        print(f"saved image number {n}")
        n += 1  # the image counter: incrementing
    
        key = cv.waitKey(1)
        if key == ord("q"):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv.destroyAllWindows()
    print("Total saved Images:", n)