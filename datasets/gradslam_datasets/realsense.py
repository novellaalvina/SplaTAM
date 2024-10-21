import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from natsort import natsorted
import open3d as o3d
from .basedataset import GradSLAMDataset

class RealsenseDataset(GradSLAMDataset):
    """
    Dataset class to process depth images captured by realsense camera on the tabletop manipulator
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # print("input folder", self.input_folder)
        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(os.path.join(self.input_folder, "rgb_*.png")))
        depth_paths = natsorted(glob.glob(os.path.join(self.input_folder, "depth_*.png")))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths
    
    """Helper function to convert a depth image into a point cloud"""
    def depth_to_point_cloud(self,pose, depth_image):
        
        # convert depth image to point cloud using Open3D
        # Assuming you have a known camera instrinsics matrix
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=593.8348999023438, fy=593.8348999023438, cx = 314.661865234375, cy = 242.97659301757812)

        # create point cloud from depth image
        depth_o3d = o3d.geometry.Image(depth_image)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, camera_intrinsics)
        
        return pcd
    
    """ Pose drift or misalignment will cause the 3D Gaussians to be incorrectly placed in the map.

        Use loop closure techniques: If you're seeing increasing errors over time (i.e., pose drift), implementing loop closure detection and pose graph optimization can help reduce accumulated errors.

        Refine tracking with ICP (Iterative Closest Point): For more accurate alignment of frames, you can use ICP to refine the pose estimation, which ensures that splats are fused into the correct position.
    """    
    def ICP_pose(self, poses, depth_paths):

        # Perform ICP to refine poses between consecutive frames
        refined_poses = []

        for i in range(len(poses)-1):
            # get the current pose and the next pose
            current_pose = poses[i]
            next_pose = poses[i+1]

            # convert current and next poses to numpy for ICP processing
            current_pose_np = current_pose.numpy()
            next_pose_np = next_pose.numpy()

            # get the current depth image
            print(depth_paths[i])
            depth_image = o3d.io.read_image(depth_paths[i])
            
            # convert depth image to point clouds for ICP
            pcd_current = self.depth_to_point_cloud(current_pose_np, depth_image)
            pcd_next = self.depth_to_point_cloud(next_pose_np, depth_image)

            # apply ICP to refine pose
            icp_result = o3d.pipelines.registration.registration_icp(
                pcd_current, pcd_next, max_correspondence_distance=0.05, 
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            # get the refined transformation from ICP
            refined_transform = torch.from_numpy(icp_result.transformation).float()
            
            # update the next pose by applying the refined transformation
            refined_pose = next_pose @ refined_transform

            # add the refined pose to the refined poses list
            refined_poses.append(refined_pose)
        
        return refined_poses

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        
        icp = False
        dummy_pose = False
        
        poses = []
        # get the depth file paths
        _, depth_paths, _ = self.get_filepaths()

        P = torch.tensor([[1, 0, 0, 0], 
                          [0, -1, 0, 0], 
                          [0, 0, -1, 0], 
                          [0, 0, 0, 1]]).float()
        if (dummy_pose):
            for pose in range(self.num_imgs):
                poses.append(torch.eye(4).float())
            # for pose in range(self.num_imgs):
            #     poses.append(torch.zeros(4, 4))
        else:
            for posefile in posefiles:
                # print(posefile)
                c2w = torch.from_numpy(np.load(posefile)).float()
                _R = c2w[:3, :3]
                _t = c2w[:3, 3]
                _pose = P @ c2w @ P.T # transforming the pose to the appropriate coordinate system
                poses.append(_pose)

            if (icp):
                poses = self.ICP_pose(poses, depth_paths)

        # print(poses)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)