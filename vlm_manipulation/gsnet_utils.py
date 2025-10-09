"""This file is used to grasp an object from a point cloud."""

import os
import time

import numpy as np
import open3d as o3d
import rootutils
import torch

# from graspnetAPI.graspnet_eval import GraspGroup
from graspnetAPI import Grasp, GraspGroup

rootutils.setup_root(__file__, pythonpath=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from third_party.gsnet.dataset.graspnet_dataset import minkowski_collate_fn
from third_party.gsnet.models.graspnet import GraspNet, pred_decode
from third_party.gsnet.utils.collision_detector import ModelFreeCollisionDetector


class GSNet:
    """This class is used to grasp an object from a point cloud."""

    def __init__(self):
        """This function is used to initialize the configuration."""
        self.checkpoint_path = "third_party/gsnet/assets/minkuresunet_realsense_tune_epoch20.tar"
        self.seed_feat_dim = 512
        self.num_point = 15000
        self.batch_size = 1
        self.voxel_size = 0.005
        self.collision_thresh = 0.01
        self.voxel_size_cd = 0.01
        self.infer = False
        self.vis = False
        self.dump_dir = "logs"
        self.camera = "realsense"
        self.scene = "0188"
        self.index = "0000"

    def inference(self, cloud_masked, max_grasps=200):
        """This function is used to infer the grasp from the point cloud."""
        # sample points random
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
            # print("sampled point cloud idxs:", idxs.shape)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        data_dict = {
            "point_clouds": cloud_sampled.astype(np.float32),
            "coors": cloud_sampled.astype(np.float32) / self.voxel_size,
            "feats": np.ones_like(cloud_sampled).astype(np.float32),
        }

        batch_data = minkowski_collate_fn([data_dict])
        net = GraspNet(seed_feat_dim=self.seed_feat_dim, is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        net.load_state_dict(checkpoint["model_state_dict"])
        # start_epoch = checkpoint["epoch"]
        # print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

        net.eval()
        tic = time.time()

        for key in batch_data:
            if "list" in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            if end_points is None:
                return None
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        # collision detection
        if self.collision_thresh > 0:
            cloud = data_dict["point_clouds"]

            # Model-free collision detector
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size_cd)
            collision_mask_mfc = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
            gg = gg[~collision_mask_mfc]

        gg = gg.nms()
        gg = gg.sort_by_score()

        if gg.__len__() > max_grasps:
            gg = gg[:max_grasps]

        return gg

    def visualize(self, cloud, gg: GraspGroup = None, g: Grasp = None, image_only=False, filename=None, save_dir=""):
        """This function is used to visualize the grasp group or grasp."""
        pcd = cloud
        if image_only:
            # save image
            points = np.asarray(pcd.points)

            # along to x-axis
            rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
            rotation_tilt = np.array([[1, 0, 0], [0, np.cos(70), -np.sin(70)], [0, np.sin(70), np.cos(70)]])

            # along to -x-axis
            # rotation = np.array([
            #     [0, 0, -1],
            #     [1, 0, 0],
            #     [0, 1, 0]
            # ])
            # # Optional: tilt the view by 70 degrees around the x axis
            # theta = np.deg2rad(70)
            # rotation_tilt = np.array([
            #     [1, 0, 0],
            #     [0, np.cos(theta), -np.sin(theta)],
            #     [0, np.sin(theta), np.cos(theta)]
            # ])

            # along to y-axis
            # rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
            # # Optional: tilt the view by 70 degrees around the y axis
            # theta = np.deg2rad(70)
            # rotation_tilt = np.array([
            #     [np.cos(theta), 0, np.sin(theta)],
            #     [0, 1, 0],
            #     [-np.sin(theta), 0, np.cos(theta)]
            # ])

            # along to z-axis
            # rotation = np.eye(3)
            # # Optional: tilt the view by 70 degrees around the x axis
            # theta = np.deg2rad(70)
            # rotation_tilt = np.array([
            #     [1, 0, 0],
            #     [0, np.cos(theta), -np.sin(theta)],
            #     [0, np.sin(theta), np.cos(theta)]
            # ])

            # along to -z-axis
            # rotation = np.array([
            #     [1, 0, 0],
            #     [0, 1, 0],
            #     [0, 0, -1]
            # ])
            # # Optional: tilt the view by 70 degrees around the x axis
            # theta = np.deg2rad(70)
            # rotation_tilt = np.array([
            #     [1, 0, 0],
            #     [0, np.cos(theta), -np.sin(theta)],
            #     [0, np.sin(theta), np.cos(theta)]
            # ])

            rotation = rotation_tilt @ rotation
            points = points @ rotation.T
            pcd.points = o3d.utility.Vector3dVector(points)

            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            # downpcd = pcd.voxel_down_sample(voxel_size=0.01)
            vis.add_geometry(pcd)
            grippers = gg.to_open3d_geometry_list()
            # Add each gripper individually
            for i, gripper in enumerate(grippers):
                # if i == 0:  # Only transform the first gripper for visualization
                vertices = np.asarray(gripper.vertices)
                vertices = vertices @ rotation.T
                gripper.vertices = o3d.utility.Vector3dVector(vertices)
                vis.add_geometry(gripper)

            # vertices = np.asarray(grippers[0].vertices)
            # vertices = vertices @ rotation.T
            # grippers[0].vertices = o3d.utility.Vector3dVector(vertices)
            # vis.add_geometry(*grippers)
            vis.poll_events()
            vis.update_renderer()

            image = vis.capture_screen_float_buffer()
            import imageio

            image = np.asarray(image)
            imageio.imwrite(
                f"get_started/output/motion_planning/{save_dir}/{filename}_gsnet_visualization.png", (image * 255).astype(np.uint8)
            )
            vis.destroy_window()
            return
        if gg is not None:
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([pcd, *grippers])
        elif g is not None:
            gripper = g.to_open3d_geometry()
            o3d.visualization.draw_geometries([pcd, gripper])
        else:
            o3d.visualization.draw_geometries([pcd])
