import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import open3d as o3d
import imageio
from PIL import Image

import numpy as np
import torch
import rootutils
from loguru import logger as log
from rich.logging import RichHandler
from collections import OrderedDict
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
)
from vlm_manipulation.curobo_utils import TrajOptimizer
from curobo.types.robot import JointState

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

H, W = 1024, 1024


def depth_buffer_to_depth_image(depth, zfar, znear):
    return znear / (1.0 - depth * (1.0 - znear / zfar))


def get_pcd_from_rgbd(depth, rgb_img, cam_intr_mat, cam_extr_mat):
    """Get the point cloud from the RGBD image."""
    if type(cam_intr_mat) is not np.ndarray:
        cam_intr_mat = np.array(cam_intr_mat)
    if type(cam_extr_mat) is not np.ndarray:
        cam_extr_mat = np.array(cam_extr_mat)

    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_img).astype(np.uint8))
    rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    )

    cam_intr = o3d.camera.PinholeCameraIntrinsic(
        width=depth.shape[1],
        height=depth.shape[0],
        fx=cam_intr_mat[0, 0],
        fy=cam_intr_mat[1, 1],
        cx=cam_intr_mat[0, 2],
        cy=cam_intr_mat[1, 2],
    )
    cam_extr = np.array(cam_extr_mat)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_o3d,
        cam_intr,
        cam_extr,
    )

    return pcd


def get_point_cloud_from_camera(img, depth, camera_name):
    """Get the point cloud from the observation."""
    max_depth = np.max(depth)
    min_depth = np.min(depth)
    scene_depth = (
        (depth - min_depth) / (max_depth - min_depth) * 255.0
    )  # normalize depth to [0, 1]
    scene_img = Image.fromarray(img)
    scene_depth = Image.fromarray(scene_depth[:, :, 0].astype(np.uint8))
    scene_img.save(f"outputs/img_{camera_name}.png")
    scene_depth.save(f"outputs/depth_{camera_name}.png")

    extr = get_camera_extrinsic_matrix(env.sim, camera_name)
    extr = np.linalg.inv(extr)
    intr = get_camera_intrinsic_matrix(env.sim, camera_name, H, W)

    pcd = get_pcd_from_rgbd(depth, img, intr, extr)
    pcd.estimate_normals()
    return pcd, intr, extr


def get_point_cloud_from_obs(obs, zfar, znear, camera_names=["agentview", "sideview"]):
    depth1 = depth_buffer_to_depth_image(obs[camera_names[0] + "_depth"], zfar, znear)[
        ::-1
    ]
    depth2 = depth_buffer_to_depth_image(obs[camera_names[1] + "_depth"], zfar, znear)[
        ::-1
    ]
    img1 = obs[camera_names[0] + "_image"][::-1]
    img2 = obs[camera_names[1] + "_image"][::-1]
    pcd1, intr, extr = get_point_cloud_from_camera(
        img1,
        depth1,
        camera_names[0],
    )
    pcd2, _, _ = get_point_cloud_from_camera(
        img2,
        depth2,
        camera_names[1],
    )

    reg = o3d.pipelines.registration.registration_icp(
        pcd2,
        pcd1,
        0.0025,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    # Convert to numpy arrays
    pts1 = np.asarray(pcd1.points)  # shape (N, 3)
    pts2 = np.asarray(pcd2.points)  # shape (M, 3)

    # Apply transformation to pcd2
    T = reg.transformation  # shape (4, 4)
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))  # shape (M, 4)
    pts2_transformed = (T @ pts2_h.T).T[:, :3]

    pts1 = np.asarray(pcd1.points)
    all_points = np.vstack((pts1, pts2_transformed))

    colors1 = np.asarray(pcd1.colors)
    colors2 = np.asarray(pcd2.colors)
    all_colors = np.vstack((colors1, colors2))

    pcd_merged = o3d.geometry.PointCloud()
    pcd_merged.points = o3d.utility.Vector3dVector(all_points)
    pcd_merged.colors = o3d.utility.Vector3dVector(all_colors)

    return pcd_merged, depth1, intr, extr


def get_joint_state_from_obs(obs):
    return torch.tensor(
        np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
    )


def step_to_target_pos(env, obs, target_pos):
    diff = (target_pos - get_joint_state_from_obs(obs))[:-1]

    # invert for gripper qpos
    diff[-1] *= -1

    return env.step(diff.tolist())


class MotionController:
    """
    MotionController is used to control the robot from prompt.
    It gets the prompt and return the trajectory.
    It is dependent on the RoboVerse simulator setup.
    """

    def __init__(
        self,
        env: OffScreenRenderEnv,
        obs: OrderedDict,
        traj_optimizer: TrajOptimizer,
    ):
        self.env = env
        self.obs = obs
        self.traj_optimizer = traj_optimizer
        self.images = []

    def dummy_action(self, step: int):
        return torch.zeros((step, 9))

    def get_joint_state(self):
        """Get the current joint position."""
        joint_pos = get_joint_state_from_obs(self.obs)
        joint_pos[8] = -joint_pos[8]
        js = JointState.from_position(
            position=joint_pos.unsqueeze(0).to(torch.float32).to("cuda:0"),
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ],
        )
        log.info(f"Current robot joint state: {js}")
        return js

    def act(self, actions, save_obs: bool = True):
        for action in actions:
            # log.info(f"Action: {action}")
            obs, _, _, _ = step_to_target_pos(self.env, self.obs, action.detach().cpu())
            self.obs = obs
            if save_obs:
                self.images.append(obs["agentview_image"][::-1])
        if save_obs:
            self.make_video()

        return obs

    def get_point_cloud(self):
        # point cloud from agentview
        sim = self.env.env.sim
        extent = float(sim.model.stat.extent)
        zfar = sim.model.vis.map.zfar * extent
        znear = sim.model.vis.map.znear * extent
        pcd, depth, intr, extr = get_point_cloud_from_obs(self.obs, zfar, znear)

        return pcd, depth, intr, extr

    def pcd_to_robot_center(self, pcd):
        points = np.asarray(pcd.points)

        robot = self.env.env.robots[0]
        sim = self.env.env.sim
        bid = sim.model.body_name2id(robot.robot_model.root_body)
        robot_position = sim.data.body_xpos[bid].copy()
        robot_rotation_matrix = sim.data.body_xmat[bid].reshape(3, 3).copy()

        points = points - robot_position
        points = points @ robot_rotation_matrix.T
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd, robot_position, robot_rotation_matrix

    def simulate_from_prompt(self, prompt: str):
        """Simulate the robot from prompt."""
        actions = self.dummy_action(10)
        obs = self.act(actions, save_obs=True)

        img = Image.fromarray(obs["agentview_image"][::-1])
        pcd, depth, cam_intr_mat, cam_extr_mat = self.get_point_cloud()
        pcd, robot_position, robot_rotation_matrix = self.pcd_to_robot_center(pcd)
        o3d.io.write_point_cloud("outputs/merged.ply", pcd)

        # transform camera extrinsic matrix with respect to robot center and rotation matrix
        T_robot = np.eye(4, dtype=np.float64)
        T_robot[:3, :3] = robot_rotation_matrix
        T_robot[:3, 3] = robot_position
        cam_extr_mat = cam_extr_mat @ T_robot

        js = self.get_joint_state()
        actions = self.traj_optimizer.plan_trajectory(
            js, img, depth, pcd, prompt, cam_intr_mat, cam_extr_mat
        )
        obs = self.act(actions, save_obs=True)
        return obs

    def make_video(self):
        # make video
        video_writer = imageio.get_writer("outputs/test.mp4", fps=30)
        for image in self.images:
            video_writer.append_data(image)
        video_writer.close()


if __name__ == "__main__":
    benchmark_root_path = get_libero_path("benchmark_root")
    init_states_default_path = get_libero_path("init_states")
    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")
    benchmark_dict = benchmark.get_benchmark_dict()
    print(benchmark_dict)

    # initialize a benchmark
    benchmark_instance = benchmark_dict["libero_10"]()
    num_tasks = benchmark_instance.get_num_tasks()

    # Load torch init files
    init_states = benchmark_instance.get_task_init_states(0)

    # task_id is the (task_id + 1)th task in the benchmark
    task_id = 0
    task = benchmark_instance.get_task(task_id)

    env_args = {
        "bddl_file_name": os.path.join(
            bddl_files_default_path, task.problem_folder, task.bddl_file
        ),
        "camera_names": ["agentview", "sideview"],
        "camera_depths": True,
        "camera_heights": H,
        "camera_widths": W,
        "controller": "JOINT_POSITION",
    }

    env = OffScreenRenderEnv(**env_args)

    # Controller output limits
    env.env.robot_configs[0]["controller_config"]["output_min"] = -1.0
    env.env.robot_configs[0]["controller_config"]["output_max"] = 1.0

    init_states = benchmark_instance.get_task_init_states(task_id)

    # Fix random seeds for reproducibility
    env.seed(0)

    env.reset()

    for eval_index in range(len(init_states)):
        images = []
        obs = env.set_init_state(init_states[eval_index])

        sim = env.env.sim
        robot = env.env.robots[0]

        traj_optimizer = TrajOptimizer()
        mc = MotionController(env, obs, traj_optimizer)

        mc.simulate_from_prompt("pick up the bbq sauce and place it in the basket")

        break

    env.close()
