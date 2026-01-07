import os
import argparse
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import open3d as o3d
import imageio
import time
from PIL import Image

import numpy as np
import torch
import rootutils
from loguru import logger as log
from rich.logging import RichHandler
from scipy.spatial.transform import Rotation as R
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
camera_names = ["agentview", "robot0_eye_in_hand"]


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


def get_point_cloud_from_obs(obs, zfar, znear, camera_names=camera_names):
    depth = [None, None]
    intr = [None, None]
    extr = [None, None]
    depth[0] = depth_buffer_to_depth_image(
        obs[camera_names[0] + "_depth"], zfar, znear
    )[::-1]
    depth[1] = depth_buffer_to_depth_image(
        obs[camera_names[1] + "_depth"], zfar, znear
    )[::-1]
    img1 = obs[camera_names[0] + "_image"][::-1]
    img2 = obs[camera_names[1] + "_image"][::-1]
    pcd1, intr[0], extr[0] = get_point_cloud_from_camera(
        img1,
        depth[0],
        camera_names[0],
    )
    pcd2, intr[1], extr[1] = get_point_cloud_from_camera(
        img2,
        depth[1],
        camera_names[1],
    )

    reg = o3d.pipelines.registration.registration_icp(
        pcd2,
        pcd1,
        0.0025,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    pcd = [None, None]
    pcd[0] = pcd1
    pcd[1] = pcd2.transform(reg.transformation)

    return pcd, depth, intr, extr


def get_joint_state_from_obs(obs):
    return torch.tensor(
        np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
    )


def step_to_target_pos(env, obs, target_pos):
    diff = (target_pos - get_joint_state_from_obs(obs))[:-1]

    # invert for gripper qpos
    diff[-1] *= -1
    # print(f"diff: {diff}")

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
        self.done = False
        self.traj_optimizer = traj_optimizer
        self.images = []

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

    def dummy_act(self, step: int):
        for _ in range(step):
            obs, _, done, _ = self.env.step([0.0] * 8)
            self.obs = obs
            self.done = self.done or done

    def initial_act(self, step: int):
        for _ in range(step):
            obs, _, done, _ = step_to_target_pos(
                self.env,
                self.obs,
                torch.tensor(
                    [
                        0.0474,
                        -1.2568,
                        -0.0058,
                        -2.2100,
                        -0.0277,
                        1.4764,
                        0.8196,
                        0.0400,
                        0.0400,
                    ]
                ),
            )
            self.obs = obs
            self.done = self.done or done
            # self.images.append(obs["agentview_image"][::-1])

    def act(self, actions):
        for action in actions:
            # log.info(f"Action: {action}")
            obs, _, done, _ = step_to_target_pos(
                self.env, self.obs, action.detach().cpu()
            )
            self.done = self.done or done
            self.obs = obs
            self.images.append(obs["agentview_image"][::-1])

    def get_point_clouds(self):
        # point cloud from agentview
        sim = self.env.env.sim
        extent = float(sim.model.stat.extent)
        zfar = sim.model.vis.map.zfar * extent
        znear = sim.model.vis.map.znear * extent
        pcds, depth, intr, extr = get_point_cloud_from_obs(self.obs, zfar, znear)

        return pcds, depth, intr, extr

    def pcd_to_robot_center(self, pcds: list[o3d.geometry.PointCloud]):
        robot = self.env.env.robots[0]
        sim = self.env.env.sim
        bid = sim.model.body_name2id(robot.robot_model.root_body)
        robot_position = sim.data.body_xpos[bid].copy()
        robot_rotation_matrix = sim.data.body_xmat[bid].reshape(3, 3).copy()

        pcds_transformed = []
        for pcd in pcds:
            pcd = pcd.translate(-robot_position)
            pcd = pcd.rotate(robot_rotation_matrix.T)
            pcds_transformed.append(pcd)
        return pcds_transformed, robot_position, robot_rotation_matrix

    def simulate_from_prompt(self, prompt: str):
        start_time = time.time()

        """Simulate the robot from prompt."""
        self.initial_act(50)

        log.error(f"[Main] Time taken to dummy act: {time.time() - start_time}")

        images = []
        for camera_name in camera_names:
            images.append(Image.fromarray(self.obs[camera_name + "_image"][::-1]))
        pcds, depth, cam_intr_mat, cam_extr_mat = self.get_point_clouds()
        pcds, robot_position, robot_rotation_matrix = self.pcd_to_robot_center(pcds)
        # o3d.io.write_point_cloud("outputs/merged.ply", pcd)

        end_time = time.time()
        log.error(f"[Main] Time taken to get point cloud: {end_time - start_time}")

        # transform camera extrinsic matrix with respect to robot center and rotation matrix
        T_robot = np.eye(4, dtype=np.float64)
        T_robot[:3, :3] = robot_rotation_matrix
        T_robot[:3, 3] = robot_position
        for i in range(len(cam_extr_mat)):
            cam_extr_mat[i] = cam_extr_mat[i] @ T_robot

        js = self.get_joint_state()
        self.traj_optimizer.grasp_finder.set_transform_matrix(
            cam_extr_mat=cam_extr_mat[1]
        )
        try:
            actions = self.traj_optimizer.plan_trajectory(
                js,
                images,
                depth,
                pcds,
                prompt,
                cam_intr_mat,
                cam_extr_mat,
                task_type,
                task_id,
                eval_index,
            )
        except Exception as e:
            log.error(f"Error: {e}")
            self.done = False
            return self.obs, self.done
        # pose_actions = self.traj_optimizer.plan_pose_single(
        # self.traj_optimizer.get_joint_state(self.get_joint_state().position),
        # torch.tensor([0.12, 0.0, 0.75], dtype=torch.float32).to("cuda:0"),
        # torch.tensor([0.0, -0.965926, 0.0, -0.258819], dtype=torch.float32).to(
        # "cuda:0"
        # ),
        # open_gripper=False,
        # )
        # print(pose_actions)

        end_time = time.time()
        log.error(f"[Main] Time taken to plan trajectory: {end_time - start_time}")

        self.act(actions)

        end_time = time.time()
        log.error(f"[Main] Time taken to act: {end_time - start_time}")

        return self.obs, self.done

    def extract_points_only(self, prompt: str, task_type, task_id, eval_index):
        """Extract and visualize points from the prompt without trajectory planning."""
        start_time = time.time()

        # Get images from observation
        self.initial_act(20)
        images = []
        for camera_name in camera_names:
            images.append(Image.fromarray(self.obs[camera_name + "_image"][::-1]))

        # Call point-only extraction
        try:
            self.traj_optimizer.plan_point_only(
                images,
                prompt,
                task_type,
                task_id,
                eval_index,
            )
        except Exception as e:
            log.error(f"[PointOnly] Error during point extraction: {e}")
            log.error(f"[PointOnly] Failed for task {task_id}, eval {eval_index}")
            return

        end_time = time.time()
        log.info(f"[PointOnly] Total extraction time: {end_time - start_time}")

    def extract_sequence_only(self, prompt: str, task_type, task_id, eval_index):
        """Extract and visualize sequence points from the prompt without trajectory planning."""
        start_time = time.time()

        # Get images from observation
        self.initial_act(20)
        images = []
        for camera_name in camera_names:
            images.append(Image.fromarray(self.obs[camera_name + "_image"][::-1]))

        # Call multipoint extraction
        try:
            self.traj_optimizer.plan_multipoint(
                images,
                prompt,
                task_type,
                task_id,
                eval_index,
            )
        except Exception as e:
            log.error(f"[SeqOnly] Error during sequence extraction: {e}")
            log.error(f"[SeqOnly] Failed for task {task_id}, eval {eval_index}")
            return

        end_time = time.time()
        log.info(f"[SeqOnly] Total extraction time: {end_time - start_time}")

    def make_video(self, task_id=None, eval_index=None):
        # make video
        video_writer = imageio.get_writer(
            f"outputs/{task_type}_{task_id if task_id is not None else 'latest'}_{eval_index if eval_index is not None else 'latest'}_{'success' if self.done else 'fail'}.mp4",
            fps=30,
        )
        for image in self.images:
            video_writer.append_data(image)
        video_writer.close()


def modify_sideview_camera(env):
    """Modify the sideview camera position at runtime"""
    sim = env.env.sim

    # manually tuned camera position and rotation (for libero_object)

    if task_type == "libero_object":
        camera_id = sim.model.camera_name2id("sideview")
        sim.model.cam_pos[camera_id] = np.array([0.141838 - 0.6, -0.988357, 0.52037])
        sim.model.cam_quat[camera_id] = np.array(
            [
                0.819536,
                0.507731,
                -0.139905,
                -0.225823,
            ]
        )

        camera_id = sim.model.camera_name2id("agentview")
        sim.model.cam_pos[camera_id] = np.array([0.45, 0.0, 0.75])
        sim.model.cam_quat[camera_id] = np.array(
            [0.683013, 0.183013, 0.183013, 0.683013]
        )

    sim.forward()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Manipulation with LIBERO")
    parser.add_argument(
        "--task_type",
        type=str,
        default="libero_object",
        help="Task type for LIBERO benchmark (e.g., libero_object, libero_spatial, libero_goal)"
    )
    parser.add_argument(
        "--point_only",
        action="store_true",
        help="Extract and visualize points only without trajectory planning (uses extract_point_prediction)"
    )
    parser.add_argument(
        "--seq_only",
        action="store_true",
        help="Extract and visualize sequence points without trajectory planning (uses extract_sequence)"
    )
    args = parser.parse_args()

    task_type = args.task_type
    point_only = args.point_only
    seq_only = args.seq_only

    # Check for mutually exclusive options
    if point_only and seq_only:
        parser.error("--point_only and --seq_only are mutually exclusive")

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[task_type]()
    # traj_optimizer = TrajOptimizer("Qwen/Qwen2.5-VL-7B-Instruct")
    # traj_optimizer = TrajOptimizer("Efficient-Large-Model/NVILA-LITE-2B")
    # traj_optimizer = TrajOptimizer("allenai/Molmo-7B-D-0924")
    traj_optimizer = TrajOptimizer("allenai/Molmo-7B-O-0924")

    for task_id in range(benchmark_instance.get_num_tasks()):
        task = benchmark_instance.get_task(task_id)
        init_states = benchmark_instance.get_task_init_states(task_id)
        env_args = {
            "bddl_file_name": os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            ),
            "camera_names": camera_names,
            "camera_depths": True,
            "camera_heights": H,
            "camera_widths": W,
            "controller": "JOINT_POSITION",
        }

        env = OffScreenRenderEnv(**env_args)
        # Controller output limits
        env.env.robot_configs[0]["controller_config"]["output_min"] = -1.0
        env.env.robot_configs[0]["controller_config"]["output_max"] = 1.0

        print(f"Task Name: {task.name}")
        print(f"Task Description: {task.language}")

        if not point_only and not seq_only:
            success = 0
            total = 0

        for eval_index in range(min(len(init_states), 10)):
            #     Fix random seeds for reproducibility
            env.seed(0)
            env.reset()

            obs = env.set_init_state(init_states[eval_index])
            sim = env.env.sim
            robot = env.env.robots[0]

            # since the sideview camera is too high that the objects are not visible,
            # we need to modify the camera position
            modify_sideview_camera(env)

            mc = MotionController(env, obs, traj_optimizer)

            if point_only:
                # Point-only mode: extract and visualize points without trajectory planning
                mc.extract_points_only(task.language, task_type, task_id, eval_index)
            elif seq_only:
                # Sequence-only mode: extract and visualize sequence points without trajectory planning
                mc.extract_sequence_only(task.language, task_type, task_id, eval_index)
            else:
                # Normal mode: full trajectory planning and execution
                obs, done = mc.simulate_from_prompt(task.language)
                mc.make_video(task_id, eval_index)
                if done:
                    success += 1
                total += 1
                log.info(f"Success Rate for Task {task_id}: {success} / {total}")

        # log to external file (only in normal mode)
        if not point_only and not seq_only:
            with open("outputs/success_rate.txt", "a") as f:
                f.write(f"{task_type} {task_id}: {success} / {total}\n")

        env.close()
