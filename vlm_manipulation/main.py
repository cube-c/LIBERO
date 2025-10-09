
from typing import Literal

import os
import open3d as o3d
import tyro
import torch
import numpy as np
import rootutils
from PIL import Image
from loguru import logger as log
from rich.logging import RichHandler
from scipy.spatial.transform import Rotation as R

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from vlm_manipulation.metasim_utils import ObsSaver, get_pcd_from_rgbd
from vlm_manipulation.curobo_utils import TrajOptimizer
from curobo.types.robot import JointState
from libero.libero import benchmark, get_libero_path, set_libero_default_path

def get_environment(scenario, sim="isaaclab", robot_offset=1.15):
    """Initialize the simulation environment."""
    log.info(f"Using simulator: {sim}")
    env_class = get_sim_env_class(SimType(sim))
    env = env_class(scenario)

    robot = scenario.robots[0]
    init_states, _, _ = get_traj(task, robot, env.handler)
    init_state = init_states[0]

    log.info(
        f"robot position: {init_state['robots'][robot.name]['pos']} | rotation: {init_state['robots'][robot.name]['rot']}"
    )
    # log all objects position
    for obj_name, obj_pos in init_state["objects"].items():
        log.info(f"object {obj_name} position: {obj_pos['pos']} | rotation: {obj_pos['rot']}")

    # translate robot and all objects to make robot at the origin
    for obj_name, obj_pos in init_state["objects"].items():
        init_state["objects"][obj_name]["pos"] -= init_state["robots"][robot.name]["pos"]

    # translate robot to (robot_offset, 0.0, 0.0) and rotate 180 degrees around z axis
    init_state["robots"][robot.name]["pos"] = torch.tensor([robot_offset, 0.0, 0.0])
    init_state["robots"][robot.name]["rot"] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    # translate/rotate all objects relative to robot
    q_z_180 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    q_x_180 = torch.tensor([1.0, 0.0, 0.0, 0.0])

    for obj_name, obj_pos in init_state["objects"].items():
        init_state["objects"][obj_name]["pos"][0] = robot_offset - init_state["objects"][obj_name]["pos"][0] + 0.05
        init_state["objects"][obj_name]["pos"][1] = -init_state["objects"][obj_name]["pos"][1]
        if obj_name == "ketchup" or obj_name == "salad_dressing":
            init_state["objects"][obj_name]["rot"] = torch.tensor(
                (R.from_quat(q_z_180) * R.from_quat(init_state["objects"][obj_name]["rot"])).as_quat()
            )
        if obj_name == "bbq_sauce":
            init_state["objects"][obj_name]["rot"] = torch.tensor(
                (R.from_quat(q_x_180) * R.from_quat(init_state["objects"][obj_name]["rot"])).as_quat()
            )

    log.info(
        f"[After translation] robot position: {init_state['robots'][robot.name]['pos']} | rotation: {init_state['robots'][robot.name]['rot']}"
    )
    for obj_name, obj_pos in init_state["objects"].items():
        log.info(f"[After translation] object {obj_name} position: {obj_pos['pos']} | rotation: {obj_pos['rot']}")

    robot.default_position = torch.tensor([robot_offset, 0.0, 0.0])
    robot.default_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])

    init_state["robots"][robot.name]["dof_pos"]["panda_finger_joint1"] = 0.04
    init_state["robots"][robot.name]["dof_pos"]["panda_finger_joint2"] = 0.04

    env.reset(states=[init_state])
    return env


def get_point_cloud_from_camera(img, depth, camera, output_suffix=""):
    """Get the point cloud from the observation."""
    log.info(f"img shape: {img.shape}, depth shape: {depth.shape}")
    max_depth = np.max(depth[0].cpu().numpy())
    scene_depth = depth / max_depth * 255.0  # normalize depth to [0, 1]
    scene_img = Image.fromarray(img[0].cpu().numpy())
    scene_depth = Image.fromarray((scene_depth[0].squeeze(-1).cpu().numpy() / max_depth * 255.0).astype("uint8"))
    scene_img.save(f"vlm_manipulation/output/img{output_suffix}.png")
    scene_depth.save(f"vlm_manipulation/output/depth{output_suffix}.png")

    extr, intr = get_cam_params(
        cam_pos=torch.tensor([camera.pos]),
        cam_look_at=torch.tensor([camera.look_at]),
        width=camera.width,
        height=camera.height,
        focal_length=camera.focal_length,
        horizontal_aperture=camera.horizontal_aperture,
    )
    pcd = get_pcd_from_rgbd(depth.cpu()[0], img.cpu()[0], intr[0], extr[0])
    pcd.estimate_normals()
    log.info(f"pcd shape: {np.array(pcd.points).shape}")
    return pcd, intr, extr


def get_point_cloud_from_obs(obs):
    # check shapes
    #######################################################################
    # merge pcds
    # before merging, we need to align the two point clouds
    # Align the second viewpoint to the first viewpoint
    # 1. Get the transformation matrix from the second viewpoint to the first viewpoint
    # 2. Apply the transformation matrix to the second viewpoint
    # 3. Merge the two point clouds
    # 4. Save the merged point cloud
    depth = obs.cameras["camera0"].depth
    pcd1, intr, extr = get_point_cloud_from_camera(
        obs.cameras["camera0"].rgb, obs.cameras["camera0"].depth, scenario.cameras[0], output_suffix="1"
    )
    pcd2, _, _ = get_point_cloud_from_camera(
        obs.cameras["camera1"].rgb, obs.cameras["camera1"].depth, scenario.cameras[1], output_suffix="2"
    )

    reg = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, 0.0025, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane()
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
    o3d.io.write_point_cloud("get_started/output/motion_planning/3_object_grasping_vlm/pcd_merged.ply", pcd_merged)

    return pcd_merged, depth[0], intr[0], extr[0]


class MotionController:
    """
    MotionController is used to control the robot from prompt.
    It gets the prompt and return the trajectory.
    It is dependent on the RoboVerse simulator setup.
    """
    def __init__(
        self,
        sim: str,
        env: IdentityEnvWrapper,
        obs_saver: ObsSaver,
        traj_optimizer: TrajOptimizer,
    ):
        self.sim = sim
        self.env = env
        self.obs_saver = obs_saver
        self.robot = env.handler.robots[0]
        self.traj_optimizer = traj_optimizer

    def dummy_action(self, step: int):
        return torch.tensor(list(self.robot.default_joint_positions.values())).repeat(step, 1)

    def wrap_action(self, actions):
        if self.sim == "isaaclab":
            return [{"dof_pos_target": dict(zip(self.robot.actuators.keys(), action.tolist()))} for action in actions]
        elif self.sim == "mujoco":
            return [{self.robot.name: {"dof_pos_target": dict(zip(
                self.robot.actuators.keys(), action.tolist()
            ))}} for action in actions]
        else:
            raise ValueError(f"Invalid simulator: {self.sim}")

    def get_joint_state(self):
        """Get the current joint position."""
        joint_reindex = self.env.handler.get_joint_reindex(self.robot.name)
        joint_pos = self.env.handler.get_states().robots[self.robot.name].joint_pos.cuda()
        js = JointState.from_position(
            position=joint_pos[:, torch.argsort(torch.tensor(joint_reindex))],
            joint_names=list(self.robot.actuators.keys())
        )
        log.info(f"Current robot joint state: {js}")
        return js

    def act(self, actions, save_obs: bool = True):
        """Act the robot and save the observation"""
        actions = self.wrap_action(actions)
        for action in actions:
            # log.info(f"Action: {action}")
            obs, _, _, _, _ = self.env.step([action])
            if save_obs:
                obs = self.env.handler.get_states()
                self.obs_saver.add(obs)
        if save_obs:
            self.obs_saver.save()
        return obs

    def simulate_from_prompt(self, prompt: str):
        """Simulate the robot from prompt."""
        actions = self.dummy_action(120)
        obs = self.act(actions, save_obs=False)

        img = Image.fromarray(obs.cameras["camera0"].rgb[0].cpu().numpy())
        pcd, depth, cam_intr_mat, cam_extr_mat = get_point_cloud_from_obs(obs)

        js = self.get_joint_state()
        actions = self.traj_optimizer.plan_trajectory(js, img, depth, pcd, prompt, cam_intr_mat, cam_extr_mat)
        obs = self.act(actions, save_obs=True)
        return obs



@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = False
    task_name: str = "LiberoPickChocolatePudding"
    prompt: str = "pick up the bbq sauce and place it in the basket"

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # initialize scenario
    task_name = args.task_name
    prompt = args.prompt

    task = get_task(task_name)

    # default
    scenario = ScenarioCfg(
        task=task_name,
        cameras=[
            PinholeCameraCfg(name="camera0", width=1024, height=1024, pos=(0.0, 0.0, 0.8), look_at=(0.9, 0.0, 0.0)),
            PinholeCameraCfg(name="camera1", width=1024, height=1024, pos=(0.9, 1.0, 0.6), look_at=(0.9, 0.0, 0.0)),
        ],
        robots=[args.robot],
        try_add_table=False,
        sim=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )
    robot = scenario.robots[0]
    robot_offset = 1.15

    env = get_environment(scenario, sim=args.sim)

    ## Main loop
    os.makedirs("vlm_manipulation/output", exist_ok=True)
    obs_saver = ObsSaver(video_path=f"vlm_manipulation/output/{task_name}_{args.sim}.mp4")
    traj_optimizer = TrajOptimizer([robot_offset, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    motion_controller = MotionController(args.sim, env, obs_saver, traj_optimizer)
    motion_controller.simulate_from_prompt(prompt)
