import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import imageio
from PIL import Image

import numpy as np
import torch
import rootutils
from loguru import logger as log
from rich.logging import RichHandler
from vlm_manipulation.camera_utils import get_cam_params, get_pcd_from_rgbd

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def get_point_cloud_from_camera(img, depth, camera, output_suffix=""):
    """Get the point cloud from the observation."""
    log.info(f"img shape: {img.shape}, depth shape: {depth.shape}")
    max_depth = np.max(depth[0].cpu().numpy())
    scene_depth = depth / max_depth * 255.0  # normalize depth to [0, 1]
    scene_img = Image.fromarray(img[0].cpu().numpy())
    scene_depth = Image.fromarray(
        (scene_depth[0].squeeze(-1).cpu().numpy() / max_depth * 255.0).astype("uint8")
    )
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


def get_joint_state_from_obs(obs):
    return torch.tensor(
        np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
    )


def step_to_target_pos(env, obs, target_pos):
    diff = (target_pos - get_joint_state_from_obs(obs))[:-1]

    # invert for gripper qpos
    diff[-1] *= -1

    return env.step(diff.tolist())


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
        "camera_heights": 1024,
        "camera_widths": 1024,
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

    # Initial pose by franka cfg (see RoboVerse/roboverse_pack/robots/franka_cfg.py)
    target_pos = torch.tensor(
        [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398, 0.04, 0.04]
    )

    for eval_index in range(len(init_states)):
        images = []
        env.set_init_state(init_states[eval_index])

        # wait until the objects are stable
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 8)

        for key in obs:
            print(f"{key}: {obs[key]}")

        # action
        for _ in range(200):
            obs, _, _, _ = step_to_target_pos(env, obs, target_pos)
            # images.append(obs["agentview_image"])
            images.append(obs["sideview_image"])

        # make video
        video_writer = imageio.get_writer("test.mp4", fps=30)
        for image in images:
            video_writer.append_data(image[::-1])
        video_writer.close()
        break

    env.close()
