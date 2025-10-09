"""This script is used to grasp an object from a point cloud."""

from __future__ import annotations

"""This script is used to test the static scene."""


import copy

import numpy as np
import rootutils
import torch
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import json
import re
from typing import Any, List

import open3d as o3d
import rootutils
from loguru import logger as log
from rich.logging import RichHandler
from scipy.spatial.transform import Rotation as R
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenPlanConfig, MotionGenConfig
from vlm_manipulation.gsnet_utils import GSNet

class VLMPointExtractor:
    def __init__(self, ckpt_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.ckpt_path = ckpt_path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(ckpt_path)

    def inference(self, img, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            images=img,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    def extract_points(self, img, object_name):
        prompt = f"{object_name}\nOutput its coordinates in XML format <points x y>object</points>."
        output_text = self.inference(img, prompt)
        point = self._extract_points_from_text(output_text, img.width, img.height)
        log.info(f"Qwen2.5-VL point: {point}")
        return point

    def extract_sequence(self, img, prompt):
        # example prompt from LIBERO
        prompt_suffix = 'Report action sequence of the robot in JSON format like this: \
        [{"pick_up": [x, y], "put_down": [x, y]}, ...]'
        prompt = prompt + "\n" + prompt_suffix
        output_text = self.inference(img, prompt)
        log.info(f"Qwen2.5-VL action sequence: {output_text}")
        seq = self._extract_sequence(output_text)
        return seq

    def _extract_sequence(self, text):
        def _is_pick_put_obj(obj: Any) -> bool:
            if not isinstance(obj, dict):
                return False
            if not ("pick_up" in obj and "put_down" in obj):
                return False

            def ok(v):
                return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(n, (int, float)) for n in v)

            return ok(obj["pick_up"]) and ok(obj["put_down"])

        def _normalize_pair(v):  # ints are usually what you want for pixels
            x, y = v
            return [int(x), int(y)]

        def flatten(obj: Any):
            out = []
            if _is_pick_put_obj(obj):
                out.append({
                    "pick_up": _normalize_pair(obj["pick_up"]),
                    "put_down": _normalize_pair(obj["put_down"]),
                })
            elif isinstance(obj, list):
                for it in obj:
                    if _is_pick_put_obj(it):
                        out.append({
                            "pick_up": _normalize_pair(it["pick_up"]),
                            "put_down": _normalize_pair(it["put_down"]),
                        })
            return out

        def code_fences(s: str) -> List[str]:
            pat = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
            return [m.group(1) for m in pat.finditer(s)]

        def balanced_json_slices(s: str) -> List[str]:
            # find balanced {...} or [...] substrings, skipping over quoted strings
            out, n, i = [], len(s), 0
            while i < n:
                if s[i] in "{[":
                    stack = ["}" if s[i] == "{" else "]"]
                    j = i + 1
                    while j < n and stack:
                        c = s[j]
                        if c == '"':  # skip strings (with escapes)
                            j += 1
                            while j < n:
                                if s[j] == "\\":
                                    j += 2
                                elif s[j] == '"':
                                    j += 1
                                    break
                                else:
                                    j += 1
                            continue
                        if c in "{[":
                            stack.append("}" if c == "{" else "]")
                        elif c in "}]":
                            if not stack or c != stack[-1]:
                                stack = []  # mismatch -> abort this slice
                                break
                            stack.pop()
                        j += 1
                    if not stack:  # found a balanced slice
                        out.append(s[i:j])
                        i = j
                        continue
                i += 1
            return out

        candidates = code_fences(text) + balanced_json_slices(text)
        seen, results = set(), []
        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                # minor cleanup: remove trailing commas
                cand2 = re.sub(r",\s*([}\]])", r"\1", cand)
                try:
                    obj = json.loads(cand2)
                except json.JSONDecodeError:
                    continue
            for item in flatten(obj):
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    results.append(item)
        return results

    def _extract_points_from_text(self, text, image_w, image_h):
        all_points = []
        for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)

        for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                # if np.max(point) > 100:
                #     # Treat as an invalid output
                #     continue
                # point /= 100.0
                # point = point * np.array([image_w, image_h])
                all_points.append(point)
        for match in re.finditer(r"(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})", text):
            try:
                point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        return all_points


class GraspPoseFinder:
    def __init__(self):
        self.gsnet = GSNet()

        # for axis conversion between GSNet and isaaclab
        # TODO: check if this is necessary
        self.transform_matrix = np.diag([1, -1, -1])

    def find(self, pcd: o3d.geometry.PointCloud):
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        log.info(f"Point cloud shape: {points.shape}")
        log.info(
            f"Point cloud bounds: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
            f"Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
            f"Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]"
        )

        # do not consider the arm points
        # point_mask = points[:,2] <= 0.1
        point_mask = np.ones(points.shape[0], dtype=bool)
        point_mask = np.logical_and(
            points[:, 2] <= 0.2,
            np.logical_and(points[:, 0] <= 1.0, np.logical_and(points[:, 1] >= -1.0, points[:, 1] <= 1.0)),
        )

        # point_mask = np.logical_and(points[:,2] <= 0.1 , points[:,0] >= 0.1)
        points_masked = points[point_mask]
        colors_masked = colors[point_mask]
        log.info(
            f"New Point cloud bounds: X[{points_masked[:, 0].min():.3f}, {points_masked[:, 0].max():.3f}], "
            f"Y[{points_masked[:, 1].min():.3f}, {points_masked[:, 1].max():.3f}], "
            f"Z[{points_masked[:, 2].min():.3f}, {points_masked[:, 2].max():.3f}]"
        )
        pcd.points = o3d.utility.Vector3dVector(points_masked)
        pcd.colors = o3d.utility.Vector3dVector(colors_masked)

        grasps = self.gsnet.inference(np.array(pcd.points) @ self.transform_matrix)
        grasps.translations = grasps.translations @ self.transform_matrix.T
        grasps.rotation_matrices = self.transform_matrix @ grasps.rotation_matrices @ self.transform_matrix.T

        # Filter out grasps that has width larger than 0.08 (franka finger width)
        grasps = grasps[grasps.widths <= 0.08]

        # log best grasp candidate
        log.info(f"Total grasp candidates: {len(grasps)}")
        log.info(f"Best grasp candidate: {grasps[0].translation}")
        log.info(f"Best grasp candidate rotation: {grasps[0].rotation_matrix}")

        return grasps

    def visualize(self, pcd: o3d.geometry.PointCloud, grasps, image_only=False, save_dir="", filename=""):
        pcd_clone = copy.deepcopy(pcd)
        grasps_clone = copy.deepcopy(grasps)
        pcd_clone.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) @ self.transform_matrix)
        grasps_clone.translations = grasps_clone.translations @ self.transform_matrix.T
        grasps_clone.rotation_matrices = (
            self.transform_matrix @ grasps_clone.rotation_matrices @ self.transform_matrix.T
        )
        self.gsnet.visualize(pcd_clone, grasps_clone, image_only=image_only, save_dir=save_dir, filename=filename)


class TrajOptimizer:
    """
    TrajOptimizer is used to optimize the trajectory of the robot.
    It gets the point cloud, robot pose, robot config, prompt and return the optimized trajectory.
    It must not be dependent on the RoboVerse simulator setup. (only curobo is used)

    Args:
        pcd: Point cloud
        camera: Camera
        image: RGB Image rendered from camera
        robot_pose: Robot pose
        robot_config: Curobo robot config
    """
    def __init__(self, robot_pose: List[float]):
        self.point_extractor = VLMPointExtractor()
        self.grasp_finder = GraspPoseFinder()
        self.robot_position = np.array(robot_pose[:3])
        self.robot_orientation = np.array(robot_pose[3:]) # quaternion
        self.robot_rotation_matrix = R.from_quat(self.robot_orientation).as_matrix()

        self.robot_gripper_open_q = [0.04, 0.04]
        self.robot_gripper_close_q = [0.00, 0.00]
        self.robot_tcp_rel_pos = [0.0, 0.0, 0.10312]
        self.curobo_n_dof = 7
        self.ee_n_dof = 2

        tensor_args = TensorDeviceType()
        self.config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]
        # config_file = load_yaml(join_path(get_robot_path(), robot_cfg.curobo_ref_cfg_name))["robot_cfg"]
        self.robot_config = RobotConfig.from_dict(self.config_file, tensor_args)
        self.kin_model = CudaRobotModel(self.robot_config.kinematics)
        log.info(f"Joint names: {self.kin_model.joint_names}")


    def _get_3d_point_from_pixel(self, pixel_point, depth, cam_intr_mat, cam_extr_mat):
        """
        Convert 2D pixel coordinates to 3D points using the point cloud.

        Since the point cloud was generated from RGBD using get_pcd_from_rgbd(),
        there's a direct mapping between pixels and 3D points.

        Args:
            pixel_points: List of 2D pixel coordinates [(x, y), ...]
            pcd_array: numpy array of point cloud
            camera_intrinsics: Camera intrinsic matrix (optional)
            camera_extrinsics: Camera extrinsic matrix (optional)

        Returns:
            List of 3D points [(x, y, z), ...]
        """
        x, y = int(pixel_point[0]), int(pixel_point[1])
        z = depth[y, x].item()
        log.info(f"depth: {z}")

        # 2. Unproject (x, y, z) into camera coordinates
        fx = cam_intr_mat[0, 0]
        fy = cam_intr_mat[1, 1]
        cx = cam_intr_mat[0, 2]
        cy = cam_intr_mat[1, 2]
        log.info(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

        x_cam = (x - cx) * z / fx
        y_cam = (y - cy) * z / fy
        z_cam = z
        log.info(f"x_cam: {x_cam}, y_cam: {y_cam}, z_cam: {z_cam}")

        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])  # homogeneous coordinates

        # 3. Transform to world coordinates using extrinsic matrix
        cam_extr = np.array(cam_extr_mat)  # should be a 4x4 matrix
        cam_extr = np.linalg.inv(cam_extr)
        log.info(f"cam_extr: {cam_extr}")
        point_world = cam_extr @ point_cam

        # 4. Get 3D coordinate in world space
        xyz = point_world[:3]
        log.info(f"xyz: {xyz}")
        return xyz


    def _filter_out_robot_from_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Filtering out a robot region from pcd.
        """
        # Currently this is a hard-coded script, based on initial robot position
        # TODO : remove robot using segmentation
        # ex) https://github.com/NVlabs/curobo/blob/ebb71702f3f70e767f40fd8e050674af0288abe8/examples/robot_image_segmentation_example.py
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)

        robot_offset = np.array([0.0, 0.0, 1.0]) + self.robot_position
        robot_dimension = np.array([0.3, 0.3, 2.0])
        point_mask = np.logical_and(
            np.logical_or(
                np.abs(points[:, 0] - robot_offset[0]) > robot_dimension[0] / 2,
                np.abs(points[:, 1] - robot_offset[1]) > robot_dimension[1] / 2,
                np.abs(points[:, 2] - robot_offset[2]) > robot_dimension[2] / 2,
            ),
            points[:, 2] < 0.3,
        )
        points_filtered = points[point_mask]
        colors_filtered = colors[point_mask]
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(points_filtered)
        pcd_filtered.colors = o3d.utility.Vector3dVector(colors_filtered)
        return pcd_filtered

    def _sorted_grasp_by_distance(self, target_point, grasp_group):
        dists = [np.linalg.norm(g.translation - target_point) for g in grasp_group]
        sorted_indices = np.argsort(dists)
        return grasp_group[sorted_indices]

    def _grasp_to_franka(self, grasps):
        """Convert the grasp pose to franka end-effector pose."""
        positions = grasps.translations.copy()
        rotations = grasps.rotation_matrices.copy()
        positions, rotations = self._world_to_franka(positions, rotations)

        # franka transform
        franka_L = np.diag([1, -1, 1])
        franka_R = np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]])
        rotations = franka_L @ rotations.transpose(0, 2, 1) @ franka_R

        quats = R.from_matrix(rotations).as_quat()

        # convert ee pose to tcp pose
        positions, quats = self._ee_pose_from_tcp_pose(
            tcp_pos=torch.tensor(positions, dtype=torch.float32),
            tcp_quat=torch.tensor(quats, dtype=torch.float32),
            depth=0.03,
        )

        # shape of (1, n_goalset, 3)
        ee_pos_target = positions.to("cuda:0").unsqueeze(0)
        ee_quat_target = quats.to("cuda:0").unsqueeze(0)
        return ee_pos_target, ee_quat_target

    def _ee_pose_from_tcp_pose(self, tcp_pos, tcp_quat, depth):
        tcp_rel_pos = (
            (torch.tensor(self.robot_tcp_rel_pos) + torch.tensor([0.0, 0.0, -depth])).unsqueeze(0).to(tcp_pos.device)
        )
        ee_pos = tcp_pos + torch.matmul(
            matrix_from_quat(tcp_quat),
            -tcp_rel_pos.unsqueeze(-1)
        ).squeeze()
        return ee_pos, tcp_quat

    def _world_to_franka(self, positions, rotations):
        positions = self.robot_position + positions @ self.robot_rotation_matrix.T
        rotations = self.robot_rotation_matrix @ rotations @ self.robot_rotation_matrix.T
        return positions, rotations

    def _set_motion_gen_with_pcd(self, pcd):
        world_cfg = WorldConfig(
            cuboid=[
                Cuboid(
                    name="ground",
                    pose=[0.0, 0.0, -0.4, 1.0, 0.0, 0.0, 0.0],
                    dims=[10.0, 10.0, 0.8],
                ),
            ],
            # TODO: is there any better method (using nvblox?)
            # TODO: get robot position and quaternion from args and apply to mesh pose
            mesh=[Mesh.from_pointcloud(np.asarray(pcd.points), pose=[1.15, 0.0, 0.0, 0, 0, 0, 1], pitch=0.005)],
        )
        world_cfg.save_world_as_mesh("get_started/output/motion_planning/3_object_grasping_vlm/world.ply")
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            world_cfg,
            TensorDeviceType(),
            self_collision_check=True,
            self_collision_opt=True,
            use_cuda_graph=False,  # True
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()

        self.motion_gen = motion_gen

    def do_fk(self, q: torch.Tensor):
        log.info(f"q: {q}")
        robot_state = self.kin_model.get_state(q[:self.curobo_n_dof], self.config_file["kinematics"]["ee_link"])
        return robot_state.ee_position.unsqueeze(0), robot_state.ee_quaternion.unsqueeze(0)


    def get_plan_config(self):
        return MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=10,
            enable_graph_attempt=None,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            parallel_finetune=True,
        )

    def get_joint_state(self, joint_pos: torch.Tensor):
        cu_js = JointState.from_position(
            position=joint_pos[:, :self.curobo_n_dof],
            joint_names=list(self.kin_model.joint_names)
        )
        return cu_js

    def plan_gripper(self, js: JointState, open_gripper: bool, step: int = 20):
        joint_pos = js.position.squeeze().repeat(step, 1)
        # if joint pos does not include ee dof, add zero to the end
        if joint_pos.shape[1] != self.curobo_n_dof + self.ee_n_dof:
            joint_pos = torch.cat([
                joint_pos,
                torch.zeros((joint_pos.shape[0], self.ee_n_dof), device=joint_pos.device)],
                dim=1)
        joint_pos[:, -self.ee_n_dof :] = torch.tensor(
            self.robot_gripper_open_q if open_gripper else self.robot_gripper_close_q
        )
        return joint_pos

    def plan_grasp(
        self,
        js: JointState,
        ee_pos_target: torch.Tensor,
        ee_quat_target: torch.Tensor,
        depth: float = 0.03, # TODO: use depth from grasp finder
    ):
        """Plan the grasp."""
        ik_goal = Pose(position=ee_pos_target, quaternion=ee_quat_target)
        log.debug(f"IK goal: {ik_goal}")
        cu_js = JointState.get_ordered_joint_state(js, list(self.kin_model.joint_names))
        log.info(f"Target EE pose: {ik_goal}")
        log.debug(f"Current robot joint state: {cu_js}")
        result = self.motion_gen.plan_grasp(
            start_state=cu_js,
            grasp_poses=ik_goal,
            plan_config=self.get_plan_config(),
            grasp_approach_offset=Pose(
                position=torch.tensor([0.0, 0.0, -depth], device="cuda:0"),
                quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0"),
            ),
            retract_offset=Pose(
                position=torch.tensor([0.0, 0.0, -depth], device="cuda:0"),
                quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0"),
            ),
            disable_collision_links=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
        )
        log.debug(f"Motion planning result:{result.success}")
        if not result.success:
            log.debug("No successful grasp plan found.")
            log.debug(f"Result: {result}")
            return None
        index = result.goalset_index.item()
        log.debug(f"Grasp index: {index}")

        def trajectory_plan(plan, open_gripper=False):
            joint_pos = torch.zeros((plan.shape[0], self.curobo_n_dof + self.ee_n_dof), device="cuda:0")
            joint_pos[:, : self.curobo_n_dof] = plan[:, :].position
            joint_pos[:, -self.ee_n_dof :] = torch.tensor(
                self.robot_gripper_open_q if open_gripper else self.robot_gripper_close_q
            )
            return joint_pos

        joint_pos = []
        joint_pos.append(trajectory_plan(result.grasp_interpolated_trajectory, open_gripper=True))
        cu_js = self.get_joint_state(joint_pos[-1][-1:, :])
        joint_pos.append(self.plan_gripper(cu_js, open_gripper=False, step=20))
        joint_pos.append(trajectory_plan(result.retract_interpolated_trajectory, open_gripper=False))
        joint_pos = torch.cat(joint_pos, dim=0)
        return joint_pos

    def plan_pose_single(
        self,
        js: JointState,
        ee_pos_target: torch.Tensor,
        ee_quat_target: torch.Tensor,
        open_gripper: bool = False
    ):
        """Move the robot to the target pose."""
        ik_goal = Pose(position=ee_pos_target, quaternion=ee_quat_target)
        log.info(f"Target EE pose: {ik_goal}")
        log.info(f"Current robot joint state: {js}")
        log.info(f"Current robot position shape: {js.position.shape}")
        result = self.motion_gen.plan_single(js, ik_goal, self.get_plan_config())
        log.debug(f"Motion planning result:{result.success}")

        if not result.success:
            log.debug("No successful motion plan found.")
            log.debug(f"Result: {result}")
            return None

        cmd_plan = result.get_interpolated_plan().position
        joint_pos = torch.zeros((cmd_plan.shape[0], self.curobo_n_dof + self.ee_n_dof), device="cuda:0")
        joint_pos[:, : self.curobo_n_dof] = cmd_plan[:, :]
        joint_pos[:, -self.ee_n_dof :] = torch.tensor(
            self.robot_gripper_open_q if open_gripper else self.robot_gripper_close_q
        )
        return joint_pos


    def plan_trajectory(self, js: JointState, img, depth, pcd, prompt, camera_intr_mat, camera_extr_mat):
        seq = self.point_extractor.extract_sequence(img, prompt)
        assert isinstance(seq, list) and len(seq) > 0, "No valid action sequence found"

        # TODO: support multiple steps in a sequence
        # Get 3D points from pixel coordinates
        start_point = seq[0]["pick_up"]
        end_point = seq[0]["put_down"]

        log.info(f"3d point of pixel: {start_point} / {end_point}")
        start_point_3d = self._get_3d_point_from_pixel(start_point, depth, camera_intr_mat, camera_extr_mat)
        end_point_3d = self._get_3d_point_from_pixel(end_point, depth, camera_intr_mat, camera_extr_mat)

        # TODO: get robot pose from this function
        pcd = self._filter_out_robot_from_pcd(pcd)
        self._set_motion_gen_with_pcd(pcd)

        # TODO: filter out pcd that are not close from start point, before getting the grasp candidates
        N = 8
        gg = self.grasp_finder.find(pcd)
        gg = self._sorted_grasp_by_distance(start_point_3d, gg)
        self.grasp_finder.visualize(
            pcd,
            gg[0:N],
            image_only=True,
            save_dir="3_object_grasping_vlm",
            filename=f"qwen2.5vl_top_one_{prompt}",
        )
        log.info(f"Grasp candidates: {gg[:N]}")
        ee_pos_pickup, ee_quat_pickup = self._grasp_to_franka(gg[:N])

        # Grasp
        joint_pos = []
        joint_pos.append(self.plan_gripper(js, open_gripper=True, step=20))
        joint_pos.append(self.plan_grasp(js, ee_pos_pickup, ee_quat_pickup))
        cu_js = self.get_joint_state(joint_pos[-1][-1:, :])
        ee_pos_pickup, ee_quat_pickup = self.do_fk(cu_js.position)

        # Pick up
        ee_pos_pickup[:, :, 2] += 0.2
        joint_pos.append(self.plan_pose_single(cu_js, ee_pos_pickup, ee_quat_pickup, open_gripper=False))
        cu_js = self.get_joint_state(joint_pos[-1][-1:, :])

        # Put down
        tcp_pos_putdown, _ = self._world_to_franka(np.array([end_point_3d]), np.eye(3))
        tcp_pos_putdown = torch.tensor(tcp_pos_putdown, dtype=torch.float32).unsqueeze(1).to("cuda:0")
        tcp_pos_putdown[:, :, 2] += 0.3  # lift up a bit
        ee_pos_putdown, ee_quat_putdown = self._ee_pose_from_tcp_pose(tcp_pos_putdown, ee_quat_pickup, 0.03)
        joint_pos.append(self.plan_pose_single(cu_js, ee_pos_putdown[0], ee_quat_putdown[0], open_gripper=False))
        cu_js = self.get_joint_state(joint_pos[-1][-1:, :])

        # Open Gripper
        joint_pos.append(self.plan_gripper(cu_js, open_gripper=True, step=20))

        # TODO: attach object to robot

        # Concat All Plans
        joint_pos = torch.cat(joint_pos, dim=0)
        return joint_pos

@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
