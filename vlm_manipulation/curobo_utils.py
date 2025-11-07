"""This script is used to grasp an object from a point cloud."""

from __future__ import annotations

"""This script is used to test the static scene."""


import copy

import numpy as np
import rootutils
import torch
import cv2
from loguru import logger as log
from rich.logging import RichHandler
from PIL import Image, ImageDraw

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import json
import re
from typing import Any, List, Optional
import time

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
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenPlanConfig,
    MotionGenConfig,
)

from third_party.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from third_party.sam2.sam2.build_sam import build_sam2
from third_party.GraspGen.grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg


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

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=img,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
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
        prompt_suffix = """
            Plan the picking up and placing down actions with points. \
            Report the point coordinates in JSON array like this: \
            [{"pick_up": [x, y], "place_down": [x, y]}, ...] \
            Use the object center for pick_up; use the intended contact point for place_down.
            For the place_down with prepositions like “in/into/inside”, it must be \
            on interior surface of the container, NOT the rim, NOT the outer wall, NOT the exterior top.
        """
        prompt = "Instruction: " + prompt + "\n" + prompt_suffix
        output_text = self.inference(img, prompt)
        log.info(f"Qwen2.5-VL action sequence: {output_text}")
        seq = self._extract_sequence(output_text)
        return seq

    def _extract_sequence(self, text):
        def _is_pick_put_obj(obj: Any) -> bool:
            if not isinstance(obj, dict):
                return False
            if not ("pick_up" in obj and "place_down" in obj):
                return False

            def ok(v):
                return (
                    isinstance(v, (list, tuple))
                    and len(v) == 2
                    and all(isinstance(n, (int, float)) for n in v)
                )

            return ok(obj["pick_up"]) and ok(obj["place_down"])

        def _normalize_pair(v):  # ints are usually what you want for pixels
            x, y = v
            return [int(x), int(y)]

        def flatten(obj: Any):
            out = []
            if _is_pick_put_obj(obj):
                out.append(
                    {
                        "pick_up": _normalize_pair(obj["pick_up"]),
                        "place_down": _normalize_pair(obj["place_down"]),
                    }
                )
            elif isinstance(obj, list):
                for it in obj:
                    if _is_pick_put_obj(it):
                        out.append(
                            {
                                "pick_up": _normalize_pair(it["pick_up"]),
                                "place_down": _normalize_pair(it["place_down"]),
                            }
                        )
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
        for match in re.finditer(
            r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text
        ):
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


class SAM2:
    def __init__(self):
        self.checkpoint = "./third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2 = SAM2ImagePredictor(build_sam2(self.model_cfg, self.checkpoint))

    def get_segmentation_mask(self, img: Image.Image, point_coords: np.ndarray):
        self.sam2.set_image(img)
        masks, _, _ = self.sam2.predict(
            point_coords=np.expand_dims(point_coords, axis=0),
            point_labels=np.array([1]),
        )
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask = mask.astype(np.uint8)
            mask = mask * 255
            mask = Image.fromarray(mask)
            mask.save(f"outputs/mask_{i}.png")

        _, labels = cv2.connectedComponents(masks[1].astype(np.uint8) * 255)
        seed_label = labels[point_coords[1], point_coords[0]]
        connected_mask = (labels == seed_label)

        connected_mask_render = connected_mask.astype(np.uint8) * 255
        connected_mask_render = Image.fromarray(connected_mask_render)
        connected_mask_render.save(f"outputs/mask_connected.png")

        return connected_mask

    def segment_from_pcd(self, 
        pcd: o3d.geometry.PointCloud,
        img: Image.Image,
        focus_point: np.ndarray, # 3d
        camera_intr_mat: np.ndarray,
        camera_extr_mat: np.ndarray,
    ):
        points = np.array(pcd.points)
        points_2d = (
            camera_intr_mat
            @ camera_extr_mat[:3, :]
            @ np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).T
        ).T
        
        focus_point_2d = (
            camera_intr_mat
            @ camera_extr_mat[:3, :]
            @ np.concatenate([focus_point.reshape(1, -1), np.ones((1, 1))], axis=1).T
        ).T
        focus_point_2d = focus_point_2d[:, :2] / focus_point_2d[:, 2:3]
        focus_point_2d = focus_point_2d.squeeze().astype(np.int32)
        log.debug(f"focus_point_2d: {focus_point_2d}")

        segmentation_mask = self.get_segmentation_mask(img, focus_point_2d)

        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        # out of bounds check
        points_x = np.clip(points_2d[:, 0], 0, segmentation_mask.shape[1] - 1).astype(
            int
        )
        points_y = np.clip(points_2d[:, 1], 0, segmentation_mask.shape[0] - 1).astype(
            int
        )
        segmentation_mask = segmentation_mask[points_y, points_x]
        return pcd.select_by_index(np.where(segmentation_mask)[0])

class GraspPoseFinder:
    def __init__(self):
        self.grasp_cfg = load_grasp_cfg(
            "third_party/GraspGen/models/checkpoints/graspgen_franka_panda.yml"
        )
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
        self.transform_matrix = None

    def set_transform_matrix(self, cam_extr_mat):
        self.transform_matrix = cam_extr_mat

    def find(
        self,
        pcd: o3d.geometry.PointCloud,
        focus_point: np.ndarray,
    ):
        points = np.array(pcd.points)

        # mask with focus point
        point_mask = np.linalg.norm(points - focus_point, axis=1) < 0.2
        pcd = pcd.select_by_index(np.where(point_mask)[0])

        o3d.io.write_point_cloud("outputs/pcd_masked.ply", pcd)
        pcd_transformed = pcd.transform(self.transform_matrix)
        points_transformed = np.array(pcd_transformed.points)
        o3d.io.write_point_cloud("outputs/pcd_transformed.ply", pcd_transformed)

        grasps_inferred, _ = GraspGenSampler.run_inference(
            points_transformed,
            self.grasp_sampler,
            grasp_threshold=0.8,
            num_grasps=200,
            topk_num_grasps=-1,
        )
        grasps_translations = grasps_inferred[:, :3, 3].detach().cpu().numpy()
        grasps_rotation_matrices = grasps_inferred[:, :3, :3].detach().cpu().numpy()

        grasps_translations = (
            grasps_translations - self.transform_matrix[:3, 3]
        ) @ self.transform_matrix[:3, :3]

        grasps_rotation_matrices = (
            self.transform_matrix[:3, :3].T @ grasps_rotation_matrices
        )

        log.info(f"Total grasp candidates: {len(grasps_translations)}")

        return grasps_translations, grasps_rotation_matrices

    def visualize(
        self,
        pcd: o3d.geometry.PointCloud,
        grasps,
        image_only=False,
        filename=None,
    ):
        pcd_clone = copy.deepcopy(pcd)
        grasps_clone = copy.deepcopy(grasps)
        pcd_clone.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points) @ self.transform_matrix
        )
        grasps_clone.translations = grasps_clone.translations @ self.transform_matrix.T
        grasps_clone.rotation_matrices = (
            self.transform_matrix
            @ grasps_clone.rotation_matrices
            @ self.transform_matrix.T
        )
        self.gsnet.visualize(
            pcd_clone,
            grasps_clone,
            image_only=image_only,
            filename=filename,
        )


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

    def __init__(self):
        self.point_extractor = VLMPointExtractor()
        self.grasp_finder = GraspPoseFinder()
        self.sam2 = SAM2()

        self.robot_gripper_open_q = [0.04, 0.04]
        self.robot_gripper_close_q = [0.00, 0.00]
        self.robot_tcp_rel_pos = [0.0, 0.0, 0.10312]
        self.curobo_n_dof = 7
        self.ee_n_dof = 2

        self.motion_gen = None
        self.motion_gen_count = 0

        tensor_args = TensorDeviceType()
        self.config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))[
            "robot_cfg"
        ]
        # config_file = load_yaml(join_path(get_robot_path(), robot_cfg.curobo_ref_cfg_name))["robot_cfg"]
        self.robot_config = RobotConfig.from_dict(self.config_file, tensor_args)
        self.kin_model = CudaRobotModel(self.robot_config.kinematics)

    def _get_3d_point_from_pixels(
        self,
        pixels,  # iterable of (x, y) pixel coords
        depth,  # HxW depth array (same frame as pixels)
        cam_intr_mat,  # 3x3 intrinsics K
        cam_extr_mat,  # 4x4 extrinsics; by default assumed world->camera
    ):
        """
        Returns:
            pts_3d: (N, 3) ndarray of 3D points in *world*
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixels must be an array of shape (N,2)")

        H, W = depth.shape[:2]
        fx, fy = cam_intr_mat[0, 0], cam_intr_mat[1, 1]
        cx, cy = cam_intr_mat[0, 2], cam_intr_mat[1, 2]

        # index into depth
        xs = pixels[:, 0]
        ys = pixels[:, 1]
        xi = xs.astype(int)
        yi = ys.astype(int)
        z = depth[yi, xi].astype(np.float64).flatten()

        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        Z = z
        pts_cam = np.stack([X, Y, Z], axis=1)

        # to world frame via a rigid transform
        cam_extr = np.array(cam_extr_mat)
        cam_extr = np.linalg.inv(cam_extr)
        T = np.asarray(cam_extr, dtype=np.float64)

        pts_cam_h = np.concatenate(
            [pts_cam, np.ones((len(pixels), 1))], axis=1
        )  # (N,4)
        pts_world_h = (T @ pts_cam_h.T).T
        pts_world = pts_world_h[:, :3]
        return pts_world

    def _get_3d_point_from_pixel(self, pixel_point, depth, cam_intr_mat, cam_extr_mat):
        """
        Convert 2D pixel coordinates to 3D points using the point cloud.

        Since the point cloud was generated from RGBD using get_pcd_from_rgbd(),
        there's a direct mapping between pixels and 3D points.

        Args:
            pixel_point: 2D pixel coordinates (x, y)
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

    def _filter_out_robot_from_pcd(
        self, pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        """
        Filtering out a robot region from pcd.
        """
        # Currently this is a hard-coded script, based on initial robot position
        # TODO : remove robot using segmentation
        # ex) https://github.com/NVlabs/curobo/blob/ebb71702f3f70e767f40fd8e050674af0288abe8/examples/robot_image_segmentation_example.py
        points = np.array(pcd.points)

        robot_offset = np.array([0.0, 0.0, 1.0])
        robot_dimension = np.array([0.3, 0.3, 2.0])
        point_mask = np.logical_and(
            np.logical_or(
                np.abs(points[:, 0] - robot_offset[0]) > robot_dimension[0] / 2,
                np.abs(points[:, 1] - robot_offset[1]) > robot_dimension[1] / 2,
                np.abs(points[:, 2] - robot_offset[2]) > robot_dimension[2] / 2,
            ),
            points[:, 2] < 0.5,
        )
        return pcd.select_by_index(np.where(point_mask)[0])

    def _sorted_grasp_by_distance(
        self, target_point, grasp_translations, grasp_rotation_matrices
    ):
        dists = [np.linalg.norm(g - target_point) for g in grasp_translations]
        sorted_indices = np.argsort(dists)
        return (
            grasp_translations[sorted_indices],
            grasp_rotation_matrices[sorted_indices],
        )

    def _grasp_to_franka(self, grasp_translations, grasp_rotation_matrices):
        """Convert the grasp pose to franka end-effector pose."""
        positions = grasp_translations.copy()
        rotations = grasp_rotation_matrices.copy()
        rotation_transform_for_franka = torch.tensor(
            [
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
        )
        rotations = (
            torch.tensor(rotations, dtype=torch.float32) @ rotation_transform_for_franka
        )

        quats = R.from_matrix(rotations).as_quat()
        # convert to wxyz format
        quats = np.concatenate([quats[..., 3:4], quats[..., :3]], axis=-1)

        log.info(f"Positions: {positions}")
        log.info(f"Quats: {quats}")

        # convert ee pose from tcp pose
        positions = torch.tensor(positions, dtype=torch.float32)
        quats = torch.tensor(quats, dtype=torch.float32)
        # positions, quats = self._ee_pose_from_tcp_pose(
        # tcp_pos=torch.tensor(positions, dtype=torch.float32),
        # tcp_quat=torch.tensor(quats, dtype=torch.float32),
        # )
        # grasp_offset = torch.cat(
        # [
        # torch.zeros(len(grasps.depths), 2),
        # torch.tensor(grasps.depths, dtype=torch.float32).unsqueeze(1),
        # ],
        # dim=1,
        # ).unsqueeze(2)
        # positions = positions + torch.matmul(
        # matrix_from_quat(quats), grasp_offset
        # ).squeeze(2)

        # shape of (1, n_goalset, 3)
        ee_pos_target = positions.to("cuda:0").unsqueeze(0)
        ee_quat_target = quats.to("cuda:0").unsqueeze(0)
        return ee_pos_target, ee_quat_target

    def _ee_pose_from_tcp_pose(self, tcp_pos, tcp_quat):
        tcp_rel_pos = (
            (torch.tensor(self.robot_tcp_rel_pos)).unsqueeze(0).to(tcp_pos.device)
        )
        ee_pos = (
            tcp_pos
            + torch.matmul(
                matrix_from_quat(tcp_quat), -tcp_rel_pos.unsqueeze(-1)
            ).squeeze()
        )
        return ee_pos, tcp_quat

    def _set_motion_gen_with_pcd(self, pcd):
        start_time = time.time()
        world_cfg = WorldConfig(
            # cuboid=[
            # Cuboid(
            # name="ground",
            # pose=[0.0, 0.0, -0.25, 1.0, 0.0, 0.0, 0.0],
            # dims=[3.0, 3.0, 0.4],
            # ),
            # ],
            # TODO: is there any better method (using nvblox?)
            mesh=[
                Mesh.from_pointcloud(
                    name="pcd mesh" + str(self.motion_gen_count),
                    pointcloud=np.asarray(pcd.points),
                    pose=[0.0, 0.0, 0.0, 1, 0, 0, 0],
                    pitch=0.005,
                )
            ],
        )
        world_cfg.save_world_as_mesh("outputs/world.ply")
        self.motion_gen_count += 1
        end_time = time.time()
        log.error(f"[MotionGen] Time taken to process pcd: {end_time - start_time}")

        if self.motion_gen is not None:
            self.motion_gen.update_world(world_cfg)
            end_time = time.time()
            log.error(
                f"[MotionGen] Time taken to update world: {end_time - start_time}"
            )
            return

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            world_cfg,
            TensorDeviceType(),
            self_collision_check=True,
            self_collision_opt=True,
            use_cuda_graph=False,  # True,
        )
        end_time = time.time()
        log.error(
            f"[MotionGen] Time taken to load world config: {end_time - start_time}"
        )
        motion_gen = MotionGen(motion_gen_config)
        end_time = time.time()
        log.error(f"[MotionGen] Time taken to load motion gen: {end_time - start_time}")
        motion_gen.warmup()

        end_time = time.time()
        log.error(
            f"[MotionGen] Time taken to warmup motion gen: {end_time - start_time}"
        )

        self.motion_gen = motion_gen

    def do_fk(self, q: torch.Tensor):
        log.info(f"q: {q}")
        robot_state = self.kin_model.get_state(
            q[: self.curobo_n_dof], self.config_file["kinematics"]["ee_link"]
        )
        return robot_state.ee_position.unsqueeze(
            0
        ), robot_state.ee_quaternion.unsqueeze(0)

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
            position=joint_pos[:, : self.curobo_n_dof],
            joint_names=list(self.kin_model.joint_names),
        )
        return cu_js

    def plan_gripper(self, js: JointState, open_gripper: bool, step: int = 20):
        joint_pos = js.position.squeeze().repeat(step, 1)
        # if joint pos does not include ee dof, add zero to the end
        if joint_pos.shape[1] != self.curobo_n_dof + self.ee_n_dof:
            joint_pos = torch.cat(
                [
                    joint_pos,
                    torch.zeros(
                        (joint_pos.shape[0], self.ee_n_dof), device=joint_pos.device
                    ),
                ],
                dim=1,
            )
        joint_pos[:, -self.ee_n_dof :] = torch.tensor(
            self.robot_gripper_open_q if open_gripper else self.robot_gripper_close_q
        )
        return joint_pos

    def plan_grasp(
        self,
        js: JointState,
        ee_pos_target: torch.Tensor,
        ee_quat_target: torch.Tensor,
        depth: float = 0.03,  # TODO: use depth from grasp finder
    ):
        """Plan the grasp."""

        cu_js = JointState.get_ordered_joint_state(js, list(self.kin_model.joint_names))
        disable_collision_links = [
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        # make batch instead of goalset
        ik_goal = Pose(
            position=ee_pos_target.transpose(0, 1),
            quaternion=ee_quat_target.transpose(0, 1),
        )
        batch_cu_js = cu_js.repeat_seeds(ee_pos_target.shape[1])

        # first, check reachability with plan_goalset
        self.motion_gen.toggle_link_collision(disable_collision_links, False)
        batch_motion_gen_result = self.motion_gen.plan_batch(
            batch_cu_js,
            ik_goal,
            self.get_plan_config(),
        )
        self.motion_gen.toggle_link_collision(disable_collision_links, True)
        if not torch.any(batch_motion_gen_result.success):
            log.error("No successful grasp motion plan found.")
            return None
        else:
            log.debug(f"Reachablility: {batch_motion_gen_result.success}")

        for i in range(ee_pos_target.shape[1]):
            if not batch_motion_gen_result.success[i].item():
                continue
            ik_goal = Pose(
                position=ee_pos_target[:, i : i + 1, :],
                quaternion=ee_quat_target[:, i : i + 1, :],
            )
            log.debug(f"IK goal: {ik_goal}")

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
                disable_collision_links=disable_collision_links,
            )
            log.debug(f"Motion planning result:{result.success}")
            if result.success:
                break
            else:
                log.debug(f"Motion planning result: {result.status}")

        if not result.success:
            log.debug("No successful grasp motion plan found.")
            return None

        # index = result.goalset_index.item()
        index = i
        log.debug(f"Grasp index: {index}")

        def trajectory_plan(plan, open_gripper=False):
            joint_pos = torch.zeros(
                (plan.shape[0], self.curobo_n_dof + self.ee_n_dof), device="cuda:0"
            )
            joint_pos[:, : self.curobo_n_dof] = plan[:, :].position
            joint_pos[:, -self.ee_n_dof :] = torch.tensor(
                self.robot_gripper_open_q
                if open_gripper
                else self.robot_gripper_close_q
            )
            return joint_pos

        joint_pos = []
        joint_pos.append(
            trajectory_plan(result.grasp_interpolated_trajectory, open_gripper=True)
        )
        cu_js = self.get_joint_state(joint_pos[-1][-1:, :])
        joint_pos.append(self.plan_gripper(cu_js, open_gripper=False, step=20))
        joint_pos.append(
            trajectory_plan(result.retract_interpolated_trajectory, open_gripper=False)
        )
        joint_pos = torch.cat(joint_pos, dim=0)
        return joint_pos

    def plan_pose_single(
        self,
        js: JointState,
        ee_pos_target: torch.Tensor,
        ee_quat_target: torch.Tensor,
        open_gripper: bool = False,
    ):
        """Move the robot to the target pose."""
        ik_goal = Pose(position=ee_pos_target, quaternion=ee_quat_target)
        log.info(f"Target EE pose: {ik_goal}")
        result = self.motion_gen.plan_single(js, ik_goal, self.get_plan_config())
        log.debug(f"Motion planning result:{result.success}")

        if not result.success:
            log.debug("No successful motion plan found.")
            log.debug(f"Result: {result}")
            return None

        cmd_plan = result.get_interpolated_plan().position
        joint_pos = torch.zeros(
            (cmd_plan.shape[0], self.curobo_n_dof + self.ee_n_dof), device="cuda:0"
        )
        joint_pos[:, : self.curobo_n_dof] = cmd_plan[:, :]
        joint_pos[:, -self.ee_n_dof :] = torch.tensor(
            self.robot_gripper_open_q if open_gripper else self.robot_gripper_close_q
        )
        return joint_pos

    def plan_trajectory(
        self,
        js: JointState,
        images: list[Image.Image],
        depth: list[np.ndarray],
        pcds: list[o3d.geometry.PointCloud],
        prompt,
        camera_intr_mat: list[np.ndarray],
        camera_extr_mat: list[np.ndarray],
        task_id,
        eval_index,
    ):
        start_time = time.time()
        seq = self.point_extractor.extract_sequence(images[0], prompt)
        end_time = time.time()
        log.error(
            f"[TrajOptimizer] Time taken to extract sequence: {end_time - start_time}"
        )
        assert isinstance(seq, list) and len(seq) > 0, "No valid action sequence found"

        # TODO: support multiple steps in a sequence
        # Get 3D points from pixel coordinates
        start_point = seq[0]["pick_up"]
        end_point = seq[0]["place_down"]

        log.info(f"2d point of pixel: {start_point} / {end_point}")

        start_point_3d = self._get_3d_point_from_pixel(
            start_point, depth[0], camera_intr_mat[0], camera_extr_mat[0]
        )
        log.debug(f"start_point_3d: {start_point_3d}")

        end_point_3d = self._get_3d_point_from_pixel(
            end_point, depth[0], camera_intr_mat[0], camera_extr_mat[0]
        )

        pcd_original = o3d.geometry.PointCloud()
        for pcd in pcds:
            pcd_original += pcd

        pcd_segmented = o3d.geometry.PointCloud()
        for pcd, img, intr, extr in zip(pcds, images, camera_intr_mat, camera_extr_mat):
            pcd = self.sam2.segment_from_pcd(
                pcd,
                img,
                start_point_3d,
                intr,
                extr,
            )
            pcd_segmented += pcd

        log.debug(f"pcd_original: {len(pcd_original.points)}")
        log.debug(f"pcd_segmented: {len(pcd_segmented.points)}")

        # TODO: add debug flag for visualization
        # Draw image
        draw = ImageDraw.Draw(images[0])
        draw.circle(start_point, 4, fill="red")
        draw.circle(end_point, 4, fill="blue")
        images[0].save(f"outputs/image_{task_id}_{eval_index}.png")

        pcd_original = self._filter_out_robot_from_pcd(pcd_original)
        end_time = time.time()
        log.error(
            f"[TrajOptimizer] Time taken to filter out robot from pcd: {end_time - start_time}"
        )
        self._set_motion_gen_with_pcd(pcd_original)

        end_time = time.time()
        log.error(
            f"[TrajOptimizer] Time taken to set motion gen with pcd: {end_time - start_time}"
        )

        # number of grasp candidates to check
        N = 128
        gg_translations, gg_rotation_matrices = self.grasp_finder.find(pcd_segmented, start_point_3d)
        gg_translations, gg_rotation_matrices = self._sorted_grasp_by_distance(
            start_point_3d, gg_translations, gg_rotation_matrices
        )

        end_time = time.time()
        log.error(f"[TrajOptimizer] Time taken to find grasp: {end_time - start_time}")

        # TODO: fix visualization bug
        # self.grasp_finder.visualize(
        # pcd,
        # gg[0:N],
        # image_only=True,
        # filename=f"outputs/gsnet.png",
        # )
        ee_pos_pickup, ee_quat_pickup = self._grasp_to_franka(
            gg_translations[:N], gg_rotation_matrices[:N]
        )

        # Grasp
        joint_pos = []
        joint_pos.append(self.plan_gripper(js, open_gripper=True, step=20))
        joint_pos.append(self.plan_grasp(js, ee_pos_pickup, ee_quat_pickup))
        cu_js = self.get_joint_state(joint_pos[-1][-1:, :])
        ee_pos_pickup, ee_quat_pickup = self.do_fk(cu_js.position)
        ee_from_point = ee_pos_pickup - torch.tensor(
            start_point_3d, dtype=torch.float32
        ).to("cuda:0").unsqueeze(0).unsqueeze(0)

        # Pick up
        ee_pos_pickup_lift = ee_pos_pickup.clone()
        ee_quat_pickup_lift = ee_quat_pickup.clone()
        ee_pos_pickup_lift[:, :, 2] += 0.2
        self.motion_gen.toggle_link_collision(
            [
                "panda_link0",
                "panda_link1",
                "panda_link2",
                "panda_link3",
                "panda_link4",
                "panda_link5",
                "panda_link6",
                "panda_link7",
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
            ],
            False,
        )
        joint_pos.append(
            self.plan_pose_single(
                cu_js, ee_pos_pickup_lift, ee_quat_pickup_lift, open_gripper=False
            )
        )
        self.motion_gen.toggle_link_collision(
            [
                "panda_link0",
                "panda_link1",
                "panda_link2",
                "panda_link3",
                "panda_link4",
                "panda_link5",
                "panda_link6",
                "panda_link7",
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
            ],
            True,
        )
        cu_js = self.get_joint_state(joint_pos[-1][-1:, :])

        # Put down
        # trial 1
        ee_pos_putdown = (
            torch.tensor(end_point_3d, dtype=torch.float32)
            .to("cuda:0")
            .unsqueeze(0)
            .unsqueeze(0)
            + ee_from_point
        )
        ee_quat_putdown = ee_quat_pickup.clone()
        ee_pos_putdown[:, :, 2] += 0.3

        # TODO: move linearly to the target pose
        joint_pos.append(
            self.plan_pose_single(
                cu_js, ee_pos_putdown[0], ee_quat_putdown[0], open_gripper=False
            )
        )
        if joint_pos[-1] is None:
            # trial 2, rotate the target pose 180 degrees around z-axis and re-plan the trajectory
            joint_pos = joint_pos[:-1]  # remove trial 1

            rotations = matrix_from_quat(ee_quat_pickup).squeeze()
            R_180 = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
            ).to(rotations.device)
            rotations = R_180 @ rotations
            ee_quat_putdown = (
                torch.tensor(quat_from_matrix(rotations))
                .to(ee_quat_putdown.device)
                .to(torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            ee_pos_putdown = torch.tensor(end_point_3d, dtype=torch.float32).to(
                "cuda:0"
            ).unsqueeze(0).unsqueeze(0) + (R_180 @ ee_from_point.squeeze()).unsqueeze(
                0
            ).unsqueeze(
                0
            )
            ee_pos_putdown[:, :, 2] += 0.3

            joint_pos.append(
                self.plan_pose_single(
                    cu_js, ee_pos_putdown[0], ee_quat_putdown[0], open_gripper=False
                )
            )

        cu_js = self.get_joint_state(joint_pos[-1][-1:, :])

        # Stay for a while and then open gripper
        joint_pos.append(self.plan_gripper(cu_js, open_gripper=False, step=30))
        joint_pos.append(self.plan_gripper(cu_js, open_gripper=True, step=50))

        # TODO: attach object to robot
        end_time = time.time()
        log.error(
            f"[TrajOptimizer] Time taken to plan trajectory: {end_time - start_time}"
        )

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


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    # let matrix is a nx3x3 matrix
    q_xyzw = (
        torch.tensor(R.from_matrix(matrix.detach().cpu().numpy()).as_quat())
        .to(matrix.device)
        .to(matrix.dtype)
    )
    x, y, z, w = torch.unbind(q_xyzw, -1)
    q_wxyz = torch.stack([w, x, y, z], -1)
    return q_wxyz
