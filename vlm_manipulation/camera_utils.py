import torch


def get_cam_params(
    cam_pos: torch.Tensor,
    cam_look_at: torch.Tensor,
    width=640,
    height=480,
    focal_length=24,
    horizontal_aperture=20.955,
    vertical_aperture=None,
):
    """Get the camera parameters.

    Args:
        cam_pos: The camera position.
        cam_look_at: The camera look at point.
        width: The width of the image.
        height: The height of the image.
        focal_length: The focal length of the camera.
        horizontal_aperture: The horizontal aperture of the camera.
        vertical_aperture: The vertical aperture of the camera.

    Returns:
        The camera parameters.
    """
    if vertical_aperture is None:
        vertical_aperture = horizontal_aperture * height / width

    device = cam_pos.device
    num_envs = len(cam_pos)
    cam_front = cam_look_at - cam_pos
    cam_right = torch.cross(
        cam_front, torch.tensor([[0.0, 0.0, 1.0]], device=device), dim=1
    )
    cam_up = torch.cross(cam_right, cam_front)

    cam_right = cam_right / (torch.norm(cam_right, dim=-1, keepdim=True) + 1e-12)
    cam_front = cam_front / (torch.norm(cam_front, dim=-1, keepdim=True) + 1e-12)
    cam_up = cam_up / (torch.norm(cam_up, dim=-1, keepdim=True) + 1e-12)

    # Camera convention difference between ROS and Isaac Sim
    R = torch.stack([cam_right, -cam_up, cam_front], dim=1)  # .transpose(-1, -2)
    t = -torch.bmm(R, cam_pos.unsqueeze(-1)).squeeze()
    extrinsics = torch.eye(4, device=device).unsqueeze(0).tile([num_envs, 1, 1])
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = t

    fx = width * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    cx = width * 0.5
    cy = height * 0.5

    intrinsics = (
        torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device)
        .unsqueeze(0)
        .tile([num_envs, 1, 1])
    )

    return extrinsics, intrinsics


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
