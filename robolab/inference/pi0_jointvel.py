# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image
from scipy.spatial.transform import Rotation as ScipyRotation

from .base_client import InferenceClient

# Isaac PinholeCameraCfg: focal_length & apertures in cm (see isaaclab PinholeCameraCfg).
_STEREO_FOCAL_CM = 2.1
_STEREO_H_APERTURE_CM = 5.376
_STEREO_V_APERTURE_CM = 3.024
_WRIST_FOCAL_CM = 2.8
_SRC_W, _SRC_H = 1280, 720
_RES = 224


def _pinhole_K_pixels(
    focal_length_cm: float,
    horizontal_aperture_cm: float,
    vertical_aperture_cm: float,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    """Pixel K from Isaac pinhole params (fx, fy, cx, cy), OpenCV-style."""
    fx = (focal_length_cm / horizontal_aperture_cm) * float(width_px)
    fy = (focal_length_cm / vertical_aperture_cm) * float(height_px)
    cx = 0.5 * float(width_px)
    cy = 0.5 * float(height_px)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _scale_K_to_resolution(K_src: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Independent scale of fx, cx by w ratio and fy, cy by h ratio (matches naive resize)."""
    sw = dst_wh[0] / float(src_wh[0])
    sh = dst_wh[1] / float(src_wh[1])
    return np.array(
        [
            [K_src[0, 0] * sw, 0.0, K_src[0, 2] * sw],
            [0.0, K_src[1, 1] * sh, K_src[1, 2] * sh],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _omniguide_intrinsics_224() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intrinsics aligned to ``resize_with_pad(..., 224, 224)`` inputs (approximate linear scaling)."""
    K_stereo_full = _pinhole_K_pixels(
        _STEREO_FOCAL_CM, _STEREO_H_APERTURE_CM, _STEREO_V_APERTURE_CM, _SRC_W, _SRC_H
    )
    K_wrist_full = _pinhole_K_pixels(
        _WRIST_FOCAL_CM, _STEREO_H_APERTURE_CM, _STEREO_V_APERTURE_CM, _SRC_W, _SRC_H
    )
    dst = (_RES, _RES)
    src = (_SRC_W, _SRC_H)
    return (
        _scale_K_to_resolution(K_stereo_full, src, dst),
        _scale_K_to_resolution(K_stereo_full, src, dst),
        _scale_K_to_resolution(K_wrist_full, src, dst),
    )


def _T_env_from_cam_open_gl(
    rot_wxyz: tuple[float, float, float, float],
    pos: tuple[float, float, float],
) -> np.ndarray:
    """``T_parent_from_cam`` for Isaac ``OffsetCfg``: ``p_parent = R @ p_cam + t`` (OpenGL camera).

    Matches ``principal_ray_hit_y_plane`` / tiled camera: ``R.apply([0,0,-1])`` is optical axis in parent.
    Parent is the env / camera mount frame (not true world when ``num_envs > 1`` with env spacing).
    """
    w, x, y, z = rot_wxyz
    r = ScipyRotation.from_quat([x, y, z, w])
    t = np.asarray(pos, dtype=np.float32).reshape(3)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.as_matrix().astype(np.float32)
    out[:3, 3] = t
    return out


def _omniguide_extrinsics_from_robolab_cfgs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left / right from ``OverShoulderLeftCameraCfg``; overhead (wrist) from ``DroidCfg.wrist_cam`` offset.

    Wrist offset is **relative to the gripper link**, not env—omniguide may still expect a consistent world
    frame; full accuracy needs composing with live EE pose from the sim.
    """
    from robolab.robots.droid import DroidCfg
    from robolab.variations.camera import OverShoulderLeftCameraCfg

    # @configclass fields are not visible on the type; read from instances.
    shoulder = OverShoulderLeftCameraCfg()
    robot = DroidCfg()
    ext = shoulder.external_cam.offset
    rc = shoulder.right_cam.offset
    wr = robot.wrist_cam.offset
    return (
        _T_env_from_cam_open_gl(ext.rot, ext.pos),
        _T_env_from_cam_open_gl(rc.rot, rc.pos),
        _T_env_from_cam_open_gl(wr.rot, wr.pos),
    )


_E_LEFT, _E_RIGHT, _E_OVER = _omniguide_extrinsics_from_robolab_cfgs()


def _optional_omniguide_image_observations(curr_obs: dict) -> dict:
    """Extra omniguide keys (stereo + wrist + intrinsics/extrinsics); ignored by standard OpenPI servers."""
    if "stereo_right_image" not in curr_obs:
        return {}
    h = w = _RES
    K_left, K_right, K_over = _omniguide_intrinsics_224()
    wrist224 = image_tools.resize_with_pad(curr_obs["wrist_image"], h, w)
    return {
        "observation/left_image": image_tools.resize_with_pad(curr_obs["right_image"], h, w),
        "observation/right_image": image_tools.resize_with_pad(curr_obs["stereo_right_image"], h, w),
        # Omniguide pi0_pytorch stacks (left, right, overhead); overhead is the wrist view in sim.
        "observation/overhead_image": wrist224,
        "observation/wrist_image": wrist224,
        "observation/left_intrin": K_left,
        "observation/left_extrin": _E_LEFT.copy(),
        "observation/right_intrin": K_right,
        "observation/right_extrin": _E_RIGHT.copy(),
        "observation/overhead_intrin": K_over,
        "observation/overhead_extrin": _E_OVER.copy(),
    }


class Pi0DroidJointvelClient(InferenceClient):
    """OpenPI client that interprets arm outputs as normalized joint velocities.

    The environment expects joint *position* targets. We therefore integrate:
      q_target[t+1] = q_target[t] + (v_norm * vel_limits) * dt
    """

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 8,
        dt: float = 1.0 / 15.0,
        vel_limits: tuple[float, float, float, float, float, float, float] = (
            2.175,
            2.175,
            2.175,
            2.175,
            2.61,
            2.61,
            2.61,
        ),
    ) -> None:
        print(f"[{self.__class__.__name__}] Awaiting for server on {remote_host}:{remote_port} to be ready...")
        self.client = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)
        print(f"[{self.__class__.__name__}] Server on {remote_host}:{remote_port} is ready.")

        self.open_loop_horizon = int(open_loop_horizon)
        self.dt = float(dt)
        self.vel_limits = np.asarray(vel_limits, dtype=np.float32)

        # Per-env state for multi-env runs.
        self._env_chunk: dict[int, object] = {}
        self._env_counter: dict[int, int] = {}
        self._env_q_target: dict[int, np.ndarray] = {}

    def visualize(self, request: dict):
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        return np.concatenate([base_img, wrist_img], axis=1)

    def reset(self):
        self._env_chunk.clear()
        self._env_counter.clear()
        self._env_q_target.clear()

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        curr_obs = self._extract_observation(obs, env_id=env_id)

        counter = self._env_counter.get(env_id, 0)
        chunk = self._env_chunk.get(env_id, None)

        if counter == 0 or counter >= self.open_loop_horizon or chunk is None:
            counter = 0
            # Anchor open-loop integration to current measured arm position.
            self._env_q_target[env_id] = np.asarray(curr_obs["joint_position"], dtype=np.float32).copy()
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs["right_image"], 224, 224
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    curr_obs["wrist_image"], 224, 224
                ),
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            request_data.update(_optional_omniguide_image_observations(curr_obs))
            chunk = np.asarray(self.client.infer(request_data)["actions"], dtype=np.float32)
            self._env_chunk[env_id] = chunk

        action_raw = np.asarray(chunk[counter], dtype=np.float32)
        self._env_counter[env_id] = counter + 1

        # Convert normalized velocity command to position target.
        v_norm = np.clip(action_raw[:7], -1.0, 1.0)
        v_cmd = v_norm * self.vel_limits
        q_target = self._env_q_target.get(env_id)
        if q_target is None:
            q_target = np.asarray(curr_obs["joint_position"], dtype=np.float32).copy()
        q_target = q_target + v_cmd * self.dt
        self._env_q_target[env_id] = q_target

        # Binarize gripper command.
        gripper = float(action_raw[7]) if action_raw.shape[0] >= 8 else 0.0
        gripper = 1.0 if gripper > 0.5 else 0.0

        action = np.concatenate([q_target.astype(np.float32), np.array([gripper], dtype=np.float32)], axis=0)

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        both = np.concatenate([img1, img2], axis=1)

        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, env_id=0, save_to_disk=False):
        right_image = obs_dict["image_obs"]["external_cam"][env_id].clone().detach().cpu().numpy()
        wrist_image = obs_dict["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        robot_state = obs_dict["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

        out = {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
        if "right_cam" in obs_dict["image_obs"]:
            out["stereo_right_image"] = obs_dict["image_obs"]["right_cam"][env_id].clone().detach().cpu().numpy()
        return out
