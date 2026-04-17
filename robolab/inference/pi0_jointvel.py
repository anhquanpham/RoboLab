# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image

from .base_client import InferenceClient


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

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
