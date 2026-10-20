#!/usr/bin/env python3
"""Policy-agnostic deploy / replay / eval tool."""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tyro
from loguru import logger

# robot/gr00t imports are lazy/inline to keep module imports light.
from dex_deploy.policies import auto_detect_policy, get_policy
from dex_deploy.lerobot_dataset import LeRobotEpisode


def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict:
    """Per-dim MAE / overall MSE / per-frame L1 of predicted vs GT actions."""
    preds = np.asarray(preds, dtype=np.float64)
    gts = np.asarray(gts, dtype=np.float64)
    if preds.shape != gts.shape:
        raise ValueError(
            f"preds/gts shape mismatch: {preds.shape} vs {gts.shape}"
        )
    diff = preds - gts
    per_frame_l1 = np.mean(np.abs(diff), axis=1)
    return {
        "frames": int(preds.shape[0]),
        "l1": {
            "mean": float(per_frame_l1.mean()),
            "std": float(per_frame_l1.std()),
            "min": float(per_frame_l1.min()),
            "max": float(per_frame_l1.max()),
            "per_frame": per_frame_l1.tolist(),
        },
        "mse": float((diff ** 2).mean()),
        "per_dim_mae": np.abs(diff).mean(axis=0).tolist(),
    }


def _plot_actions(preds: np.ndarray, gts: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = preds.shape[1]
    frames = range(preds.shape[0])
    fig, axes = plt.subplots(d, 1, figsize=(14, max(2, d * 1.4)), sharex=True)
    if d == 1:
        axes = [axes]
    fig.suptitle("Predicted vs Ground-Truth Actions", fontsize=14, fontweight="bold")
    for j in range(d):
        axes[j].plot(frames, preds[:, j], "b-", lw=1.5, alpha=0.8, label="Predicted")
        axes[j].plot(frames, gts[:, j], "r--", lw=1.5, alpha=0.8, label="GT")
        axes[j].set_ylabel(f"dim_{j}", fontsize=8)
        axes[j].grid(True, alpha=0.3)
        if j == 0:
            axes[j].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Frame", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


##### KEY MAPPING (robot side: raw robot camera names -> policy video keys) #####
CAMERA_NAME_MAP: Dict[str, str] = {
    "head_left_rgb": "head_left_rgb_img",
    "head_right_rgb": "head_right_rgb_img",
    "left_wrist_rgb": "left_wrist_rgb_img",
    "right_wrist_rgb": "right_wrist_rgb_img",
}
##### END KEY MAPPING (robot side) #####


class Deployer:
    """Policy-agnostic deploy / replay / eval runner."""

    def __init__(
        self,
        policy,
        task: str,
        *,
        use_dataset_state: bool = False,
        use_dataset_images: bool = False,
        publish_actions: bool = False,
        dataset_path: Optional[str] = None,
        episode_index: int = 0,
        namespace: str = "",
        action_mode: str = "absolute",
        img_resolution: Optional[Tuple[int, int]] = None,
        debug: bool = False,
        ramp_steps: int = 0,
        max_steps: int = 0,
    ):
        self.policy = policy
        self.task = task
        self.ramp_steps = int(ramp_steps)
        # EDIT 3: optional cap on rollout iterations. 0 = unlimited (default).
        # Caps BOTH live (n=10_000_000) and dataset (n=num_frames) loops.
        self.max_steps = int(max_steps)
        self.use_dataset_state = use_dataset_state
        self.use_dataset_images = use_dataset_images
        self.publish_actions = publish_actions
        self.dataset_path = dataset_path
        self.episode_index = int(episode_index)
        self.namespace = namespace
        self.img_resolution = img_resolution
        self.debug = debug

        self.output_dir = "eval"

        if action_mode not in ("absolute", "relative"):
            raise ValueError(
                f"Invalid action_mode: {action_mode}. Must be 'absolute' or 'relative'"
            )
        self.action_mode = action_mode

        self.video_keys: List[str] = list(policy.video_keys)

        # ----------------------------------------------------------------------
        # EDIT 1: Build the episode whenever a dataset_path is provided, not only
        # when dataset OBSERVATIONS are requested. This lets the safe ramp read the
        # episode's frame-0 start pose even in LIVE closed-loop rollout (where the
        # policy reads real cameras/joints, but we still want to position the robot
        # at the demo's start before handing control to the policy).
        #   - --dataset-path alone  -> episode built, used ONLY for ramp target.
        #   - --use-dataset-state/--use-dataset-images -> episode also feeds obs.
        # ----------------------------------------------------------------------
        self.episode: Optional[LeRobotEpisode] = None
        if use_dataset_state or use_dataset_images:
            if not dataset_path:
                raise ValueError(
                    "--dataset-path is required when --use-dataset-state or "
                    "--use-dataset-images is set."
                )
        if dataset_path:
            self.episode = LeRobotEpisode(
                dataset_path, self.episode_index, image_resolution=img_resolution
            )

        self.config = None
        self._node = None
        self.command_pub = None
        self.joint_feedback_pub = None
        self.robot_joints_sub = None
        self.robot = None
        self.rate_limiter = None
        self.publish_rate = None

        self.latest_robot_joints: Optional[Dict[str, Any]] = None
        self._robot_joints_lock = threading.RLock()
        self._robot_joint_data_initialized = False
        self.running = False

    def _ensure_comm(self):
        # robot packages imported lazily/inline here.
        if self._node is not None:
            return
        from dexcomm import Node
        from dexcomm.utils import RateLimiter
        from dexcomm.codecs import DictDataCodec
        from omniteleop.common import get_config

        self.config = get_config()
        self._node = Node(name="policy_deployer", namespace=self.namespace)

        robot_joints_topic = self.config.get_topic("robot_joints")
        commands_topic = self.config.get_topic("robot_commands")

        self.robot_joints_sub = self._node.create_subscriber(
            robot_joints_topic,
            self._on_robot_joints_received,
            decoder=DictDataCodec.decode,
        )
        self.command_pub = self._node.create_publisher(
            commands_topic, encoder=DictDataCodec.encode
        )
        self.joint_feedback_pub = self._node.create_publisher(
            robot_joints_topic, encoder=DictDataCodec.encode
        )

        self.publish_rate = self.config.get_rate("command_rate", 20)
        self.rate_limiter = RateLimiter(self.publish_rate)

        logger.info(
            f"Subscribed to robot joint feedback: {self._node.resolve_topic(robot_joints_topic)}"
        )
        logger.info(
            f"Publishing commands to: {self._node.resolve_topic(commands_topic)}"
        )
        logger.info(f"Will publish at {self.publish_rate}Hz")

    def _on_robot_joints_received(self, robot_joints_data: Dict[str, Any]) -> None:
        try:
            with self._robot_joints_lock:
                self.latest_robot_joints = robot_joints_data
            if self.latest_robot_joints:
                self._robot_joint_data_initialized = True
        except Exception as e:
            logger.error(f"Error storing robot joints feedback: {e}")

    def _get_home_pose(self) -> Dict[str, Any]:
        home_pose = {}
        for component in ["left_arm", "right_arm", "left_hand", "right_hand", "torso", "head"]:
            pos = self.config.get_init_pos(component)
            if pos is not None:
                home_pose[component] = {"pos": pos}
        return home_pose

    def _initialize_robot_interface(self):
        # dexcontrol imported lazily/inline here.
        if self.robot is not None:
            return
        from dexcontrol.robot import Robot
        from dexcontrol.core.config import get_robot_config

        logger.info("Initializing robot interface for observations...")
        robot_configs = get_robot_config()
        robot_configs.sensors["head_camera"].enabled = True
        robot_configs.sensors["left_wrist_camera"].enabled = True
        robot_configs.sensors["right_wrist_camera"].enabled = True
        self.robot = Robot(configs=robot_configs)

        logger.info("Waiting for camera streams to become active...")
        if self.robot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.success("Camera streams active!")
        else:
            logger.warning("Some camera streams may not be active")

        logger.info("Setting head to fixed position...")
        head_pos = self.config.get_init_pos("head")
        if head_pos:
            self.robot.head.set_joint_pos(np.array(head_pos))
        logger.success("Robot interface initialized")

    def _wait_for_user_start(self):
        logger.info("Press Enter to start policy deployment...")
        input()
        logger.info("Starting policy deployment!")

    def _get_robot_state(self) -> Dict[str, Any]:
        state: Dict[str, np.ndarray] = {}
        with self._robot_joints_lock:
            if self.latest_robot_joints is None:
                return state
            joints_data = self.latest_robot_joints.get("joints", {})
            for component in ["left_arm", "right_arm", "torso", "head", "left_hand", "right_hand"]:
                if component in joints_data:
                    pos_data = joints_data[component]
                    if isinstance(pos_data, dict):
                        pos_data = pos_data.get("pos", [])
                    if hasattr(pos_data, "tolist"):
                        pos_data = pos_data.tolist()
                    state[component] = np.asarray(pos_data, dtype=np.float32).reshape(-1)
        return state

    def _get_camera_images(self) -> Dict[str, Optional[np.ndarray]]:
        import cv2

        processed_images: Dict[str, Optional[np.ndarray]] = {}

        head_obs_keys = ["left_rgb", "right_rgb"]
        image_dict = self.robot.sensors.head_camera.get_obs(obs_keys=head_obs_keys)
        image_dict = {f"head_{key}": value for key, value in image_dict.items()}

        if hasattr(self.robot.sensors, "left_wrist_camera"):
            image_dict["left_wrist_rgb"] = self.robot.sensors.left_wrist_camera.get_obs()
        if hasattr(self.robot.sensors, "right_wrist_camera"):
            image_dict["right_wrist_rgb"] = self.robot.sensors.right_wrist_camera.get_obs()

        for key, value in image_dict.items():
            if value is not None:
                img = value.get("data") if isinstance(value, dict) else value
                if img is not None and self.img_resolution:
                    img = cv2.resize(img, self.img_resolution)
                processed_images[key] = img  # images stay RGB: policy was trained on RGB
        return processed_images

    def _frame_state(self, frame_idx: int) -> Dict[str, np.ndarray]:
        if self.use_dataset_state:
            state_keys = self._dataset_state_keys()
            nested = self.episode.get_state(frame_idx, state_keys)
            return dict(nested.get("joint_pos", {}))
        return self._get_robot_state()

    def _frame_images(self, frame_idx: int) -> Dict[str, np.ndarray]:
        if self.use_dataset_images:
            images = self.episode.get_images(frame_idx, self.video_keys)
        else:
            ##### KEY MAPPING (robot side: raw camera name -> policy video key) #####
            raw = self._get_camera_images()
            images = {}
            for raw_name, policy_key in CAMERA_NAME_MAP.items():
                if policy_key in self.video_keys and raw.get(raw_name) is not None:
                    images[policy_key] = raw[raw_name]
            ##### END KEY MAPPING (robot side: cameras) #####

        ##### KEY MAPPING (robot side: TEMP HACK swap left/right wrist cameras) #####
        _lw, _rw = "left_wrist_rgb_img", "right_wrist_rgb_img"
        if _lw in images and _rw in images:
            images[_lw], images[_rw] = images[_rw], images[_lw]
        ##### END KEY MAPPING (robot side: wrist swap) #####
        return images

    def _dataset_state_keys(self) -> List[str]:
        iface = getattr(self.policy, "_iface", None)
        if iface is not None:
            mc = getattr(iface, "modality_config", None)
            if mc is not None and "state" in mc:
                return list(mc["state"].modality_keys)
        cols = getattr(self.episode, "df", None)
        if cols is not None:
            prefix = "observation.state."
            return [c[len(prefix):] for c in self.episode.df.columns if c.startswith(prefix)]
        return []

    def _dataset_action_keys(self) -> List[str]:
        iface = getattr(self.policy, "_iface", None)
        if iface is not None:
            keys = getattr(iface, "modality_action_keys", None)
            if keys is not None:
                return list(keys)
        cols = getattr(self.episode, "df", None)
        if cols is not None:
            prefix = "action."
            return [c[len(prefix):] for c in self.episode.df.columns if c.startswith(prefix)]
        return []

    def _bringup(self):
        self._wait_for_user_start()
        home_pose = self._get_home_pose()

        logger.info("Step 1: Activating emergency stop to initialize robot controller state...")
        self.command_pub.publish({
            "timestamp_ns": time.time_ns(),
            "components": home_pose,
            "safety_flags": {"emergency_stop": True, "exit_requested": False},
        })
        time.sleep(1.0)

        logger.info("Step 2: Sending home pose command to move robot to initial position...")
        self.command_pub.publish({
            "timestamp_ns": time.time_ns(),
            "components": home_pose,
            "safety_flags": {"emergency_stop": False, "exit_requested": False},
        })
        logger.info("Home pose command sent, waiting 4 seconds for robot to reach position...")
        time.sleep(4.0)

        input("Press Enter to test hands (open for 1s then close)...")
        logger.info("Step 3: Testing hands - opening...")
        self.command_pub.publish({
            "timestamp_ns": time.time_ns(),
            "components": {
                "left_arm": {"pos": home_pose["left_arm"]["pos"]},
                "right_arm": {"pos": home_pose["right_arm"]["pos"]},
                "left_hand": {"pos": [0.78]},
                "right_hand": {"pos": [0.78]},
            },
            "safety_flags": {"emergency_stop": False, "exit_requested": False},
        })
        logger.info("Hands opened, waiting 4 seconds...")
        time.sleep(4.0)
        logger.info("Closing hands...")
        self.command_pub.publish({
            "timestamp_ns": time.time_ns(),
            "components": {
                "left_arm": {"pos": home_pose["left_arm"]["pos"]},
                "right_arm": {"pos": home_pose["right_arm"]["pos"]},
                "left_hand": {"pos": [0.0]},
                "right_hand": {"pos": [0.0]},
            },
            "safety_flags": {"emergency_stop": False, "exit_requested": False},
        })
        logger.info("Hands closed")
        input("Press Enter to move robot using policy...")

    @staticmethod
    def _action_to_components(action_np: np.ndarray) -> Optional[Dict[str, Any]]:
        ##### KEY MAPPING (action -> robot components) #####
        # Split by action width: 14 arms-only, 16 +1-DOF grippers, 26 +6-DOF F5D6 hands.
        action_size = len(action_np)
        if action_size == 14:
            components = {
                "left_arm": {"pos": action_np[0:7].tolist()},
                "right_arm": {"pos": action_np[7:14].tolist()},
            }
        elif action_size == 16:
            components = {
                "left_arm": {"pos": action_np[0:7].tolist()},
                "right_arm": {"pos": action_np[7:14].tolist()},
                "left_hand": {"pos": action_np[14:15].tolist()},
                "right_hand": {"pos": action_np[15:16].tolist()},
            }
        elif action_size == 26:
            components = {
                "left_arm": {"pos": action_np[0:7].tolist()},
                "right_arm": {"pos": action_np[7:14].tolist()},
                "left_hand": {"pos": action_np[14:20].tolist()},
                "right_hand": {"pos": action_np[20:26].tolist()},
            }
        elif action_size == 22:
            # GR00T: full Vega joint vector (arms + torso + head + 1-DOF grippers).
            # Commanding torso/head keeps the head-mounted cameras at the training viewpoint.
            components = {
                "left_arm": {"pos": action_np[0:7].tolist()},
                "right_arm": {"pos": action_np[7:14].tolist()},
                "torso": {"pos": action_np[14:17].tolist()},
                "head": {"pos": action_np[17:20].tolist()},
                "left_hand": {"pos": action_np[20:21].tolist()},
                "right_hand": {"pos": action_np[21:22].tolist()},
            }
        else:
            logger.warning(
                f"Unexpected action size {action_size}, expected 14 (arms), "
                f"16 (gripper), 22 (GR00T Vega) or 26 (F5D6)"
            )
            return None
        ##### END KEY MAPPING (action -> robot components) #####
        return components

    def _publish_action(self, action_np: np.ndarray, iteration: int) -> None:
        components = self._action_to_components(action_np)
        if not components:
            return
        self.command_pub.publish({
            "timestamp_ns": time.time_ns(),
            "components": components,
            "safety_flags": {"emergency_stop": False, "exit_requested": False},
        })
        if self.debug:
            logger.info(f"[{iteration:4d}] Published action_size={len(action_np)}")

    ##### GR00T SAFE RAMP #####
    _RAMP_ORDER = ["left_arm", "right_arm", "torso", "head", "left_hand", "right_hand"]

    def _assemble_22(self, comp: Dict[str, Any]) -> Optional[np.ndarray]:
        """Concat 6 joint components -> 22-D vector (action layout), or None if incomplete."""
        parts = []
        for c in self._RAMP_ORDER:
            if c not in comp:
                return None
            parts.append(np.asarray(comp[c], dtype=np.float32).reshape(-1))
        v = np.concatenate(parts)
        return v if v.shape[0] == 22 else None

    def _ramp_to_episode_start(self) -> None:
        """Smoothly move the arm from its current pose to the episode's first state."""
        target = self._assemble_22(self.episode.get_state(0, self._dataset_state_keys())["joint_pos"])
        if target is None:
            logger.warning("Ramp skipped: could not assemble 22-D target from dataset frame 0.")
            return
        # current pose: prefer live robot feedback, else fall back to the home pose
        t0 = time.time()
        while not self._robot_joint_data_initialized and time.time() - t0 < 3.0:
            time.sleep(0.1)
        current = self._assemble_22(self._get_robot_state())
        if current is None:
            hp = self._get_home_pose()
            current = self._assemble_22({c: v["pos"] for c, v in hp.items()})
        if current is None:
            logger.warning("Ramp skipped: no current robot pose (joint feedback or home pose).")
            return
        move = float(np.abs(target - current).max())
        logger.info(f"Ramping to episode start over {self.ramp_steps} steps "
                    f"(max joint move {move:.3f} rad)...")
        for i in range(1, self.ramp_steps + 1):
            interp = current + (target - current) * (i / self.ramp_steps)
            self._publish_action(interp.astype(np.float32), -1)
            if self.rate_limiter is not None:
                self.rate_limiter.sleep()
        time.sleep(0.5)
        logger.success("At episode start pose; beginning policy rollout.")
    ##### END GR00T SAFE RAMP #####

    def run(self) -> int:
        self.running = True

        needs_robot = (
            not self.use_dataset_state
            or not self.use_dataset_images
            or self.publish_actions
        )

        if needs_robot:
            self._ensure_comm()
            if not self.use_dataset_state or not self.use_dataset_images:
                self._initialize_robot_interface()
                logger.info("Waiting for robot joint data from robot_controller...")
                while not self._robot_joint_data_initialized:
                    time.sleep(0.1)
                logger.info("Robot joint data received!")
            if self.publish_actions:
                self._bringup()
                ##### GR00T SAFE RAMP (opt-in: --ramp-steps N) #####
                # EDIT 2: Gate the ramp on having an EPISODE, not on dataset-observation
                # mode. This lets the ramp run in LIVE rollout too: pass --dataset-path
                # + --ramp-steps WITHOUT --use-dataset-state/--use-dataset-images to move
                # to the demo's frame-0 pose, then hand control to the policy on live obs.
                # The ramp target (episode.get_state(0)) reads the parquet only -> no
                # video decode, so the live+ramp path does NOT require torchcodec.
                if self.ramp_steps > 0 and self.episode is not None:
                    self._ramp_to_episode_start()
                ##### END GR00T SAFE RAMP #####
        else:
            logger.info("Pure dataset/eval mode: no robot packages needed")

        if self.use_dataset_state or self.use_dataset_images:
            n = self.episode.num_frames
        else:
            n = 10_000_000

        # EDIT 3: cap rollout length if --max-steps was given (0 = unlimited).
        if self.max_steps > 0:
            n = min(n, self.max_steps)
            logger.info(f"max_steps set: rollout capped at {n} frames.")

        have_gt = self.use_dataset_state or self.use_dataset_images
        action_keys = self._dataset_action_keys() if have_gt else []

        logger.info(f"Starting policy inference loop ({n} frames)...")
        pred_actions: List[np.ndarray] = []
        gt_actions: List[np.ndarray] = []

        try:
            iteration = 0
            while self.running and iteration < n:
                state = self._frame_state(iteration)
                images = self._frame_images(iteration)

                action = self.policy.predict_action(state, images, self.task)
                action_np = np.asarray(action, dtype=np.float32).reshape(-1)
                pred_actions.append(action_np)

                if have_gt:
                    gt_actions.append(self.episode.get_gt_action(iteration, action_keys))

                if self.publish_actions:
                    self._publish_action(action_np, iteration)

                iteration += 1
                if self.rate_limiter is not None:
                    self.rate_limiter.sleep()
        finally:
            self.stop()

        logger.info(f"Loop completed; produced {len(pred_actions)} actions")

        if have_gt and pred_actions:
            self._dump_metrics(pred_actions, gt_actions)
        elif pred_actions:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(self.output_dir) / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run_dir / "traces.npz", pred_actions=np.stack(pred_actions))
            logger.info(f"No ground truth (live obs); predictions saved to {run_dir}")
        return 0

    def _dump_metrics(self, pred_actions: List[np.ndarray], gt_actions: List[np.ndarray]) -> None:
        preds = np.stack(pred_actions, axis=0)
        gts = np.stack(gt_actions, axis=0)
        m = min(preds.shape[1], gts.shape[1])
        preds_m, gts_m = preds[:, :m], gts[:, :m]
        metrics = compute_metrics(preds_m, gts_m)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.output_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        stats = dict(metrics)
        stats.update({
            "episode_index": self.episode_index,
            "dataset_path": self.dataset_path,
            "n_frames": int(preds.shape[0]),
        })
        with open(run_dir / "stats.json", "w") as fp:
            json.dump(stats, fp, indent=2)

        np.savez_compressed(
            run_dir / "traces.npz",
            pred_actions=preds_m,
            gt_actions=gts_m,
            per_frame_l1=np.asarray(metrics["l1"]["per_frame"], dtype=np.float64),
        )
        _plot_actions(preds_m, gts_m, run_dir / "actions.png")

        print("=" * 60)
        print("DATASET EVAL SUMMARY")
        print("=" * 60)
        print(f"Frames evaluated : {metrics['frames']}")
        print(f"L1 mean / std    : {metrics['l1']['mean']:.6f} / {metrics['l1']['std']:.6f}")
        print(f"L1 min / max     : {metrics['l1']['min']:.6f} / {metrics['l1']['max']:.6f}")
        print(f"MSE              : {metrics['mse']:.6f}")
        print(f"Output           : {run_dir}")
        print("=" * 60)

    def stop(self):
        """Stop the loop and clean up robot infra."""
        self.running = False
        if self.robot is not None:
            try:
                self.robot.shutdown()
            except Exception as e:
                logger.warning(f"Robot shutdown error: {e}")
            self.robot = None
        if self._node is not None:
            try:
                self._node.shutdown()
            except Exception as e:
                logger.warning(f"Node shutdown error: {e}")
            self._node = None


def main(
    pretrained_path: Optional[str] = None,
    task: str = "place the parts in slot",
    device: str = "cuda:0",
    action_chunk_threshold: float = 1.0,
    use_dataset_state: bool = False,
    use_dataset_images: bool = False,
    dataset_path: Optional[str] = None,
    episode_index: int = 0,
    publish_actions: bool = False,
    namespace: str = "",
    action_mode: str = "absolute",
    image_resolution: Optional[Tuple[int, int]] = None,
    seed: int = 42,
    debug: bool = False,
    ramp_steps: int = 0,
    max_steps: int = 0,
) -> int:
    """Policy-agnostic deploy / replay / eval entry point."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        from omniteleop.common.logging import setup_logging
        setup_logging(debug)
    except Exception:
        pass

    if not pretrained_path:
        logger.error("Error: --pretrained-path is required")
        return 1

    policy_name = auto_detect_policy(pretrained_path)
    logger.info(f"Policy: {policy_name}")
    PolCls = get_policy(policy_name)
    loaded_policy = PolCls.load(
        pretrained_path,
        device,
        action_chunk_threshold=action_chunk_threshold,
    )
    loaded_policy.reset(task)
    logger.success("Policy loaded successfully!")

    img_resolution = tuple(image_resolution) if image_resolution is not None else None

    deployer = Deployer(
        loaded_policy,
        task,
        use_dataset_state=use_dataset_state,
        use_dataset_images=use_dataset_images,
        publish_actions=publish_actions,
        dataset_path=dataset_path,
        episode_index=episode_index,
        namespace=namespace,
        action_mode=action_mode,
        img_resolution=img_resolution,
        debug=debug,
        ramp_steps=ramp_steps,
        max_steps=max_steps,
    )
    return deployer.run()


if __name__ == "__main__":
    sys.exit(tyro.cli(main))