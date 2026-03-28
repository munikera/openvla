"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# ── Import tensorflow BEFORE torch to avoid LLVM duplicate-symbol abort ─────────
import tensorflow as _tf  # noqa: F401
# ────────────────────────────────────────────────────────────────────────────────

import draccus
import numpy as np
import tqdm

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# ── Subprocess-based LIBERO env proxy (avoids MKL-SYCL vs osmesa segfault) ──────
import base64
import pickle
import subprocess
import time


class LiberoEnvProxy:
    """Wraps the LIBERO env running in a clean subprocess (no torch/MKL loaded)."""

    def __init__(self, task_suite_name: str, task_id: int, resolution: int = 256):
        worker = Path(__file__).parent / "libero_env_worker.py"
        # Build a clean env: inherit everything EXCEPT GL-related vars which we force to osmesa
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_DEVICE_ID", "DISPLAY")}
        clean_env["MUJOCO_GL"] = "osmesa"
        clean_env["PYOPENGL_PLATFORM"] = "osmesa"
        clean_env["ZE_AFFINITY_MASK"] = "1"  # pin worker to card 1, leaving card 0 for model inference
        self._proc = subprocess.Popen(
            [sys.executable, str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit terminal stderr so worker warnings are visible and don't block
            env=clean_env,
            text=True,
        )
        resp = self._recv()
        assert resp["status"] == "ready"

        self._send({"cmd": "make_env", "task_suite_name": task_suite_name,
                    "task_id": task_id, "resolution": resolution})
        resp = self._recv()
        assert resp["status"] == "ok"
        self.task_description = resp["task_description"]
        self.n_initial_states = resp["n_initial_states"]

    def _send(self, obj):
        data = base64.b64encode(pickle.dumps(obj)).decode("ascii")
        self._proc.stdin.write(data + "\n")
        self._proc.stdin.flush()

    def _recv(self):
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Worker subprocess died unexpectedly — check stderr output above.")
        return pickle.loads(base64.b64decode(line.strip()))

    def reset(self, episode_idx: int):
        self._send({"cmd": "reset", "episode_idx": episode_idx})
        return self._recv()["obs"]

    def step(self, action):
        self._send({"cmd": "step", "action": action})
        r = self._recv()
        return r["obs"], r["reward"], r["done"], r["info"]

    def close(self):
        self._send({"cmd": "exit"})
        self._proc.wait()


def get_num_tasks(task_suite_name: str) -> int:
    """Ask a temporary worker subprocess for the task count (avoids loading libero into main process)."""
    worker = Path(__file__).parent / "libero_env_worker.py"
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_DEVICE_ID", "DISPLAY")}
    clean_env["MUJOCO_GL"] = "osmesa"
    clean_env["PYOPENGL_PLATFORM"] = "osmesa"
    clean_env["ZE_AFFINITY_MASK"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(worker)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        env=clean_env, text=True,
    )
    # wait for ready
    resp = pickle.loads(base64.b64decode(proc.stdout.readline().strip()))
    assert resp["status"] == "ready"
    # query num tasks
    proc.stdin.write(base64.b64encode(pickle.dumps(
        {"cmd": "get_num_tasks", "task_suite_name": task_suite_name}
    )).decode() + "\n")
    proc.stdin.flush()
    resp = pickle.loads(base64.b64decode(proc.stdout.readline().strip()))
    proc.stdin.write(base64.b64encode(pickle.dumps({"cmd": "exit"})).decode() + "\n")
    proc.stdin.flush()
    proc.wait()
    return resp["num_tasks"]
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    unnorm_key: Optional[str] = None                 # Action un-normalization key override. Defaults to task_suite_name.
                                                     # Use "bridge_orig" with the base openvla/openvla-7b checkpoint
                                                     # (which has no LIBERO stats). A LIBERO fine-tuned checkpoint
                                                     # will have the task suite name as the key automatically.

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key (use override if provided, else default to task suite name)
    cfg.unnorm_key = cfg.unnorm_key if cfg.unnorm_key is not None else cfg.task_suite_name

    # Get number of tasks (without loading env)
    num_tasks_in_suite = get_num_tasks(cfg.task_suite_name)
    print(f"Task suite: {cfg.task_suite_name} ({num_tasks_in_suite} tasks)")

    # Load model (torch/MKL-SYCL stay in this process; envs run in clean subprocesses)
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    per_task_results = []  # list of (task_description, successes, episodes)
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Initialize LIBERO environment in a clean subprocess (no MKL-SYCL conflict with osmesa)
        env = LiberoEnvProxy(cfg.task_suite_name, task_id, resolution=256)
        task_description = env.task_description

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_steps = 0  # total inference steps (excluding wait steps) across all episodes in this task
        inference_times = []  # track per-step inference latency (seconds)
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment and set initial state
            obs = env.reset(episode_idx)

            # Setup
            t = 0
            episode_steps = 0  # inference steps this episode (excludes num_steps_wait)
            replay_images = []
            done = False
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    t0 = time.perf_counter()
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )
                    inference_times.append(time.perf_counter() - t0)

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    episode_steps += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            task_steps += episode_steps
            _cur_avg_inf = np.mean(inference_times) if inference_times else 0.0
            ep_inf_s = episode_steps * _cur_avg_inf

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}  |  steps: {episode_steps}  |  est. inf. time: {ep_inf_s:.1f}s")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done} | steps: {episode_steps} | est_inf_time: {ep_inf_s:.1f}s\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Close env subprocess before moving to next task
        env.close()

        per_task_results.append((task_description, task_successes, task_episodes, task_steps))

        # Log final results
        task_sr = float(task_successes) / float(task_episodes)
        total_sr = float(total_successes) / float(total_episodes)
        avg_task_steps = task_steps / task_episodes
        task_inf_s = task_steps * (np.mean(inference_times) if inference_times else 0.0)
        print(f"\n[Task {task_id+1}/{num_tasks_in_suite}] '{task_description}'")
        print(f"  Success rate      : {task_successes}/{task_episodes} = {task_sr*100:.1f}%")
        print(f"  Avg steps/episode : {avg_task_steps:.1f}  |  Total steps: {task_steps}")
        print(f"  Est. inf. time    : {task_inf_s:.1f}s  ({task_inf_s/task_episodes:.1f}s avg/episode)")
        print(f"  Running total     : {total_successes}/{total_episodes} = {total_sr*100:.1f}%")
        log_file.write(f"Current task success rate: {task_sr}\n")
        log_file.write(f"Current total success rate: {total_sr}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    # ── Final benchmark summary table ────────────────────────────────────────────
    avg_inf = np.mean(inference_times) if inference_times else 0.0
    inf_hz  = 1.0 / avg_inf if avg_inf > 0 else 0.0
    total_steps = sum(s for _, _, _, s in per_task_results)
    total_inf_s = total_steps * avg_inf
    summary_lines = [
        "",
        "=" * 90,
        f"  BENCHMARK RESULTS — {cfg.task_suite_name}",
        f"  Checkpoint : {cfg.pretrained_checkpoint}",
        f"  Inference  : {avg_inf*1000:.1f} ms/step  →  {inf_hz:.2f} Hz  (steps/sec)",
        f"  Total steps: {total_steps}  |  Est. total inference time: {total_inf_s:.0f}s  ({total_inf_s/60:.1f} min)",
        "=" * 90,
        f"  {'Task':<45} {'SR':>7} {'Succ':>5} {'Eps':>5} {'Steps':>7} {'Steps/Ep':>9} {'InfTime':>9}",
        "-" * 90,
    ]
    for desc, succ, eps, steps in per_task_results:
        sr = succ / eps * 100
        avg_steps = steps / eps
        inf_s = steps * avg_inf
        summary_lines.append(
            f"  {desc:<45} {sr:>6.1f}% {succ:>5} {eps:>5} {steps:>7} {avg_steps:>9.1f} {inf_s:>8.1f}s"
        )
    summary_lines += [
        "-" * 90,
        f"  {'TOTAL':<45} {total_successes/total_episodes*100:>6.1f}% {total_successes:>5} "
        f"{total_episodes:>5} {total_steps:>7} {total_steps/total_episodes:>9.1f} {total_inf_s:>8.1f}s",
        "=" * 90,
        "",
    ]
    summary = "\n".join(summary_lines)
    print(summary)
    log_file.write(summary + "\n")
    # ─────────────────────────────────────────────────────────────────────────────
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
