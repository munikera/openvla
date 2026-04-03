"""
libero_env_worker.py

Runs the LIBERO environment in an isolated subprocess (no torch/MKL loaded),
communicating with the parent process via stdin/stdout using pickle over base64.

This avoids a segfault caused by MKL-SYCL libraries (loaded by torch at import
time) conflicting with MuJoCo's osmesa memory allocator.
"""

import base64
import os
import pickle
import sys


def main():
    # Force osmesa BEFORE any mujoco/libero/OpenGL import — must happen first
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    # Unset any EGL vars that might override osmesa
    os.environ.pop("EGL_DEVICE_ID", None)
    os.environ.pop("DISPLAY", None)

    # Redirect stdout to stderr so libero's print() calls don't corrupt the pipe protocol.
    # All our base64 messages use the original stdout fd saved below.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr

    # Redefine send/recv to use the real stdout fd
    def _send(obj):
        data = base64.b64encode(pickle.dumps(obj)).decode("ascii")
        _real_stdout.write(data + "\n")
        _real_stdout.flush()

    def _recv():
        line = sys.stdin.readline()
        if not line:
            return None
        return pickle.loads(base64.b64decode(line.strip()))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    _send({"status": "ready"})

    while True:
        msg = _recv()
        if msg is None or msg.get("cmd") == "exit":
            break

        cmd = msg["cmd"]

        if cmd == "make_env":
            task_suite_name = msg["task_suite_name"]
            task_id = msg["task_id"]
            resolution = msg.get("resolution", 256)

            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[task_suite_name]()
            task = task_suite.get_task(task_id)
            task_description = task.language

            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=resolution,
                camera_widths=resolution,
                render_gpu_device_id=-1,
            )
            env.seed(0)

            # Cache initial states
            initial_states = task_suite.get_task_init_states(task_id)

            _send({"status": "ok", "task_description": task_description,
                   "n_initial_states": len(initial_states)})

        elif cmd == "reset":
            env.reset()
            episode_idx = msg["episode_idx"]
            obs = env.set_init_state(initial_states[episode_idx])
            _send({"status": "ok", "obs": obs})

        elif cmd == "step":
            action = msg["action"]
            obs, reward, done, info = env.step(action)
            _send({"status": "ok", "obs": obs, "reward": reward, "done": done, "info": info})

        elif cmd == "close":
            env.close()
            _send({"status": "ok"})

        elif cmd == "get_num_tasks":
            task_suite_name = msg["task_suite_name"]
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[task_suite_name]()
            _send({"status": "ok", "num_tasks": task_suite.n_tasks})


if __name__ == "__main__":
    main()
