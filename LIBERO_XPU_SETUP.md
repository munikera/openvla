# OpenVLA LIBERO Evaluation on Intel Arc B580 — Setup & Bug Fix Journal

## Goal
Run `run_libero_eval.py` end-to-end on a headless Ubuntu 24.04 server with two Intel Arc B580
XPUs (no display, no CUDA), producing LIBERO spatial success-rate metrics.

**Final command that works:**
```bash
cd /home/devcloud/munikera/openvla && \
PYTHONPATH=/home/devcloud/munikera/openvla \
ZE_AFFINITY_MASK=0 \
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 5
```

---

## System

| Item | Value |
|---|---|
| Server | `aics-hf2-ws-14`, Ubuntu 24.04 |
| Conda env | `openvla-xpu`, Python 3.12 |
| Hardware | 2× Intel Arc B580 |
| ZE_AFFINITY_MASK | `0` = main process (model inference), `1` = worker subprocess (MuJoCo env) |
| Working dir | `/home/devcloud/munikera/openvla` |
| Checkpoint | `openvla/openvla-7b-finetuned-libero-spatial` (HuggingFace) |
| MuJoCo | 3.1.6 (osmesa renderer) |

---

## Problems Encountered & Fixes Applied

### 1. XPU Support in `openvla_utils.py`
**Problem:** Code assumed CUDA; no XPU path existed.

**Fix — `experiments/robot/openvla_utils.py`:**
- Detect XPU: `_XPU = torch.xpu.is_available()`
- `get_vla()`: use `attn_implementation="eager"` (Flash-Attention not supported on XPU), skip 8/4-bit quantization
- `get_vla_action()`: drop `attention_mask` kwarg when running on XPU

---

### 2. `ModuleNotFoundError: No module named 'dlimp'`
**Problem:** `import prismatic` triggered the full training pipeline import chain, which requires `dlimp` (not needed for eval).

**Fix — `prismatic/__init__.py`:**
- Replaced eager imports with a `__getattr__` lazy loader so eval scripts don't pull in the training pipeline.

---

### 3. LIBERO Installation (sequential errors)

```bash
pip install cmake
pip install robosuite==1.4.0 --no-deps
pip install tensorflow-cpu
pip install matplotlib einops gymnasium hydra-core
sudo apt install libgl1-mesa-dev libosmesa6-dev
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

---

### 4. `AssertionError: unnorm_key libero_spatial not found`
**Problem:** Base `openvla-7b` checkpoint has no LIBERO normalization stats. Script defaulted to `task_suite_name` as key.

**Fix — `run_libero_eval.py`:**
- Added `unnorm_key: Optional[str] = None` to `GenerateConfig`
- Used fine-tuned checkpoint `openvla/openvla-7b-finetuned-libero-spatial` which contains the correct stats

---

### 5. LLVM Duplicate-Symbol Abort
**Problem:** `import torch` after `import tensorflow` caused a fatal LLVM abort due to duplicate symbol registration between the two frameworks' bundled LLVM libs.

**Error:**
```
LLVM ERROR: symbol 'LLVMContextCreate' is already defined
Aborted (core dumped)
```

**Fix — `run_libero_eval.py` (top of file):**
```python
import tensorflow as _tf  # MUST come before torch — fixes LLVM duplicate-symbol abort
```

---

### 6. MuJoCo osmesa Segfault (AVX-512 null-pointer)
**Problem:** MuJoCo 3.6.0 crashed with a segfault in `libc.so.6 memcpy` using AVX-512 on a null pointer when osmesa was active.

**Fix:** Downgrade to MuJoCo 3.1.6:
```bash
pip install mujoco==3.1.6
```
osmesa rendering confirmed working at 3.1.6.

---

### 7. MKL-SYCL + osmesa Segfault in Same Process
**Problem:** `import torch` loads ~41 MKL/SYCL shared libraries. These conflict with MuJoCo's osmesa memory allocator when both run in the same process, causing a segfault.

**Root cause confirmed:** Isolation test showed worker subprocess alone (no torch) renders fine; adding `import torch` to the same process causes the crash.

**Fix:** Subprocess isolation architecture — LIBERO env runs in a clean subprocess with no torch/MKL loaded.

---

### 8. Worker Subprocess stderr Deadlock
**Problem:** `stderr=subprocess.PIPE` caused the worker to deadlock when its stderr buffer filled.

**Fix:**
```python
stderr=None  # inherit terminal stderr — worker warnings print directly to terminal
```

---

### 9. OOM Kill During Full Eval
**Problem:** Both main process (7B model) and worker subprocess (MuJoCo env) were competing for the same GPU card, exhausting VRAM → OOM killed.

**Fix:** Pin each process to a different card:
```bash
# Main process (model inference)
ZE_AFFINITY_MASK=0

# Worker subprocess (MuJoCo env)  
clean_env["ZE_AFFINITY_MASK"] = "1"
```

---

### 10. libero/robosuite stdout Corrupts Pipe Protocol
**Problem:** The subprocess pipe used base64-encoded pickle over stdout. But libero and robosuite print info messages directly to stdout:
```
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Local assets not found. Downloading from HuggingFace Hub...
Assets already downloaded at /home/devcloud/.cache/libero/assets
```
These non-base64 lines caused `base64.b64decode` to fail with `binascii.Error`.

**Fix — `libero_env_worker.py` `main()` function:**
```python
# Save real stdout before libero imports
_real_stdout = sys.stdout
# Redirect sys.stdout → stderr so libero print() calls don't corrupt the pipe
sys.stdout = sys.stderr

# Local send/recv using the real stdout fd
def _send(obj):
    data = base64.b64encode(pickle.dumps(obj)).decode("ascii")
    _real_stdout.write(data + "\n")
    _real_stdout.flush()

def _recv():
    line = sys.stdin.readline()   # stdin is unchanged (only stdout was redirected)
    if not line:
        return None
    return pickle.loads(base64.b64decode(line.strip()))
```

---

### 11. `from libero.libero import benchmark` at Top Level (Final Hang)
**Problem:** `run_libero_eval.py` had a top-level import of `libero.libero.benchmark`, which pulled in `robosuite` → `OpenGL` → tried EGL/DRI2 initialization **in the main process** (which already had torch/MKL-SYCL loaded). This caused the process to hang trying to open `/dev/dri/card1`.

**Error:**
```
MESA: warning: Driver does not support the 0xe223 PCI ID.
libEGL warning: egl: failed to create dri2 screen
libEGL warning: failed to open /dev/dri/card1: Permission denied
```

**Fix:**

`run_libero_eval.py`:
- Removed `from libero.libero import benchmark` top-level import
- `get_num_tasks()` now spawns a **temporary worker subprocess** to query task count — same clean isolation as `LiberoEnvProxy`

`libero_utils.py`:
- Moved `from libero.libero import get_libero_path` and `from libero.libero.envs import OffScreenRenderEnv` **inside** `get_libero_env()` as lazy imports — importing the module no longer loads robosuite/OpenGL

---

## Final Architecture

```
Main Process (ZE_AFFINITY_MASK=0)
├── tensorflow (imported first — LLVM fix)
├── torch (XPU, card 0)
├── OpenVLA model (BF16, eager attention)
└── LiberoEnvProxy (stdin/stdout pipe)
        │
        └── Worker Subprocess (ZE_AFFINITY_MASK=1)
            ├── MUJOCO_GL=osmesa
            ├── PYOPENGL_PLATFORM=osmesa
            ├── NO torch / NO MKL-SYCL
            ├── libero + robosuite
            └── MuJoCo 3.1.6 osmesa renderer
                sys.stdout → sys.stderr (suppress libero prints)
                base64+pickle protocol on real stdout fd
```

### Communication Protocol
```
Parent stdin  →→→→→→→→→→→→→→→→→→→  Worker sys.stdin
                base64(pickle(cmd))

Worker _real_stdout  →→→→→→→→→→→  Parent stdout pipe  
                base64(pickle(response))
```

---

## Files Modified

| File | Changes |
|---|---|
| `experiments/robot/openvla_utils.py` | XPU detection, eager attention, drop attention_mask on XPU |
| `prismatic/__init__.py` | Lazy `__getattr__` to avoid loading training pipeline (dlimp) on eval |
| `experiments/robot/libero/libero_utils.py` | Lazy imports of `libero.libero` inside `get_libero_env()` |
| `experiments/robot/libero/run_libero_eval.py` | TF-before-torch import; removed top-level libero import; `LiberoEnvProxy` class; `get_num_tasks()` via subprocess; `unnorm_key` config field |
| `experiments/robot/libero/libero_env_worker.py` | **NEW FILE** — isolated subprocess worker for LIBERO env |

---

## `libero_env_worker.py` Overview

New file: `experiments/robot/libero/libero_env_worker.py`

Handles commands over stdin/stdout base64+pickle pipe:

| Command | Description |
|---|---|
| `make_env` | Create LIBERO env for given `task_suite_name` + `task_id` |
| `reset` | Reset env to `episode_idx` initial state, return obs |
| `step` | Step env with action, return obs/reward/done/info |
| `close` | Close env |
| `get_num_tasks` | Return number of tasks in a suite (used by `get_num_tasks()`) |
| `exit` | Terminate worker |

---

## Expected Results

From OpenVLA paper (Table 12), `libero_spatial` with fine-tuned checkpoint:

| Suite | Expected Success Rate |
|---|---|
| libero_spatial | ~84.7% |
| libero_object | ~88.4% |
| libero_goal | ~79.2% |
| libero_10 | ~84.5% |
