"""
OpenVLA (Open Vision-Language-Action) Inference Script
======================================================

This script demonstrates how to load the OpenVLA-7B model and run
robot-action predictions on both Intel XPU and CPU devices.

It includes:
  1. Model and processor loading
  2. A smoke-test with a dummy image
  3. Inference on both XPU and CPU
  4. An end-to-end + model-only latency benchmark

Prerequisites:
  pip install transformers==4.40.1 timm torch pillow numpy
  (See requirements.txt for the full list)
"""

# ─────────────────────��────���───────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────────────
import time
import statistics as stats

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# ──────────────────────────────────────────────────────────────────────
# 2. Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_ID = "openvla/openvla-7b"
XPU_DEVICE = "xpu:0"
CPU_DEVICE = "cpu"
N_WARMUP = 5       # warm-up iterations before benchmarking
N_ITERS = 30       # timed iterations for the benchmark

# ──────────────────────────────────────────────────────────────────────
# 3. Helper functions
# ──────────────────────────────────────────────────────────────────────

def move_inputs_to_device(inputs, device, dtype):
    """Move every tensor in *inputs* to *device*, casting floats to *dtype*.

    NOTE: 'attention_mask' is intentionally dropped.
    In newer transformers (> 4.40.1), generate() grows the attention_mask by 1
    per decoded token during cached generation. Prismatic's forward() builds
    its own multimodal mask for the first pass, but on subsequent cached decode
    steps the growing mask from generate() goes out of sync with the KV-cache
    length, causing:
      RuntimeError: The size of tensor a (283) must match tensor b (282) at dim 3
    Dropping the mask lets LLaMA use its internal causal masking, which is
    correct for all decode steps regardless of transformers version.
    """
    out = {}
    for k, v in inputs.items():
        if k == "attention_mask":
            continue  # Skip — causes off-by-one crash in cached decode
        if torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def sync_xpu():
    """Block until all kernels on the XPU have completed."""
    torch.xpu.synchronize()


def summarize(name, times):
    """Print mean / p50 / p90 latency and throughput for a list of timings."""
    mean_s = stats.mean(times)
    p50_s = stats.median(times)
    p90_s = sorted(times)[int(0.9 * len(times)) - 1]
    ips = 1.0 / mean_s
    print(f"\n{name}")
    print(f"  mean latency : {mean_s * 1000:.2f} ms")
    print(f"  p50  latency : {p50_s * 1000:.2f} ms")
    print(f"  p90  latency : {p90_s * 1000:.2f} ms")
    print(f"  throughput   : {ips:.3f} images/s")


def robot_act(action):
    """Placeholder for a real robot driver – just prints the action."""
    print("robot.act(action) would receive:", action)


def run_inference(vla, processor, prompt, image, device, dtype):
    """Run a single action-prediction pass and print results."""
    # Process inputs (tokenize prompt + image)
    inputs = processor(prompt, image)
    inputs = move_inputs_to_device(inputs, device, dtype)

    # Predict a 7-DoF action (un-normalized for BridgeData V2)
    with torch.inference_mode():
        action = vla.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False,
        )

    # Report
    print("prediction type:", type(action))
    print("prediction:", action)
    try:
        print("shape:", action.shape)
    except AttributeError:
        pass

    robot_act(action)
    print("SUCCESS: model load -> preprocess -> predict_action -> consume action finished.")
    return action

# ──────────────────────────────────────────────────────────────────────
# 4. Main entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    # ------------------------------------------------------------------
    # 4a. Load processor (tokenizer + image transforms)
    # ------------------------------------------------------------------
    print("[INFO] Loading processor …")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ------------------------------------------------------------------
    # 4b. Load model onto the XPU
    # ------------------------------------------------------------------
    print(f"[INFO] Loading model onto {XPU_DEVICE} …")
    vla_xpu = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(XPU_DEVICE)

    # Print VRAM usage
    vram_gb = torch.xpu.memory_allocated(0) / 1024 ** 3
    print(f"  VRAM used on {XPU_DEVICE} → {vram_gb:.1f} GB")

    # ------------------------------------------------------------------
    # 4c. Create a dummy RGB image (224×224) for smoke testing
    # ------------------------------------------------------------------
    H, W = 224, 224
    dummy = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
    image = Image.fromarray(dummy).convert("RGB")

    # Instruction prompt (BridgeData V2 style)
    instruction = "pick up the red block and place it on the green block"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # ------------------------------------------------------------------
    # 4d. Inference on XPU
    # ------------------------------------------------------------------
    xpu_dtype = torch.bfloat16 if torch.xpu.is_bf16_supported() else torch.float16

    print(f"\n{'=' * 60}")
    print(f" XPU Inference ({XPU_DEVICE})")
    print(f"{'=' * 60}")
    run_inference(vla_xpu, processor, prompt, image, XPU_DEVICE, xpu_dtype)

    # ------------------------------------------------------------------
    # 4e. Load model on CPU and run inference there
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading model onto CPU …")
    vla_cpu = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(CPU_DEVICE)

    cpu_dtype = torch.bfloat16 if torch.xpu.is_bf16_supported() else torch.float16

    print(f"\n{'=' * 60}")
    print(f" CPU Inference")
    print(f"{'=' * 60}")
    run_inference(vla_cpu, processor, prompt, image, CPU_DEVICE, cpu_dtype)

    # ------------------------------------------------------------------
    # 4f. Latency benchmark (XPU only)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f" Latency Benchmark on {XPU_DEVICE}  (warmup={N_WARMUP}, iters={N_ITERS})")
    print(f"{'=' * 60}")

    # Re-create a fresh dummy image for the benchmark
    dummy = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(dummy).convert("RGB")

    # Warm-up: run a few iterations so JIT/caches are primed
    for _ in range(N_WARMUP):
        inputs = processor(prompt, image)
        inputs = move_inputs_to_device(inputs, XPU_DEVICE, xpu_dtype)
        with torch.inference_mode():
            _ = vla_xpu.predict_action(
                **inputs, unnorm_key="bridge_orig", do_sample=False,
            )
        sync_xpu()

    # End-to-end benchmark (processor + host→device + predict_action)
    e2e_times = []
    for _ in range(N_ITERS):
        sync_xpu()
        t0 = time.perf_counter()

        inputs = processor(prompt, image)
        inputs = move_inputs_to_device(inputs, XPU_DEVICE, xpu_dtype)
        with torch.inference_mode():
            _ = vla_xpu.predict_action(
                **inputs, unnorm_key="bridge_orig", do_sample=False,
            )

        sync_xpu()
        t1 = time.perf_counter()
        e2e_times.append(t1 - t0)

    # Model-only benchmark (pre-process once, measure only predict_action)
    inputs = processor(prompt, image)
    inputs = move_inputs_to_device(inputs, XPU_DEVICE, xpu_dtype)

    model_times = []
    for _ in range(N_ITERS):
        sync_xpu()
        t0 = time.perf_counter()

        with torch.inference_mode():
            _ = vla_xpu.predict_action(
                **inputs, unnorm_key="bridge_orig", do_sample=False,
            )

        sync_xpu()
        t1 = time.perf_counter()
        model_times.append(t1 - t0)

    # Print results
    summarize("End-to-end (including image processing)", e2e_times)
    summarize("Model-only", model_times)


if __name__ == "__main__":
    main()
