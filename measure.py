import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import pynvml
import torch

from models import model


BASE_S = [32, 64, 128, 224, 256, 384, 512, 1024]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
REGIME_POINTS = [(s, b) for s in [32, 64, 128] for b in [3, 6, 12, 24, 48, 96]]
OOM_POINTS = (
    [(1024, b) for b in [560, 576, 584, 585, 586, 587, 588, 592, 600, 640, 768]]
    + [(512, b) for b in [2048, 2304, 2336, 2368, 2432, 2560]]
    + [(1536, b) for b in [256, 272, 320]]
    + [(2048, b) for b in [128, 160, 256]]
)
EXTRA_POINTS = REGIME_POINTS + OOM_POINTS
SEED, WARMUP, REPEATS, ENERGY_SECONDS = 42, 10, 30, 3


def make_grid(seed=SEED):
    rng = np.random.default_rng(seed)
    sizes = BASE_S + rng.choice([s for s in range(32, 513, 16) if s not in BASE_S], 4, replace=False).tolist()
    batches = BASE_B + rng.choice([b for b in range(1, 257) if b not in BASE_B], 3, replace=False).tolist()
    grid = [(s, b) for s in sizes for b in batches] + EXTRA_POINTS
    rng.shuffle(grid)
    return grid


def measure_energy(x, handle):
    def power():
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000

    end = time.perf_counter() + 1
    while time.perf_counter() < end:
        model(x)
        torch.cuda.synchronize()
    previous_power = power()
    start = previous_time = time.perf_counter()
    total, repeats = 0.0, 0
    while True:
        model(x)
        torch.cuda.synchronize()
        repeats += 1
        now = time.perf_counter()
        finished = now - start >= ENERGY_SECONDS
        if now - previous_time >= 0.05 or finished:
            current_power = power()
            now = time.perf_counter()
            total += (previous_power + current_power) / 2 * (now - previous_time)
            previous_power, previous_time = current_power, now
        if finished:
            return total / repeats


@torch.inference_mode()
def measure_configuration(s, b, handle):
    x = torch.randn(b, 3, s, s, device="cuda", dtype=torch.float32)
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), peak, measure_energy(x, handle)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model.to(device="cuda", dtype=torch.float32).eval()
    Path("results").mkdir(exist_ok=True)
    metadata = {"gpu": torch.cuda.get_device_name(),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "python": platform.python_version(), "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "seed": SEED, "warmup": WARMUP, "latency_repeats": REPEATS,
                "energy_seconds": ENERGY_SECONDS, "energy_method": "nvml_power_integral"}
    output = Path("results/measurements.csv")
    meta_path = Path("results/measurements.meta.json")
    completed = set()
    if output.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in metadata.items():
            if previous.get(key) != value:
                raise RuntimeError(f"Cannot resume measurements: metadata differs for {key}")
        with output.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                point = (int(row["S"]), int(row["B"]))
                if point in completed or row["status"] not in {"OK", "OOM"}:
                    raise RuntimeError(f"Invalid or duplicate measurement: {point}")
                completed.add(point)
    metadata.update({"base_image_sizes": BASE_S, "base_batch_sizes": BASE_B,
                     "extra_regime_points": REGIME_POINTS, "extra_oom_points": OOM_POINTS,
                     "num_configurations": len(make_grid()),
                     "cudnn_benchmark": False, "cudnn_allow_tf32": False,
                     "matmul_allow_tf32": False})
    pynvml.nvmlInit()
    try:
        uuid = torch.cuda.get_device_properties(0).uuid
        handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        with output.open("a", newline="", encoding="utf-8") as file:
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            writer = csv.writer(file)
            if not completed:
                writer.writerow(["S", "B", "latency", "memory", "energy", "is_validation", "status"])
            print(f"Grid: {len(make_grid())} points; already recorded: {len(completed)}", flush=True)
            for s, b in make_grid():
                if (s, b) in completed:
                    continue
                try:
                    values, status = measure_configuration(s, b, handle), "OK"
                except torch.cuda.OutOfMemoryError as error:
                    values, status = ("", "OOM", ""), "OOM"
                    print(str(error), flush=True)
                writer.writerow([s, b, *values, int(s not in BASE_S or b not in BASE_B), status])
                file.flush()
                print(f"S={s}, B={b}: {status}", flush=True)
                torch.cuda.empty_cache()
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
