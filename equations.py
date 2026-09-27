import numpy as np

def flops(image_size, batch):                  # -> float (FLOPs)
    S = np.asarray(image_size)
    B = np.asarray(batch)
    return 17751 * B * S * S + 313956 * B

def memory(image_size, batch):                 # -> float (bytes)
    S = np.asarray(image_size)
    B = np.asarray(batch)
    return 4161296 + 52 * B * S * S

_LAYER_COEFFICIENTS = [
    (2352, 0, 44, 0, 18816),       # Conv1
    (8, 0, 64, 0, 0),             # ReLU1
    (16, 0, 40, 0, 0),            # MaxPool

    (6400, 0, 24, 0, 204800),     # Conv2
    (4, 0, 32, 0, 0),             # ReLU2

    (2304, 0, 24, 0, 294912),     # Conv3
    (2, 0, 16, 0, 0),             # ReLU3

    (1024, 0, 24, 0, 131072),     # Conv4
    (4, 0, 32, 0, 0),             # ReLU4

    (4608, 0, 20, 0, 2359296),    # Conv5
    (1, 0, 8, 0, 0),              # ReLU5

    (1024, 0, 12, 0, 524288),     # Conv6
    (2, 0, 16, 0, 0),             # ReLU6

    (2, 0, 8, 2048, 0),           # GlobalAvgPool
    (0, 262400, 0, 3072, 525312), # Linear1 с bias
    (0, 256, 0, 2048, 0),         # ReLU после Linear1
    (0, 51300, 0, 1424, 102800),  # Linear2 с bias
]

def latency(image_size, batch, theta):         # -> float (seconds)
    S, B = np.broadcast_arrays(
        np.asarray(image_size, dtype=float),
        np.asarray(batch, dtype=float),
    )

    T0 = theta["overhead_s"]
    P = theta["compute_flops_s"]
    W = theta["bandwidth_bytes_s"]

    result = np.full(S.shape, T0, dtype=float)

    for a, b, c, d, e in _LAYER_COEFFICIENTS:
        F_i = a * B * S**2 + b * B
        Q_i = c * B * S**2 + d * B + e

        result += np.maximum(F_i / P, Q_i / W)

    return result.item() if result.ndim == 0 else result

def energy(image_size, batch, theta_energy):   # -> float (joules)
    time = latency(
        image_size,
        batch,
        theta_energy["latency"],
    )
    return theta_energy["mean_power_w"] * time
