import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from equations import energy, latency, memory


def fit_latency(data):
    def unpack(values):
        return dict(zip(["overhead_s", "compute_flops_s", "bandwidth_bytes_s"], np.exp(values).tolist()))

    def residuals(values):
        return latency(data.S, data.B, unpack(values)) / data.latency - 1

    fits = [least_squares(residuals, np.log(start),
                          bounds=(np.log([1e-9, 1e6, 1e6]), np.log([1, 1e17, 1e15])))
            for start in ([1e-5, 1e11, 1e10], [1e-4, 1e12, 1e11], [1e-3, 1e13, 1e12])]
    fits = [fit for fit in fits if fit.success]
    if not fits:
        raise RuntimeError("Не удалось откалибровать latency")
    return unpack(min(fits, key=lambda fit: fit.cost).x)


def main():
    data = pd.read_csv("results/measurements.csv")
    data = data[data.status == "OK"]
    train = data[data.is_validation == 0]
    test = data[data.is_validation == 1]
    theta = fit_latency(train)
    ratios = latency(train.S, train.B, theta) / train.energy
    theta_energy = {"latency": theta, "mean_power_w": float(ratios.sum() / (ratios**2).sum())}
    errors = {
        name: float(100 * np.abs(predict(test.S, test.B, parameters) / test[name] - 1).mean())
        for name, predict, parameters in (("latency", latency, theta), ("energy", energy, theta_energy))
    }
    errors["memory"] = float(100 * np.abs(memory(test.S, test.B) / pd.to_numeric(test.memory) - 1).mean())
    result = {"latency": theta, "energy": theta_energy, "validation_mape_percent": errors}
    Path("results/theta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
