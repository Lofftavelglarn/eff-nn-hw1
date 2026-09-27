import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equations import _LAYER_COEFFICIENTS, energy, flops, latency, memory


def main():
    data = pd.read_csv("results/measurements.csv")
    theta = json.loads(Path("results/theta.json").read_text())
    meta = json.loads(Path("results/measurements.meta.json").read_text())
    folder = Path("results/figures")
    folder.mkdir(exist_ok=True)
    capacity = meta["gpu_total_memory_bytes"] / 2**30
    sizes = sorted(data.S.unique())
    specs = [("latency", latency, theta["latency"], 1000, "Latency (ms)"),
             ("memory", lambda s, b, _: memory(s, b), None, 1 / 2**30, "Peak allocated memory (GiB)"),
             ("energy", energy, theta["energy"], 1, "Energy (J / forward)")]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    for name, predict, parameters, scale, label in specs:
        rows = (len(sizes) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 3.25 * rows), squeeze=False)
        for ax, s in zip(axes.flat, sizes):
            group = data[data.S == s]
            batches = np.geomspace(group.B.min(), group.B.max(), 250)
            ax.plot(batches, predict(s, batches, parameters) * scale,
                    color="black", label="Analytical prediction")
            for validation, marker, color, legend in [(0, "o", "tab:blue", "Calibration measurements"),
                                                     (1, "^", "tab:orange", "Validation measurements")]:
                ok = group[(group.status == "OK") & (group.is_validation == validation)]
                ax.scatter(ok.B, pd.to_numeric(ok[name]) * scale,
                           marker=marker, color=color, s=25, label=legend, zorder=3)
            oom = group[group.status == "OOM"]
            if len(oom):
                ax.scatter(oom.B, np.full(len(oom), 0.96), transform=ax.get_xaxis_transform(),
                           marker="x", color="red", s=55,
                           label="OOM (no measured value; top markers)", zorder=4)
            if name == "memory" and (len(oom) or pd.to_numeric(group.loc[group.status == "OK", "memory"]).max() / 2**30 >= capacity / 2):
                ax.axhline(capacity, color="red", linestyle="--", label="GPU total capacity")
            ax.set(xscale="log", yscale="log", xlabel="Batch size B", ylabel=label, title=f"S = {s}")
        for ax in list(axes.flat)[len(sizes):]:
            ax.set_visible(False)
        handles = {}
        for ax in axes.flat:
            for handle, label_text in zip(*ax.get_legend_handles_labels()):
                handles[label_text] = handle
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3)
        fig.suptitle(f"{name.capitalize()}: measured points and analytical curves")
        fig.tight_layout(rect=(0, 0.065, 1, 0.97))
        fig.savefig(folder / f"{name}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ok = data[data.status == "OK"]
    for ax, (name, predict, parameters, scale, label) in zip(axes, specs):
        for validation, marker, color, legend in [(0, "o", "tab:blue", "Calibration"),
                                                 (1, "^", "tab:orange", "Validation")]:
            subset = ok[ok.is_validation == validation]
            measured = pd.to_numeric(subset[name]).to_numpy() * scale
            predicted = np.asarray(predict(subset.S, subset.B, parameters)) * scale
            ax.scatter(measured, predicted, marker=marker, color=color, s=20, alpha=0.8, label=legend)
        actual = pd.to_numeric(ok[name]).to_numpy() * scale
        predicted = np.asarray(predict(ok.S, ok.B, parameters)) * scale
        lo, hi = min(actual.min(), predicted.min()) * 0.8, max(actual.max(), predicted.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", label="Perfect prediction")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               xlabel=f"Measured: {label}", ylabel=f"Predicted: {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "predicted_vs_measured.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, zoom in zip(axes, [False, True]):
        selection = data if not zoom else data[(data.S >= 512) & (data.B >= 128)]
        for status, marker, color in [("OK", "o", "tab:blue"), ("OOM", "x", "red")]:
            subset = selection[selection.status == status]
            ax.scatter(subset.S, subset.B, marker=marker, color=color, label=f"Measured {status}", s=45)
        s = np.linspace(480 if zoom else data.S.min(), data.S.max() * 1.04, 400)
        boundary = (meta["gpu_total_memory_bytes"] - 4161296) / (52 * s**2)
        ax.plot(s, boundary, "k--", label="Memory equation = total GPU capacity")
        ax.set(xlabel="Image side S (pixels)", ylabel="Batch size B", yscale="log",
               ylim=(100 if zoom else 0.7, data.B.max() * 1.2),
               title="OOM boundary: detail" if zoom else "All measured configurations")
        if zoom:
            ax.set_xlim(480, data.S.max() * 1.04)
            for image_size, group in selection.groupby("S"):
                successful = group[group.status == "OK"]
                failed = group[group.status == "OOM"]
                if successful.empty or failed.empty:
                    continue
                last_ok, first_oom = int(successful.B.max()), int(failed.B.min())
                right_edge = image_size == selection.S.max()
                ax.annotate(f"OK: B={last_ok}\nOOM: B={first_oom}",
                            xy=(image_size, last_ok),
                            xytext=(-12, 20) if right_edge else (12, -22),
                            textcoords="offset points", ha="right" if right_edge else "left",
                            fontsize=8, bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
                            arrowprops={"arrowstyle": "-", "linewidth": 0.6})
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(folder / "oom.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, s in enumerate([32, 64, 128]):
        measured = ok[ok.S == s]
        batches = np.geomspace(measured.B.min(), measured.B.max(), 250)
        times = latency(s, batches, theta["latency"])
        axes[0, column].plot(batches, times * 1000, color="black", label="Analytical latency")
        axes[0, column].axhline(theta["latency"]["overhead_s"] * 1000,
                               color="tab:gray", linestyle="--", label="Fitted overhead T0")
        axes[1, column].plot(batches, flops(s, batches) / times / 1e12,
                            color="black", label="Analytical throughput")
        for validation, marker, color, legend in [(0, "o", "tab:blue", "Calibration measurements"),
                                                 (1, "^", "tab:orange", "Validation measurements")]:
            subset = measured[measured.is_validation == validation]
            axes[0, column].scatter(subset.B, subset.latency * 1000,
                                    color=color, marker=marker, s=25, label=legend, zorder=3)
            axes[1, column].scatter(subset.B, flops(s, subset.B) / subset.latency / 1e12,
                                    color=color, marker=marker, s=25, label=legend, zorder=3)
        axes[0, column].set(xscale="log", yscale="log", xlabel="Batch size B",
                            ylabel="Latency (ms)", title=f"S = {s} pixels")
        axes[1, column].set(xscale="log", xlabel="Batch size B",
                            ylabel="Effective throughput (TFLOP/s)", title=f"S = {s} pixels: F / T")
        axes[0, column].legend(fontsize=8)
        axes[1, column].legend(fontsize=8)
    fig.suptitle("Launch overhead and throughput saturation; physical memory/compute limits are not profiled")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(folder / "regimes.png", dpi=150)
    plt.close(fig)
    n = (ok.B * ok.S**2).to_numpy()[:, None]
    b = ok.B.to_numpy()[:, None]
    coefficients = np.asarray(_LAYER_COEFFICIENTS)
    compute = (n * coefficients[:, 0] + b * coefficients[:, 1]) / theta["latency"]["compute_flops_s"]
    transfer = (n * coefficients[:, 2] + b * coefficients[:, 3] + coefficients[:, 4]) / theta["latency"]["bandwidth_bytes_s"]
    compute_part = np.where(compute >= transfer, compute, 0).sum(axis=1)
    transfer_part = np.where(compute < transfer, transfer, 0).sum(axis=1)
    report = ok[["S", "B", "is_validation", "latency"]].copy()
    report["effective_tflops_s"] = flops(ok.S, ok.B) / ok.latency / 1e12
    report["model_overhead_s"] = theta["latency"]["overhead_s"]
    report["model_compute_branches_s"] = compute_part
    report["model_memory_branches_s"] = transfer_part
    report.sort_values(["S", "B"]).to_csv("results/regimes.csv", index=False)
    print(f"Saved figures to {folder}")


if __name__ == "__main__":
    main()
