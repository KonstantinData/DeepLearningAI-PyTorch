"""
Helper plotting utilities for the 'Building a Simple Neural Network' notebook.

The notebook imports this module as `import helper_utils` and calls:
- helper_utils.plot_results(model, distances, times)
- helper_utils.plot_nonlinear_comparison(model, new_distances, new_times)

These functions are intentionally lightweight and only depend on matplotlib + torch (+ numpy).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor shaped [N] or [N,1] to a 1D numpy array on CPU."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(x)!r}")
    x = x.detach().cpu()
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    return x.numpy().astype(float)


def plot_results(model: torch.nn.Module, distances: torch.Tensor, times: torch.Tensor, *,
                 title: str = "Linear fit: distance → time",
                 show: bool = True,
                 save_path: Optional[str] = None) -> None:
    """
    Plot training data (distance vs time) and the model's fitted line.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model that maps distances -> times.
    distances : torch.Tensor
        Tensor shaped [N,1] or [N].
    times : torch.Tensor
        Tensor shaped [N,1] or [N].
    title : str
        Plot title.
    show : bool
        If True, call matplotlib.pyplot.show().
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    import matplotlib.pyplot as plt  # local import for notebook friendliness

    x = _to_numpy_1d(distances)
    y = _to_numpy_1d(times)

    # Build a smooth x-axis for the fitted line
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200, dtype=float)
    x_line_t = torch.tensor(x_line.reshape(-1, 1), dtype=distances.dtype)
    with torch.no_grad():
        y_line_t = model(x_line_t)
    y_line = _to_numpy_1d(y_line_t)

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x_line, y_line, label="Model")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Time (minutes)")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_nonlinear_comparison(model: torch.nn.Module, distances: torch.Tensor, times: torch.Tensor, *,
                              title: str = "Linear model on (potentially) non-linear data",
                              show: bool = True,
                              save_path: Optional[str] = None) -> None:
    """
    Plot a combined/non-linear dataset and overlay the linear model's predictions.

    This is useful to visually demonstrate underfitting when the true relationship changes
    (e.g., bikes for short distances vs cars for longer distances).

    Parameters are the same as `plot_results`.
    """
    import matplotlib.pyplot as plt  # local import

    x = _to_numpy_1d(distances)
    y = _to_numpy_1d(times)

    # Predict using the provided model across the full x-range
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 300, dtype=float)
    x_line_t = torch.tensor(x_line.reshape(-1, 1), dtype=distances.dtype)
    with torch.no_grad():
        y_pred_t = model(x_line_t)
    y_pred = _to_numpy_1d(y_pred_t)

    plt.figure()
    plt.scatter(x, y, label="Data (combined)")
    plt.plot(x_line, y_pred, label="Linear model prediction")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Time (minutes)")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
