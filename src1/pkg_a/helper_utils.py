"""
Helper plotting utilities for simple PyTorch regression notebooks/scripts.

This module is designed to be "notebook-friendly" and avoids blocking matplotlib
windows by default. You can control display behavior via the `show`, `block`,
and `close` parameters (defaults are non-blocking and keep figures open).

The functions below match the API expected by your training script:

- plot_data(distances, times, normalize=False)
- plot_training_progress(epoch, loss, model, distances_norm, times_norm)
- plot_final_fit(model, distances, times, distances_norm, times_std, times_mean)

Dependencies: matplotlib + torch (+ numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
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
    if x.ndim != 1:
        raise ValueError(f"Expected tensor of shape [N] or [N,1], got {tuple(x.shape)}")
    return x.numpy().astype(float)


def _maybe_show(*, show: bool, block: bool, close: bool, pause: float) -> None:
    """
    Display handling for matplotlib.

    - show=True, block=True  -> plt.show() blocks until windows are closed
    - show=True, block=False -> plt.show(block=False) + plt.pause(pause) (non-blocking)
    - show=False             -> do not call show; optionally close the figure if close=True
    """
    import matplotlib.pyplot as plt  # local import

    if show:
        plt.show(block=block)
        if not block:
            plt.pause(pause)
        if close:
            plt.close()
    else:
        if close:
            plt.close()


def _predict_line(
    model: torch.nn.Module,
    x_line: np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Run model on x_line (shape [N]) and return predictions as 1D numpy array."""
    x_t = torch.tensor(x_line.reshape(-1, 1), dtype=dtype)
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        y_t = model(x_t)
    if model_was_training:
        model.train()
    return _to_numpy_1d(y_t)


# -----------------------------------------------------------------------------
# Public API expected by the user's script
# -----------------------------------------------------------------------------

def plot_data(
    distances: torch.Tensor,
    times: torch.Tensor,
    *,
    normalize: bool = False,
    title: Optional[str] = None,
    show: bool = True,
    block: bool = False,
    close: bool = False,
    pause: float = 0.001,
    save_path: Optional[str] = None,
) -> None:
    """
    Scatter-plot of distances vs times.

    Parameters
    ----------
    distances, times:
        Tensors shaped [N,1] or [N].
    normalize:
        If True, labels/title indicate standardized values.
    show, block, close, pause:
        Matplotlib window behavior.
    save_path:
        Optional path to save the figure.
    """
    import matplotlib.pyplot as plt  # local import

    x = _to_numpy_1d(distances)
    y = _to_numpy_1d(times)

    if title is None:
        title = "Normalized data" if normalize else "Raw data"

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.xlabel("Distance (standardized)" if normalize else "Distance (miles)")
    plt.ylabel("Time (standardized)" if normalize else "Time (minutes)")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    _maybe_show(show=show, block=block, close=close, pause=pause)


def plot_training_progress(
    *,
    epoch: int,
    loss: torch.Tensor,
    model: torch.nn.Module,
    distances_norm: torch.Tensor,
    times_norm: torch.Tensor,
    show: bool = True,
    block: bool = False,
    close: bool = False,
    pause: float = 0.001,
    save_path: Optional[str] = None,
) -> None:
    """
    Live plot used inside a training loop.

    This draws the normalized training data and the model's current prediction
    curve. It reuses a single figure for performance.

    Notes
    -----
    - Uses figure id=1 to avoid creating thousands of windows during training.
    - Non-blocking by default.
    """
    import matplotlib.pyplot as plt  # local import

    x = _to_numpy_1d(distances_norm)
    y = _to_numpy_1d(times_norm)

    # Make a smooth line in normalized x-space
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 250, dtype=float)
    y_line = _predict_line(model, x_line, dtype=distances_norm.dtype)

    plt.figure(1)
    plt.clf()
    plt.scatter(x, y, label="Data (norm)")
    plt.plot(x_line, y_line, label="Model")
    plt.xlabel("Distance (standardized)")
    plt.ylabel("Time (standardized)")
    plt.title(f"Epoch {epoch + 1} | Loss: {float(loss.item()):.6f}")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    _maybe_show(show=show, block=block, close=close, pause=pause)


def plot_final_fit(
    model: torch.nn.Module,
    distances: torch.Tensor,
    times: torch.Tensor,
    distances_norm: torch.Tensor,
    times_std: torch.Tensor | float,
    times_mean: torch.Tensor | float,
    *,
    title: str = "Final fit: distance → time",
    show: bool = True,
    block: bool = True,
    close: bool = False,
    pause: float = 0.001,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot original (unnormalized) data and the model prediction in original units.

    Parameters
    ----------
    model:
        Trained model that predicts *normalized* time from *normalized* distance.
    distances, times:
        Original (unnormalized) tensors.
    distances_norm:
        Normalized distances tensor (used to infer the original normalization).
    times_std, times_mean:
        Scalars used to denormalize the model output:
            time = time_norm * times_std + times_mean
    """
    import matplotlib.pyplot as plt  # local import

    x_raw = _to_numpy_1d(distances)
    y_raw = _to_numpy_1d(times)

    # Infer distance normalization from provided tensors:
    # distances_norm = (distances - mean) / std  => distances = distances_norm*std + mean
    d_raw = _to_numpy_1d(distances)
    d_norm = _to_numpy_1d(distances_norm)
    dist_mean = float(np.mean(d_raw))
    dist_std = float(np.std(d_raw, ddof=1))  # torch.std default is unbiased=True

    # Use a smooth line on the original x-scale
    x_line_raw = np.linspace(float(np.min(x_raw)), float(np.max(x_raw)), 300, dtype=float)

    # Normalize x_line, predict normalized y, then denormalize
    x_line_norm = (x_line_raw - dist_mean) / dist_std
    y_line_norm = _predict_line(model, x_line_norm, dtype=distances_norm.dtype)

    times_std_f = float(times_std) if not isinstance(times_std, torch.Tensor) else float(times_std.detach().cpu().item())
    times_mean_f = float(times_mean) if not isinstance(times_mean, torch.Tensor) else float(times_mean.detach().cpu().item())
    y_line_raw = (y_line_norm * times_std_f) + times_mean_f

    plt.figure()
    plt.scatter(x_raw, y_raw, label="Data")
    plt.plot(x_line_raw, y_line_raw, label="Model (denorm)")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Time (minutes)")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    _maybe_show(show=show, block=block, close=close, pause=pause)


# -----------------------------------------------------------------------------
# Backwards-compatible aliases (in case older notebooks imported these)
# -----------------------------------------------------------------------------

def plot_results(model: torch.nn.Module, distances: torch.Tensor, times: torch.Tensor, **kwargs) -> None:
    """Alias for older notebooks: plot original data + model line."""
    plot_final_fit(
        model=model,
        distances=distances,
        times=times,
        distances_norm=distances,  # not used for plotting if caller only uses this alias
        times_std=1.0,
        times_mean=0.0,
        **kwargs,
    )


def plot_nonlinear_comparison(model: torch.nn.Module, distances: torch.Tensor, times: torch.Tensor, **kwargs) -> None:
    """Alias that simply plots the (distances, times) scatter with the model in the same scale."""
    # If caller uses this on unnormalized data with a model trained on normalized data,
    # the curve may look wrong; for new code, prefer plot_final_fit.
    import matplotlib.pyplot as plt

    x = _to_numpy_1d(distances)
    y = _to_numpy_1d(times)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 300, dtype=float)
    y_pred = _predict_line(model, x_line, dtype=distances.dtype)

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x_line, y_pred, label="Model")
    plt.xlabel("Distance")
    plt.ylabel("Time")
    plt.title(kwargs.pop("title", "Model vs data"))
    plt.legend()
    _maybe_show(
        show=bool(kwargs.pop("show", False)),
        block=bool(kwargs.pop("block", False)),
        close=bool(kwargs.pop("close", False)),
        pause=float(kwargs.pop("pause", 0.001)),
    )
