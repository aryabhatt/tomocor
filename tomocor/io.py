"""Load the 0 and 180 degree projections from an /exchange-layout HDF5 file."""

import warnings
from dataclasses import dataclass

import h5py
import numpy as np

DATA = "/exchange/data"
WHITE = "/exchange/data_white"
DARK = "/exchange/data_dark"
THETA = "/exchange/theta"  # degrees

MAX_AVG_FRAMES = 10
ANGLE_WARN_DEG = 0.5


@dataclass
class ProjectionPair:
    proj0: np.ndarray  # attenuation, (rows, cols)
    proj180: np.ndarray
    theta0: float  # actual angles of the frames used (mean if averaged)
    theta180: float
    n0: int  # number of frames averaged
    n180: int
    path: str


def _mean_stack(ds, max_frames=32):
    """Mean of a 2D image or of up to ``max_frames`` evenly spaced frames of a 3D stack."""
    if ds.ndim == 2:
        return ds[()].astype(np.float32)
    n = ds.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))
    return ds[idx].mean(axis=0, dtype=np.float32)


def select_frames(theta, target, tol_deg, max_frames=MAX_AVG_FRAMES):
    """Indices (sorted) of frames within ``tol_deg`` of ``target``, always including the nearest.

    Returns ``(indices, angular_error_of_nearest)``.
    """
    dist = np.abs(theta - target)
    nearest = int(np.argmin(dist))
    idx = np.flatnonzero(dist <= tol_deg)
    if nearest not in idx:
        idx = np.append(idx, nearest)
    idx = idx[np.argsort(dist[idx])][:max_frames]
    return np.sort(idx), float(dist[nearest])


def _attenuation(frames, dark, flat):
    """Flat/dark correct, average the transmission over frames, take -log."""
    denom = np.maximum(flat - dark, 1e-3)
    trans = ((frames.astype(np.float32) - dark) / denom).mean(axis=0)
    return -np.log(np.clip(trans, 1e-6, None)).astype(np.float32)


def load_pair(path, avg_tol_deg=0.0):
    """Read the projections nearest theta[0] and theta[0] + 180 deg.

    Frames within ``avg_tol_deg`` of either target are averaged; the default 0 uses the single
    nearest frame. Averaging is opt-in because a window is one-sided at the ends of a 0-180
    scan: the averaged angles then differ by less than 180 deg and the object blurs by
    ``radius * window`` pixels. Only the needed frames are read from disk.
    """
    with h5py.File(path, "r") as f:
        for key in (DATA, WHITE, DARK, THETA):
            if key not in f:
                raise KeyError(f"{key} not found in {path}")
        theta = np.asarray(f[THETA][()], dtype=np.float64).ravel()
        data = f[DATA]
        if data.ndim != 3 or data.shape[0] != theta.size:
            raise ValueError(
                f"{DATA} shape {data.shape} does not match {theta.size} angles in {THETA}"
            )
        dark = _mean_stack(f[DARK])
        flat = _mean_stack(f[WHITE])

        images, used = [], []
        for target in (theta[0], theta[0] + 180.0):
            idx, err = select_frames(theta, target, avg_tol_deg)
            if err > ANGLE_WARN_DEG:
                warnings.warn(
                    f"nearest frame to {target:.2f} deg is {err:.2f} deg away", stacklevel=2
                )
            images.append(_attenuation(data[idx], dark, flat))
            used.append((float(theta[idx].mean()), len(idx)))

    (t0, n0), (t180, n180) = used
    if abs((t180 - t0) - 180.0) > ANGLE_WARN_DEG:
        warnings.warn(
            f"averaged frames are {t180 - t0:.2f} deg apart, not 180; "
            "the alignment is biased. Use a smaller averaging window.",
            stacklevel=2,
        )
    return ProjectionPair(images[0], images[1], t0, t180, n0, n180, str(path))
