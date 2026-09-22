"""Synthetic projections with a known center of rotation, for tests and demos."""

import numpy as np

# Angles written by write_h5: 0/180 each have a near neighbour so frame averaging is exercised.
DEMO_THETA = (0.0, 0.3, 45.0, 90.0, 135.0, 179.7, 180.0)
DARK_LEVEL = 100.0


def _blobs(rng, width, height, n_blobs):
    """Gaussian blobs: x offset from axis, depth z, row y, sigma, amplitude."""
    return np.column_stack(
        [
            rng.uniform(-0.35, 0.35, n_blobs) * width,
            rng.uniform(-0.35, 0.35, n_blobs) * width,
            rng.uniform(0.1, 0.9, n_blobs) * height,
            rng.uniform(3.0, 25.0, n_blobs),
            rng.uniform(0.05, 0.5, n_blobs),
        ]
    )


def _line_integral(blobs, theta_deg, cor, width, height):
    """Projection (attenuation) at ``theta_deg``; ``cor`` uses the ``width / 2`` convention."""
    axis = cor - 0.5  # pixel-centre coordinate of the rotation axis
    th = np.deg2rad(theta_deg)
    x_det = axis + blobs[:, 0] * np.cos(th) + blobs[:, 1] * np.sin(th)
    sigma = blobs[:, 3:4]
    gx = np.exp(-((np.arange(width) - x_det[:, None]) ** 2) / (2 * sigma**2))
    gy = np.exp(-((np.arange(height) - blobs[:, 2:3]) ** 2) / (2 * sigma**2))
    return np.einsum("nh,nw->hw", blobs[:, 4:5] * gy, gx)


def _counts(rng, line_integral, photons, gain):
    return rng.poisson(gain * photons * np.exp(-line_integral)).astype(np.float64)


def make_pair(width=512, height=256, cor=None, photons=None, gain180=0.9, n_blobs=30, seed=0):
    """Attenuation images at 0 and 180 degrees of an object rotating about ``cor``.

    ``photons`` is the flat-field count per pixel (``None`` for noiseless). The 180-degree
    frame is scaled by ``gain180`` to mimic beam decay.
    """
    rng = np.random.default_rng(seed)
    blobs = _blobs(rng, width, height, n_blobs)
    cor = width / 2.0 + 11.3 if cor is None else cor
    out = []
    for theta, gain in ((0.0, 1.0), (180.0, gain180)):
        p = _line_integral(blobs, theta, cor, width, height)
        if photons is None:
            out.append((p - np.log(gain)).astype(np.float32))
        else:
            counts = np.maximum(_counts(rng, p, photons, gain), 0.5)
            out.append((-np.log(counts / photons)).astype(np.float32))
    return out[0], out[1]


def write_h5(path, width=512, height=256, cor=None, photons=2000, n_blobs=30, seed=0):
    """Write a small raw file with the /exchange layout (data, data_white, data_dark, theta)."""
    import h5py

    rng = np.random.default_rng(seed)
    blobs = _blobs(rng, width, height, n_blobs)
    cor = width / 2.0 + 11.3 if cor is None else cor
    frames = []
    for theta in DEMO_THETA:
        gain = 1.0 - 0.1 * theta / 180.0
        p = _line_integral(blobs, theta, cor, width, height)
        frames.append(DARK_LEVEL + _counts(rng, p, photons, gain))
    flats = [DARK_LEVEL + rng.poisson(photons, (height, width)) for _ in range(5)]
    darks = [rng.poisson(DARK_LEVEL, (height, width)) for _ in range(5)]
    with h5py.File(path, "w") as f:
        f["/exchange/data"] = np.asarray(frames, dtype=np.float32)
        f["/exchange/data_white"] = np.asarray(flats, dtype=np.float32)
        f["/exchange/data_dark"] = np.asarray(darks, dtype=np.float32)
        f["/exchange/theta"] = np.asarray(DEMO_THETA, dtype=np.float32)
    return cor
