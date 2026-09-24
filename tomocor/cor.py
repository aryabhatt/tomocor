"""Center-of-rotation math: preprocessing, overlap metrics and automatic search.

No Qt here so everything is unit-testable.

Conventions
-----------
``a`` is the 0-degree projection and ``bf`` the horizontally flipped 180-degree
projection, both (rows, cols). A *shift* ``s`` moves ``bf`` right by ``s`` pixels, so
column ``x`` of ``a`` is compared with column ``x - s`` of ``bf``. The center of
rotation (tomopy convention, ``width / 2`` for a centred axis) is
``width / 2 + s / 2``.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy import ndimage, signal

_FRAC_EPS = 1e-9


def shift_to_cor(shift, width):
    return width / 2.0 + shift / 2.0


def cor_to_shift(cor, width):
    return 2.0 * (cor - width / 2.0)


def denoise(img, median=3, sigma=3.0):
    """Suppress noise: median filter (zingers/hot pixels), then Gaussian low-pass."""
    out = np.asarray(img, dtype=np.float32)
    if median and median > 1:
        out = ndimage.median_filter(out, size=int(median))
    if sigma and sigma > 0:
        out = ndimage.gaussian_filter(out, sigma)
    return out


def metric_view(img, highpass_sigma=None, remove_row_offset=False):
    """Emphasise what a horizontal shift can change, for the metric and difference view.

    ``remove_row_offset`` subtracts each row's mean. Any row-constant part (horizontal bands,
    detector offsets) is invariant to horizontal shifts, so on banded data it dominates
    the correlation and pins ZNCC near 1 for every shift. ``highpass_sigma`` subtracts a
    heavily blurred copy (band-pass), removing slowly varying 2D background.
    """
    out = img
    if highpass_sigma:
        out = out - ndimage.gaussian_filter(out, highpass_sigma)
    if remove_row_offset:
        out = out - out.mean(axis=1, keepdims=True)
    return out


def preprocess(img, median=3, sigma=3.0, highpass_sigma=None, remove_row_offset=False):
    return metric_view(denoise(img, median, sigma), highpass_sigma, remove_row_offset)


def overlap(a, bf, shift):
    """Overlapping part of ``a`` and ``bf`` shifted right by ``shift`` (may be fractional).

    Returns ``(x0, a_crop, b_crop)`` where ``a_crop`` starts at column ``x0`` of ``a``,
    or ``None`` if the images do not overlap. Pure function of its arguments: nothing
    is carried over from previous shifts.
    """
    width = a.shape[1]
    fl = math.floor(-shift)
    frac = -shift - fl  # every column shares the same sub-pixel offset
    interp = frac > _FRAC_EPS
    x0 = max(0, math.ceil(shift))
    x1 = min(width - 1, width - 1 - fl - (1 if interp else 0)) + 1
    if x1 <= x0:
        return None
    b = bf[:, x0 + fl : x1 + fl]
    if interp:
        b = (1.0 - frac) * b + frac * bf[:, x0 + fl + 1 : x1 + fl + 1]
    return x0, a[:, x0:x1], b


@dataclass
class Score:
    zncc: float
    rms: float
    overlap_width: int
    overlap_frac: float


def score_at(a, bf, shift):
    """ZNCC and RMS difference over the overlap at an arbitrary (float) shift."""
    ov = overlap(a, bf, shift)
    if ov is None:
        return Score(math.nan, math.nan, 0, 0.0)
    _, fa, gb = ov
    fc = fa - fa.mean(dtype=np.float64)
    gc = gb - gb.mean(dtype=np.float64)
    num = float(np.sum(fc * gc, dtype=np.float64))
    den = math.sqrt(float(np.sum(fc * fc, dtype=np.float64) * np.sum(gc * gc, dtype=np.float64)))
    zncc = num / den if den > 0 else math.nan
    rms = float(np.std(fa - gb, dtype=np.float64))  # offset-free: beam decay is not misalignment
    return Score(zncc, rms, fa.shape[1], fa.shape[1] / a.shape[1])


@dataclass
class Curve:
    shifts: np.ndarray  # integer shifts, -(w-1) .. w-1
    zncc: np.ndarray  # NaN where the overlap is below the minimum
    rms: np.ndarray
    overlap_width: np.ndarray


def _range_sums(values, lo, hi):
    cs = np.concatenate([[0.0], np.cumsum(values)])
    return cs[hi] - cs[lo]


def shift_curve(a, bf, min_overlap_frac=0.3, row_chunk=256):
    """ZNCC and RMS difference for every integer shift, via FFT along x only.

    Uses the overlap-restricted (masked) normalised cross-correlation: running sums of
    f, g, f^2, g^2 over the overlap plus one FFT cross-correlation summed over rows.
    Cost is O(H W log W), so the whole curve is cheap.
    """
    h, w = a.shape
    a = a.astype(np.float64) - float(a.mean())
    bf = bf.astype(np.float64) - float(bf.mean())

    shifts = np.arange(-(w - 1), w)
    lo = np.maximum(0, shifts)
    hi = np.minimum(w, w + shifts)
    n_cols = hi - lo
    n = n_cols * h

    nfft = sfft.next_fast_len(2 * w - 1, real=True)
    acc = np.zeros(nfft // 2 + 1, dtype=np.complex128)
    for r in range(0, h, row_chunk):
        fa = sfft.rfft(a[r : r + row_chunk], n=nfft, axis=1, workers=-1)
        fb = sfft.rfft(bf[r : r + row_chunk], n=nfft, axis=1, workers=-1)
        acc += (fa * np.conj(fb)).sum(axis=0)
    # c[k] = sum_x a[x] bf[x - k]; negative k wraps around and indexes from the end
    sfg = sfft.irfft(acc, n=nfft)[shifts]

    sf = _range_sums(a.sum(axis=0), lo, hi)
    sff = _range_sums((a * a).sum(axis=0), lo, hi)
    sg = _range_sums(bf.sum(axis=0), lo - shifts, hi - shifts)
    sgg = _range_sums((bf * bf).sum(axis=0), lo - shifts, hi - shifts)

    var_f = sff - sf * sf / n
    var_g = sgg - sg * sg / n
    cov = sfg - sf * sg / n
    den = np.sqrt(np.clip(var_f, 0, None) * np.clip(var_g, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        zncc = np.where(den > 0, cov / den, np.nan)
    sq_diff = np.clip(var_f + var_g - 2.0 * cov, 0, None)
    rms = np.sqrt(sq_diff / n)  # std of the difference

    invalid = n_cols < max(1, min_overlap_frac * w)
    zncc[invalid] = np.nan
    rms[invalid] = np.nan
    return Curve(shifts, zncc, rms, n_cols)


def refine_subpixel(shifts, values):
    """Peak location of ``values`` refined with a parabola through the top 3 samples."""
    i = int(np.nanargmax(values))
    if i == 0 or i == len(values) - 1:
        return float(shifts[i])
    y0, y1, y2 = values[i - 1], values[i], values[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if not np.isfinite(denom) or denom >= 0:
        return float(shifts[i])
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(shifts[i]) + delta


def phase_corr_curve(a, bf, reg=0.03):
    """Row-summed FFT phase correlation vs horizontal shift, whitened cross-power spectrum.

    Same principle as tomopy/skimage's ``find_center_pc``: the cross-power spectrum is
    normalised toward unit magnitude before the inverse transform, so the profile depends
    mostly on phase, not amplitude. Unlike `shift_curve` there is no overlap windowing, so
    this is a circular correlation and ``shifts`` wraps at +/- w/2. A Hann taper along x kills
    the edge-discontinuity ringing that otherwise swamps the true peak (the images are not
    actually periodic).

    Full whitening (``reg=0``) boosts every frequency bin to unit magnitude regardless of its
    actual SNR, so bins with little real signal contribute noise at full strength and the curve
    is dominated by hundreds of spurious local maxima. ``reg`` softens this: the cross-power
    spectrum is divided by ``|cross| + reg * max(|cross|)`` instead of ``|cross|`` alone, so
    low-magnitude (noise-dominated) bins are down-weighted rather than amplified. On real
    projection data this reduced local maxima in the curve from ~850 to ~3 at the default
    while moving the recovered shift by <0.2px.
    """
    w = a.shape[1]
    win = np.hanning(w)
    a = (a.astype(np.float64) - float(a.mean())) * win
    bf = (bf.astype(np.float64) - float(bf.mean())) * win
    fa = sfft.fft(a, axis=1, workers=-1)
    fbf = sfft.fft(bf, axis=1, workers=-1)
    cross = (fa * np.conj(fbf)).sum(axis=0)
    mag = np.abs(cross)
    cross /= mag + reg * mag.max() + 1e-12
    r = sfft.ifft(cross).real  # r[k] = sum_x a[x] bf[x - k] (phase-only), circular in x

    shifts = np.arange(-(w // 2), w - w // 2)
    return shifts, r[shifts]  # negative shifts wrap via fancy indexing, matching the lag above


@dataclass
class PcResult:
    shift: float
    cor: float
    peak: float  # phase-correlation value at the (sub-pixel refined) shift


def find_center_pc(a, bf, reg=0.03):
    """Center of rotation via FFT phase correlation (tomopy/skimage ``find_center_pc`` style).

    A single-shot estimate from the peak of `phase_corr_curve`, refined to sub-pixel with
    the same parabolic fit `auto_cor` uses. Useful as a cross-check against the masked-NCC
    based `auto_cor`, since the two can fail on different kinds of data.
    """
    shifts, r = phase_corr_curve(a, bf, reg)
    shift = refine_subpixel(shifts, r)
    return PcResult(shift, shift_to_cor(shift, a.shape[1]), float(np.max(r)))


@dataclass
class AutoResult:
    shift: float
    cor: float
    zncc: float
    confidence: float  # peak ZNCC minus the next-highest local maximum


def auto_cor(a, bf, min_overlap_frac=0.3, curve=None):
    """Best shift/COR from the ZNCC curve, with a confidence value.

    Confidence is the gap between the highest and second-highest local maximum of the
    curve: near zero means an ambiguous result that the user should check by eye.
    """
    if curve is None:
        curve = shift_curve(a, bf, min_overlap_frac)
    z = curve.zncc
    if not np.any(np.isfinite(z)):
        raise ValueError("no valid overlap: images are constant or too small")
    shift = refine_subpixel(curve.shifts, z)
    filled = np.nan_to_num(z, nan=-1.0)
    peaks, _ = signal.find_peaks(filled)
    heights = np.sort(filled[peaks])[::-1]
    second = heights[1] if len(heights) > 1 else float(np.nanmedian(z))
    peak = float(np.nanmax(z))
    return AutoResult(shift, float(shift_to_cor(shift, a.shape[1])), peak, peak - float(second))
