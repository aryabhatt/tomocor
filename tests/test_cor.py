import numpy as np
import pytest

from tomocor import cor as C
from tomocor.synthetic import make_pair

WIDTH = 512


def _prep(a, b, sigma=3.0):
    return C.preprocess(a, sigma=sigma), C.preprocess(b[:, ::-1], sigma=sigma)


def test_shift_cor_roundtrip():
    assert C.shift_to_cor(0, WIDTH) == WIDTH / 2
    assert C.cor_to_shift(C.shift_to_cor(-17.5, WIDTH), WIDTH) == pytest.approx(-17.5)


@pytest.mark.parametrize("shift", [0, 3, -3, 25, -25, 2.5, -2.5, 0.3])
def test_overlap_region(shift):
    rng = np.random.default_rng(1)
    a, bf = rng.normal(size=(8, 40)), rng.normal(size=(8, 40))
    x0, fa, gb = C.overlap(a, bf, shift)
    assert fa.shape == gb.shape
    # a[x] pairs with bf[x - shift]; check against direct linear interpolation
    for j in (0, fa.shape[1] - 1):
        u = x0 + j - shift
        expected = np.array([np.interp(u, np.arange(40), row) for row in bf])
        np.testing.assert_allclose(gb[:, j], expected, atol=1e-12)
        np.testing.assert_array_equal(fa[:, j], a[:, x0 + j])


def test_overlap_is_history_independent():
    """The prototype kept stale columns from earlier slider positions; this must not."""
    rng = np.random.default_rng(2)
    a, bf = rng.normal(size=(8, 60)), rng.normal(size=(8, 60))
    first = C.overlap(a, bf, 20)
    for s in (-20, 7, -33, 20):
        C.overlap(a, bf, s)
    again = C.overlap(a, bf, 20)
    assert first[0] == again[0]
    np.testing.assert_array_equal(first[1], again[1])
    np.testing.assert_array_equal(first[2], again[2])
    assert first[1].shape[1] == 60 - 20  # nothing outside the overlap


def test_no_overlap_returns_none():
    a = np.zeros((4, 10))
    assert C.overlap(a, a, 10) is None
    assert C.overlap(a, a, -10) is None


def test_curve_matches_score_at():
    a, b = make_pair(width=WIDTH, height=64, cor=WIDTH / 2 + 9.0, photons=500)
    pa, pb = _prep(a, b)
    curve = C.shift_curve(pa, pb, min_overlap_frac=0.3)
    for s in (0, 18, -40, 100, -120):
        i = int(np.where(curve.shifts == s)[0][0])
        sc = C.score_at(pa, pb, s)
        assert curve.zncc[i] == pytest.approx(sc.zncc, abs=1e-6)
        assert curve.rms[i] == pytest.approx(sc.rms, rel=1e-4, abs=1e-6)
        assert curve.overlap_width[i] == sc.overlap_width
    # below the minimum overlap the curve is masked out
    assert np.isnan(curve.zncc[curve.shifts == int(0.8 * WIDTH)]).all()


@pytest.mark.parametrize("shift", [0, 18, -40, 100, -120, 11.3, -25.7])
def test_find_center_pc_recovers_shift(shift):
    """Phase-correlation estimate: coarser than auto_cor's masked NCC, but in the ballpark."""
    a, b = make_pair(width=WIDTH, height=64, cor=WIDTH / 2 + shift / 2.0, photons=2000)
    pa, pb = _prep(a, b, sigma=1.0)
    shifts, r = C.phase_corr_curve(pa, pb)
    assert shifts[0] == -(WIDTH // 2)
    assert shifts[-1] == WIDTH - WIDTH // 2 - 1
    res = C.find_center_pc(pa, pb)
    assert res.shift == pytest.approx(shift, abs=2.5)
    assert res.cor == pytest.approx(C.shift_to_cor(shift, WIDTH), abs=2.5)
    assert res.peak == pytest.approx(float(np.max(r)), abs=1e-9)


@pytest.mark.parametrize("cor", [WIDTH / 2, WIDTH / 2 + 11.3, WIDTH / 2 - 37.75])
def test_auto_cor_noiseless(cor):
    a, b = make_pair(width=WIDTH, cor=cor)
    pa, pb = _prep(a, b, sigma=1.0)
    res = C.auto_cor(pa, pb)
    assert res.cor == pytest.approx(cor, abs=0.1)
    assert res.zncc > 0.99


# (photons, max abs error in px of COR). Tolerances are what the default pipeline achieves.
@pytest.mark.parametrize("photons,tol", [(2000, 0.05), (200, 0.15), (30, 0.3)])
def test_auto_cor_with_noise(photons, tol):
    errs = []
    for seed in range(4):
        a, b = make_pair(width=WIDTH, cor=WIDTH / 2 + 11.3, photons=photons, seed=seed)
        pa, pb = _prep(a, b)
        errs.append(abs(C.auto_cor(pa, pb).cor - (WIDTH / 2 + 11.3)))
    assert max(errs) < tol, errs


def test_smoothing_helps_when_noisy():
    a, b = make_pair(width=WIDTH, cor=WIDTH / 2 + 11.3, photons=30, seed=3)
    raw = C.auto_cor(a, b[:, ::-1])
    smooth = C.auto_cor(*_prep(a, b))
    assert smooth.confidence > raw.confidence


def test_matches_tomopy_convention():
    """Project a voxel object with tomopy at a known center and recover it.

    Integer centers only: tomopy.project appears to truncate fractional centers.
    (tomopy.find_center_pc reads 0.5 px lower than this convention on the same data.)
    """
    tomopy = pytest.importorskip("tomopy")
    n, slices = 128, 24
    rng = np.random.default_rng(5)
    obj = np.zeros((slices, n, n), dtype=np.float32)
    for _ in range(60):
        z, y, x = rng.integers(2, slices - 2), rng.integers(10, n - 10), rng.integers(10, n - 10)
        obj[z, y - 2 : y + 3, x - 2 : x + 3] = rng.uniform(0.5, 1.0)
    for center in (n / 2 + 6.0, n / 2 - 10.0):
        proj = tomopy.project(obj, np.deg2rad([0.0, 180.0]), center=center, pad=False)
        p0, p1 = proj[0].astype(np.float32), proj[1].astype(np.float32)
        res = C.auto_cor(*_prep(p0, p1, sigma=1.0), min_overlap_frac=0.5)
        assert res.cor == pytest.approx(center, abs=0.1)


def _banded_pair(seed=0):
    """Object plus large row-constant bands, as on real data with a wide, layered sample."""
    a, b = make_pair(width=WIDTH, height=256, cor=WIDTH / 2 + 11.3, photons=200, seed=seed)
    rng = np.random.default_rng(seed)
    bands = np.repeat(rng.normal(0, 3.0, 64), 4)[:, None]  # variance far above the object's
    return a + bands, b + bands


def test_row_offsets_dominate_the_metric_unless_removed():
    a, b = _banded_pair()
    plain = C.auto_cor(C.preprocess(a), C.preprocess(b[:, ::-1]))
    rows = C.auto_cor(
        C.preprocess(a, remove_row_offset=True), C.preprocess(b[:, ::-1], remove_row_offset=True)
    )
    true = WIDTH / 2 + 11.3
    assert plain.zncc > 0.98 and plain.confidence < 0.05  # the real-data symptom
    assert rows.confidence > 0.3
    assert rows.cor == pytest.approx(true, abs=0.3)
    assert rows.zncc < plain.zncc  # no longer riding on the row-constant pedestal


@pytest.mark.parametrize("photons", [2000, 30])
def test_row_offset_removal_does_not_hurt_unbanded_data(photons):
    a, b = make_pair(width=WIDTH, cor=WIDTH / 2 + 11.3, photons=photons)
    res = C.auto_cor(
        C.preprocess(a, remove_row_offset=True), C.preprocess(b[:, ::-1], remove_row_offset=True)
    )
    assert res.cor == pytest.approx(WIDTH / 2 + 11.3, abs=0.3)
