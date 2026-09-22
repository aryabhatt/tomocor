import h5py
import numpy as np
import pytest

from tomocor import cor as C
from tomocor import io as tio
from tomocor.synthetic import write_h5

WIDTH = 256


@pytest.fixture(scope="module")
def h5file(tmp_path_factory):
    path = tmp_path_factory.mktemp("h5") / "scan.h5"
    write_h5(path, width=WIDTH, height=64, cor=WIDTH / 2 + 7.0, photons=3000)
    return path


def test_select_frames():
    theta = np.array([0, 0.3, 45, 90, 179.7, 180.0])
    idx, err = tio.select_frames(theta, 180.0, 1.0)
    assert list(idx) == [4, 5] and err == 0.0
    idx, _ = tio.select_frames(theta, 180.0, 0.0)
    assert list(idx) == [5]
    idx, err = tio.select_frames(theta, 60.0, 1.0)  # nothing in tolerance: nearest is used
    assert list(idx) == [2] and err == 15.0


def test_load_pair_averages_and_recovers_cor(h5file):
    pair = tio.load_pair(h5file, avg_tol_deg=1.0)
    assert pair.proj0.shape == pair.proj180.shape == (64, WIDTH)
    assert (pair.n0, pair.n180) == (2, 2)
    assert pair.theta0 == pytest.approx(0.15, abs=1e-6)
    assert pair.theta180 == pytest.approx(179.85, abs=1e-4)
    a, bf = C.preprocess(pair.proj0), C.preprocess(pair.proj180[:, ::-1])
    assert C.auto_cor(a, bf).cor == pytest.approx(WIDTH / 2 + 7.0, abs=0.3)


def test_load_pair_defaults_to_single_frames(h5file):
    pair = tio.load_pair(h5file)
    assert (pair.n0, pair.n180) == (1, 1)
    assert (pair.theta0, pair.theta180) == (0.0, 180.0)


def test_missing_dataset_message(tmp_path):
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f["/exchange/data"] = np.zeros((2, 4, 4))
    with pytest.raises(KeyError, match="data_white"):
        tio.load_pair(p)


def test_one_sided_window_warns(h5file):
    with pytest.warns(UserWarning, match="not 180"):
        tio.load_pair(h5file, avg_tol_deg=45.0)  # 0/0.3/45 vs 135/179.7/180: 179.x apart
