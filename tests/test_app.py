import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pg = pytest.importorskip("pyqtgraph")

from tomocor import cor as C
from tomocor.app import CorWindow
from tomocor.io import ProjectionPair
from tomocor.synthetic import make_pair

WIDTH, TRUE_COR = 512, 512 / 2 + 11.3


@pytest.fixture(scope="module")
def win():
    pg.mkQApp("test")
    a, b = make_pair(width=WIDTH, height=128, cor=TRUE_COR, photons=100)
    w = CorWindow()
    w.set_pair(ProjectionPair(a, b, 0.0, 180.0, 1, 1, "synthetic"))
    yield w
    w.close()


def test_starts_at_auto_estimate(win):
    assert win.cor == pytest.approx(TRUE_COR, abs=0.3)
    assert win.cor_spin.value() == pytest.approx(win.cor, abs=0.01)
    assert win.cur_line.value() == pytest.approx(win.cor, abs=0.01)


def test_readout_matches_metric(win):
    win.set_shift(-30.0)
    sc = C.score_at(win.met0, win.metbf, -30.0)
    assert win.current_score().zncc == pytest.approx(sc.zncc)
    i = int(np.where(win.curve.shifts == -30)[0][0])
    assert win.curve.zncc[i] == pytest.approx(sc.zncc, abs=1e-6)
    assert f"{sc.zncc:.4f}" in win.readout.text()


def test_overlay_is_masked_to_overlap_and_history_free(win):
    def snapshot(shift):
        win.set_shift(shift)
        return win.over.image.copy(), win.over.pos().x()

    img1, x1 = snapshot(60.0)
    assert img1.shape[1] == WIDTH - 60 and x1 == 60
    for s in (-60.0, 13.0, -200.0):
        snapshot(s)
    img2, x2 = snapshot(60.0)
    np.testing.assert_array_equal(img1, img2)
    assert x1 == x2
    img_neg, x_neg = snapshot(-60.0)
    assert img_neg.shape[1] == WIDTH - 60 and x_neg == 0


def test_shift_is_clamped_to_min_overlap(win):
    for extreme in (10_000.0, -10_000.0):
        win.set_shift(extreme)
        assert abs(win.shift) < WIDTH * 0.7
        assert win.current_score().overlap_frac >= 0.3
    win.set_shift(win.shift + 0.5)  # fractional shift at the limit is still clamped correctly
    assert win.current_score().overlap_frac >= 0.3


@pytest.mark.parametrize("mode", range(3))
def test_all_display_modes_render(win, mode):
    win.mode.setCurrentIndex(mode)
    win.set_shift(win.auto.shift)
    assert win.over.image is not None
    win.mode.setCurrentIndex(0)


def test_keyboard_step_and_snap(win):
    win.set_shift(win.auto.shift)
    win.set_shift(win.shift + 1.0)
    win.snap_btn.click()
    assert win.shift == pytest.approx(win.auto.shift)


def test_recompute_keeps_user_shift(win):
    win.set_shift(25.0)
    win.sigma.setValue(2.0)
    win.recompute()
    assert win.shift == 25.0
    win.sigma.setValue(3.0)
    win.recompute(snap=True)
    assert win.cor == pytest.approx(TRUE_COR, abs=0.3)


def test_accept_returns_result(win):
    win.snap_btn.click()
    win.accept()
    cor, shift, score = win.result
    assert cor == pytest.approx(TRUE_COR, abs=0.3)
    assert score.zncc > 0.9


def test_row_offset_option_defaults_on_and_can_be_toggled(win):
    assert win.row_offset.isChecked()
    on = win.met0
    win.row_offset.setChecked(False)
    win.recompute()
    assert not np.allclose(win.met0.mean(axis=1), 0, atol=1e-6)
    assert np.allclose(on.mean(axis=1), 0, atol=1e-5)
    win.row_offset.setChecked(True)
    win.recompute(snap=True)
    assert win.cor == pytest.approx(TRUE_COR, abs=0.3)


def test_indicator_tabs_and_norm_line_follow_shift(win):
    assert win.indicator_tabs.count() == 2
    win.set_shift(12.0)
    assert win.cur_line2.value() == pytest.approx(win.cor, abs=0.01)
    pc_shifts, pc_corr = C.phase_corr_curve(win.met0, win.metbf)
    i = int(np.where(pc_shifts == 12)[0][0])
    assert win.norm_item.yData[i] == pytest.approx(pc_corr[i], rel=1e-6)
    assert win.auto_line2.value() == pytest.approx(win.pc.cor, abs=0.01)


def test_cor_spin_steps_half_a_pixel(win):
    assert win.cor_spin.singleStep() == 0.5
