"""pyqtgraph GUI: overlay 0 deg and flipped 180 deg projections, slide horizontally."""

import argparse
import json
import math
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import cor as C
from .io import load_pair

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

MIN_OVERLAP_FRAC = 0.3
LOW_CONFIDENCE = 0.05
MODES = ("Difference", "Red / cyan", "Blend 50%")
_DIFF_LUT = pg.ColorMap(
    [0.0, 0.5, 1.0], [(40, 80, 220), (245, 245, 245), (220, 50, 40)]
).getLookupTable(0.0, 1.0, 256)


class HShiftViewBox(pg.ViewBox):
    """Left-drag slides the overlay horizontally (never vertically); other buttons as usual."""

    sigShiftDrag = QtCore.Signal(float)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            dx = self.mapToView(ev.pos()).x() - self.mapToView(ev.lastPos()).x()
            self.sigShiftDrag.emit(dx)
        else:
            super().mouseDragEvent(ev, axis)


def _normalize(img):
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)


class CorWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None, avg_tol_deg=0.0):
        super().__init__()
        self.setWindowTitle("tomocor: center of rotation")
        self.avg_tol_deg = avg_tol_deg
        self.pair = None
        self.result = None  # (cor, shift, score) once accepted
        self._shift = 0.0
        self._width = 0
        self._build_ui()
        self._recompute_timer = QtCore.QTimer(self, singleShot=True, interval=250)
        self._recompute_timer.timeout.connect(self.recompute)
        self._set_controls_enabled(False)
        if path:
            self.load(path)

    # ---- UI construction -------------------------------------------------------------
    def _build_ui(self):
        self.vb = HShiftViewBox(invertY=True, lockAspect=True)
        self.vb.sigShiftDrag.connect(lambda dx: self.set_shift(self._shift + dx))
        self.image_view = pg.GraphicsLayoutWidget()
        self.image_view.addItem(self.vb)
        self.base = pg.ImageItem()  # dimmed 0 deg image: visible outside the overlap
        self.base.setOpacity(0.6)
        self.over = pg.ImageItem()  # composite of the overlap only
        self.vb.addItem(self.base)
        self.vb.addItem(self.over)

        self.curve_plot, self.curve_item, self.cur_line, self.auto_line = self._make_curve_plot(
            "ZNCC", "line"
        )
        self.norm_plot, self.norm_item, self.cur_line2, self.auto_line2 = self._make_curve_plot(
            "Phase correlation", "line2"
        )
        self.indicator_tabs = QtWidgets.QTabWidget()
        self.indicator_tabs.addTab(self.curve_plot, "ZNCC")
        self.indicator_tabs.addTab(self.norm_plot, "Phase corr (tomopy-style)")

        self.readout = QtWidgets.QLabel()
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout.setMinimumHeight(165)
        self.readout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.cor_spin = QtWidgets.QDoubleSpinBox(decimals=2, singleStep=0.5, keyboardTracking=False)
        self.cor_spin.valueChanged.connect(
            lambda v: self.set_shift(C.cor_to_shift(v, self._width), "spin")
        )
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)  # tenths of a pixel
        self.slider.valueChanged.connect(lambda v: self.set_shift(v / 10.0, "slider"))

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(MODES)
        self.mode.currentIndexChanged.connect(lambda _: self.refresh())
        self.contrast = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.contrast.setRange(5, 200)
        self.contrast.setValue(50)
        self.contrast.valueChanged.connect(lambda _: self.refresh())

        self.median = QtWidgets.QCheckBox("Median 3x3 (zingers)")
        self.median.setChecked(True)
        self.sigma = QtWidgets.QDoubleSpinBox(decimals=1, singleStep=0.5, minimum=0, maximum=20)
        self.sigma.setValue(3.0)
        self.row_offset = QtWidgets.QCheckBox("Remove row offsets (bands)")
        self.row_offset.setChecked(True)
        self.bandpass = QtWidgets.QCheckBox("Band-pass (remove 2D background)")
        for w in (self.median, self.row_offset, self.bandpass):
            w.toggled.connect(self._schedule_recompute)
        self.sigma.valueChanged.connect(self._schedule_recompute)

        self.snap_btn = QtWidgets.QPushButton("Snap to auto")
        self.snap_btn.clicked.connect(lambda: self.set_shift(self.auto.shift))
        self.open_btn = QtWidgets.QPushButton("Open…")
        self.open_btn.clicked.connect(self.open_file)
        self.accept_btn = QtWidgets.QPushButton("Accept COR")
        self.accept_btn.clicked.connect(self.accept)

        form = QtWidgets.QFormLayout()
        form.addRow("COR (px)", self.cor_spin)
        form.addRow("Shift (px)", self.slider)
        form.addRow("Display", self.mode)
        form.addRow("Contrast", self.contrast)
        form.addRow("Gaussian σ (px)", self.sigma)
        form.addRow(self.median)
        form.addRow(self.row_offset)
        form.addRow(self.bandpass)
        buttons = QtWidgets.QHBoxLayout()
        for b in (self.open_btn, self.snap_btn, self.accept_btn):
            buttons.addWidget(b)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.indicator_tabs, stretch=2)
        side.addWidget(self.readout)
        side.addLayout(form)
        side.addLayout(buttons)
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(380)
        self._controls = [side_widget]

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self.image_view, stretch=1)
        layout.addWidget(side_widget)
        self.setCentralWidget(central)
        self.resize(1400, 800)

        Key = QtCore.Qt.Key
        Mod = QtCore.Qt.KeyboardModifier
        for key, mod, step in (
            (Key.Key_Left, Mod.NoModifier, -1.0),
            (Key.Key_Right, Mod.NoModifier, 1.0),
            (Key.Key_Left, Mod.ShiftModifier, -0.1),
            (Key.Key_Right, Mod.ShiftModifier, 0.1),
        ):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key | mod), self)
            sc.activated.connect(lambda step=step: self.set_shift(self._shift + step))

    def _make_curve_plot(self, ylabel, name):
        """Plot of a metric vs COR with a draggable current-position line and a dashed auto line."""
        plot = pg.PlotWidget()
        plot.setLabel("bottom", "COR (px)")
        plot.setLabel("left", ylabel)
        plot.showGrid(x=True, y=True, alpha=0.3)
        for axis in ("bottom", "left"):
            plot.getAxis(axis).enableAutoSIPrefix(False)
        item = plot.plot(pen=pg.mkPen("c", width=1.5), connect="finite")
        cur = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("y", width=2))
        auto = pg.InfiniteLine(angle=90, pen=pg.mkPen("g", style=QtCore.Qt.PenStyle.DashLine))
        plot.addItem(auto)
        plot.addItem(cur)
        cur.sigPositionChanged.connect(
            lambda: self.set_shift(C.cor_to_shift(cur.value(), self._width), name)
        )
        return plot, item, cur, auto

    def _set_controls_enabled(self, on):
        for w in self._controls:
            w.setEnabled(on)
        self.open_btn.setEnabled(True)
        self.image_view.setEnabled(on)

    # ---- data ------------------------------------------------------------------------
    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open projections", "", "HDF5 (*.h5 *.hdf5 *.hdf *.nxs);;All files (*)"
        )
        if path:
            self.load(path)

    def load(self, path):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            pair = load_pair(path, self.avg_tol_deg)
        except Exception as exc:  # show any loader problem instead of dying in a slot
            QtWidgets.QMessageBox.critical(self, "Cannot load file", f"{path}\n\n{exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.set_pair(pair)

    def set_pair(self, pair):
        self.pair = pair
        self._raw0 = pair.proj0
        self._rawbf = np.ascontiguousarray(pair.proj180[:, ::-1])
        h, w = self._raw0.shape
        self._width = w
        self.setWindowTitle(f"tomocor: {pair.path}")
        self.vb.setRange(xRange=(0, w), yRange=(0, h), padding=0)
        # -1: a fractional shift loses one interpolated column at the far edge
        self.max_shift = math.floor(w * (1.0 - MIN_OVERLAP_FRAC)) - 1.0
        self.cor_spin.blockSignals(True)
        self.cor_spin.setRange(C.shift_to_cor(-self.max_shift, w), C.shift_to_cor(self.max_shift, w))
        self.cor_spin.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setRange(-int(self.max_shift * 10), int(self.max_shift * 10))
        self.slider.blockSignals(False)
        self._set_controls_enabled(True)
        self.recompute(snap=True)

    def _schedule_recompute(self, *_):
        self._recompute_timer.start()

    def recompute(self, snap=False):
        """Redo preprocessing, the ZNCC curve and the auto estimate (keeps the current shift)."""
        if self.pair is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            median = 3 if self.median.isChecked() else 0
            # disp*: smoothed only (composite views). met*: also what the metric and the
            # difference view see (row offsets and/or background removed).
            self.disp0 = C.denoise(self._raw0, median, self.sigma.value())
            self.dispbf = C.denoise(self._rawbf, median, self.sigma.value())
            opts = dict(
                highpass_sigma=20.0 if self.bandpass.isChecked() else None,
                remove_row_offset=self.row_offset.isChecked(),
            )
            self.met0 = C.metric_view(self.disp0, **opts)
            self.metbf = C.metric_view(self.dispbf, **opts)
            self.norm0, self.normbf = _normalize(self.disp0), _normalize(self.dispbf)
            self.diff_scale = float(np.std(self.met0)) or 1.0
            self.curve = C.shift_curve(self.met0, self.metbf, MIN_OVERLAP_FRAC)
            self.auto = C.auto_cor(self.met0, self.metbf, curve=self.curve)
            self.pc_shifts, self.pc_corr = C.phase_corr_curve(self.met0, self.metbf)
            self.pc = C.find_center_pc(self.met0, self.metbf)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        cor_axis = C.shift_to_cor(self.curve.shifts, self._width)
        self.curve_item.setData(cor_axis, self.curve.zncc)
        self.norm_item.setData(C.shift_to_cor(self.pc_shifts, self._width), self.pc_corr)
        self.auto_line.setValue(self.auto.cor)
        self.auto_line2.setValue(self.pc.cor)
        self.base.setImage(self.norm0, levels=(0, 1))
        self.set_shift(self.auto.shift if snap else self._shift, force=True)

    # ---- state -----------------------------------------------------------------------
    def set_shift(self, shift, source=None, force=False):
        if self.pair is None or not math.isfinite(shift):
            return
        shift = float(np.clip(shift, -self.max_shift, self.max_shift))
        if shift == self._shift and not force:
            return
        self._shift = shift
        cor = C.shift_to_cor(shift, self._width)
        for widget, value, name in (
            (self.cor_spin, cor, "spin"),
            (self.slider, round(shift * 10), "slider"),
            (self.cur_line, cor, "line"),
            (self.cur_line2, cor, "line2"),
        ):
            if source != name:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        self.refresh()

    @property
    def shift(self):
        return self._shift

    @property
    def cor(self):
        return C.shift_to_cor(self._shift, self._width)

    def current_score(self):
        return C.score_at(self.met0, self.metbf, self._shift)

    def refresh(self):
        if self.pair is None:
            return
        mode = self.mode.currentText()
        if mode == MODES[0]:
            ov = C.overlap(self.met0, self.metbf, self._shift)
        else:
            ov = C.overlap(self.norm0, self.normbf, self._shift)
        if ov is None:
            self.over.clear()
        else:
            x0, fa, gb = ov
            if mode == MODES[0]:
                diff = fa - gb
                diff -= diff.mean()  # a uniform offset (beam decay) is not misalignment
                half = self.diff_scale * 50.0 / self.contrast.value()
                self.over.setImage(diff, levels=(-half, half), lut=_DIFF_LUT, autoLevels=False)
            else:
                if mode == MODES[1]:
                    rgb = np.stack([fa, gb, gb], axis=-1)
                else:
                    rgb = np.repeat(((fa + gb) * 0.5)[..., None], 3, axis=-1)
                self.over.setImage(
                    (rgb * 255).astype(np.uint8), levels=(0, 255), lut=None, autoLevels=False
                )
            self.over.setPos(x0, 0)
        self._update_readout()

    def _update_readout(self):
        s = self.current_score()
        w = self._width
        low = self.auto.confidence < LOW_CONFIDENCE
        warn = (
            "<br><b style='color:#e67e22'>Low confidence: check the overlay by eye.</b>"
            if low
            else ""
        )
        self.readout.setText(
            "<table cellspacing=3>"
            f"<tr><td>COR</td><td><b>{self.cor:.2f}</b> px</td></tr>"
            f"<tr><td>shift</td><td>{self._shift:+.2f} px</td></tr>"
            f"<tr><td>ZNCC</td><td><b>{s.zncc:.4f}</b></td></tr>"
            f"<tr><td>RMS diff</td><td>{s.rms:.4f}</td></tr>"
            f"<tr><td>overlap</td><td>{s.overlap_width} / {w} px ({100 * s.overlap_frac:.1f}%)</td></tr>"
            f"<tr><td>auto (ZNCC)</td><td>COR {self.auto.cor:.2f}, ZNCC {self.auto.zncc:.4f}, "
            f"confidence {self.auto.confidence:.2f}</td></tr>"
            f"<tr><td>phase corr</td><td>COR {self.pc.cor:.2f}, peak {self.pc.peak:.3f}</td></tr>"
            f"<tr><td>frames</td><td>{self.pair.n0} @ {self.pair.theta0:.2f}° / "
            f"{self.pair.n180} @ {self.pair.theta180:.2f}° "
            f"(Δθ {self.pair.theta180 - self.pair.theta0:.2f}°)</td></tr>"
            f"</table>{warn}"
        )

    def accept(self):
        if self.pair is None:
            return
        self.result = (self.cor, self._shift, self.current_score())
        self.close()


def run(path=None, avg_tol_deg=0.0):
    """Open the window (with a file dialog if ``path`` is None); return the accepted result."""
    app = pg.mkQApp("tomocor")
    win = CorWindow(path, avg_tol_deg)
    win.show()
    if path is None:
        QtCore.QTimer.singleShot(0, win.open_file)
    app.exec()
    return win.result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tomocor", description=__doc__)
    parser.add_argument("path", nargs="?", help="HDF5 file (file dialog if omitted)")
    parser.add_argument(
        "--tol", type=float, default=0.0, help="average frames within this many degrees (default: off)"
    )
    parser.add_argument("--out", help="write the accepted result to this JSON file")
    args = parser.parse_args(argv)

    result = run(args.path, args.tol)
    if result is None:
        print("No COR accepted.", file=sys.stderr)
        return 1
    cor, shift, score = result
    print(f"COR: {cor:.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"file": args.path, "cor": cor, "shift": shift, "zncc": score.zncc, "rms": score.rms},
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
