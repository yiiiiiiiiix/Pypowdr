"""
app_qt.py  —  powdR Native GUI
PyQt6 desktop application wrapping the afps() analysis modules.
"""

import sys, os, io, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QSplitter, QFrame, QScrollArea,
    QMessageBox, QProgressBar, QGroupBox, QSizePolicy, QHeaderView,
    QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import harmonise_data, align_sample
from fitting import (apply_nnls, optimise_coefficients, compute_concentrations,
                     compute_lods, compute_rwp, compute_r)


# ═══════════════════════════════════════════════════════════════════════════
# Stylesheet
# ═══════════════════════════════════════════════════════════════════════════
STYLE = """
QMainWindow, QWidget {
    background-color: #f5f7fa;
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #1a2332;
}
QGroupBox {
    border: 1px solid #d0dce8;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    background: white;
    font-weight: 600;
    color: #1a2332;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #005f73;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
QPushButton {
    background-color: #0077b6;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 7px 16px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover    { background-color: #005f8e; }
QPushButton:pressed  { background-color: #004a6e; }
QPushButton:disabled { background-color: #b0c4d4; color: #e0e8f0; }
QPushButton#run_btn {
    background-color: #0a9396;
    font-size: 14px;
    padding: 10px 24px;
}
QPushButton#run_btn:hover   { background-color: #007a7c; }
QPushButton#secondary_btn {
    background-color: white;
    color: #0077b6;
    border: 1.5px solid #0077b6;
}
QPushButton#secondary_btn:hover { background-color: #e8f4fa; }
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background: white;
    border: 1px solid #c8d8e4;
    border-radius: 4px;
    padding: 5px 8px;
    color: #1a2332;
}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #0077b6;
}
QLabel#file_label {
    color: #4a6278;
    font-size: 12px;
    padding: 4px 0;
}
QLabel#file_label[set="true"] { color: #0a9396; font-weight: 600; }
QTabWidget::pane {
    border: 1px solid #d0dce8;
    border-radius: 6px;
    background: white;
}
QTabBar::tab {
    background: #edf2f7;
    border: 1px solid #d0dce8;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 6px 16px;
    margin-right: 2px;
    color: #4a6278;
    font-weight: 500;
}
QTabBar::tab:selected { background: white; color: #0077b6; font-weight: 700; }
QTabBar::tab:hover    { background: #d8eaf5; }
QTableWidget {
    border: none;
    gridline-color: #e8eef4;
    background: white;
    alternate-background-color: #f5f8fb;
}
QTableWidget::item { padding: 4px 8px; }
QTableWidget::item:selected { background: #cce3f0; color: #1a2332; }
QHeaderView::section {
    background: #edf2f7;
    border: none;
    border-bottom: 2px solid #c0d4e4;
    padding: 5px 8px;
    font-weight: 700;
    color: #005f73;
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
QProgressBar {
    border: 1px solid #c8d8e4;
    border-radius: 4px;
    background: #edf2f7;
    text-align: center;
    color: #1a2332;
    font-weight: 600;
}
QProgressBar::chunk { background: #0a9396; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QStatusBar { background: #1a2332; color: #8fb8c8; font-size: 11px; padding: 2px 8px; }
QSplitter::handle { background: #d0dce8; width: 1px; }
"""


# ═══════════════════════════════════════════════════════════════════════════
# Worker thread — runs analysis off the main thread
# ═══════════════════════════════════════════════════════════════════════════
class AnalysisWorker(QThread):
    progress    = pyqtSignal(int, str)          # (percent, message)
    sample_done = pyqtSignal(str, dict)         # (name, result_dict)
    finished    = pyqtSignal(list)              # list of row dicts
    error       = pyqtSignal(str, str)          # (sample_name, traceback)

    def __init__(self, samples, ref_df, phases_df, cfg):
        super().__init__()
        self.samples   = samples    # [(name, tth, counts), ...]
        self.ref_df    = ref_df
        self.phases_df = phases_df
        self.cfg       = cfg

    def run(self):
        all_rows = []
        n = len(self.samples)
        for i, (name, tth, counts) in enumerate(self.samples):
            self.progress.emit(int(i / n * 100), f"Analysing {name}…")
            try:
                res = run_afps(tth, counts, self.ref_df, self.phases_df, self.cfg)
                self.sample_done.emit(name, res)
                row = {"sample_id": name, "Rwp": round(res["rwp"], 6)}
                for _, r in res["grouped"].iterrows():
                    row[r["phase_name"]] = round(r["wt_%"], 3)
                all_rows.append(row)
            except Exception:
                self.error.emit(name, traceback.format_exc())
        self.progress.emit(100, "Done")
        self.finished.emit(all_rows)


# ═══════════════════════════════════════════════════════════════════════════
# Core analysis (same logic as afps.py)
# ═══════════════════════════════════════════════════════════════════════════
def run_afps(tth_smpl, counts_smpl, ref_df, phases_df, cfg):
    tth_lib    = ref_df.index.to_numpy(dtype=float)
    ref_matrix = ref_df[phases_df["phase_id"].tolist()].to_numpy(dtype=float)

    active_ids  = phases_df["phase_id"].tolist()
    active_rirs = phases_df["rir"].to_numpy(dtype=float)
    active_ref  = ref_matrix.copy()
    std_id      = cfg["std_id"]
    std_conc    = cfg["std_conc"]
    std_idx     = active_ids.index(std_id)

    tth_work, counts_work, ref_work = harmonise_data(
        tth_smpl, counts_smpl, tth_lib, active_ref)

    if cfg["align"] != 0:
        std_col = ref_df[std_id].to_numpy(dtype=float)
        tth_work, counts_work = align_sample(
            tth_work, counts_work, tth_lib, std_col,
            align=cfg["align"], manual_align=False, tth_align_range=None)
        tth_work, counts_work, ref_work = harmonise_data(
            tth_work, counts_work, tth_lib, active_ref)

    mask = np.ones(len(tth_work), dtype=bool)
    if cfg["tth_min"] is not None: mask &= tth_work >= cfg["tth_min"]
    if cfg["tth_max"] is not None: mask &= tth_work <= cfg["tth_max"]
    tth_fit, counts_fit, ref_fit = tth_work[mask], counts_work[mask], ref_work[mask, :]

    coeffs = apply_nnls(counts_fit, ref_fit)

    zero_mask = coeffs == 0
    if zero_mask.any():
        keep = ~zero_mask
        active_ids  = [p for p, k in zip(active_ids, keep) if k]
        active_rirs = active_rirs[keep]; ref_fit = ref_fit[:, keep]; coeffs = coeffs[keep]
        std_idx = active_ids.index(std_id) if std_id in active_ids else 0

    coeffs = optimise_coefficients(counts_fit, ref_fit, coeffs, cfg["solver"], cfg["obj"])

    while True:
        le_zero = coeffs <= 0
        if not le_zero.any(): break
        keep = ~le_zero
        active_ids  = [p for p, k in zip(active_ids, keep) if k]
        active_rirs = active_rirs[keep]; ref_fit = ref_fit[:, keep]; coeffs = coeffs[keep]
        std_idx = active_ids.index(std_id) if std_id in active_ids else 0
        coeffs = optimise_coefficients(counts_fit, ref_fit, coeffs, cfg["solver"], cfg["obj"])

    concentrations = compute_concentrations(coeffs, active_rirs, std_idx, std_conc)
    rir_std = active_rirs[std_idx]
    lods    = compute_lods(active_rirs, cfg["lod"], rir_std)

    while True:
        below = concentrations < lods
        if not below.any(): break
        keep = ~below
        active_ids     = [p for p, k in zip(active_ids, keep) if k]
        active_rirs    = active_rirs[keep]; ref_fit = ref_fit[:, keep]
        coeffs         = coeffs[keep]; concentrations = concentrations[keep]; lods = lods[keep]
        std_idx = active_ids.index(std_id) if std_id in active_ids else 0
        coeffs = optimise_coefficients(counts_fit, ref_fit, coeffs, cfg["solver"], cfg["obj"])
        concentrations = compute_concentrations(coeffs, active_rirs, std_idx, std_conc)

    fitted  = ref_fit @ coeffs
    rwp_val = compute_rwp(counts_fit, fitted)
    r_val   = compute_r(counts_fit, fitted)
    id_to_name = dict(zip(phases_df["phase_id"], phases_df["phase_name"]))

    results = pd.DataFrame({
        "phase_id": active_ids,
        "phase_name": [id_to_name.get(p, p) for p in active_ids],
        "rir": active_rirs,
        "coefficient": coeffs,
        "concentration": concentrations,
        "lod": lods,
    })
    if cfg.get("omit_std") and std_id in results["phase_id"].values:
        results = results[results["phase_id"] != std_id].copy()

    grouped = (results.groupby("phase_name", as_index=False)["concentration"]
               .sum().rename(columns={"concentration": "wt_%"}))

    return dict(results=results, grouped=grouped, tth_fit=tth_fit,
                counts_fit=counts_fit, fitted=fitted, ref_fit=ref_fit,
                coeffs=coeffs, active_ids=active_ids, id_to_name=id_to_name,
                rwp=rwp_val, r=r_val)


def make_figure(res, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    cmap = plt.cm.get_cmap("tab10", max(len(res["active_ids"]), 1))
    ax1.plot(res["tth_fit"], res["counts_fit"], lw=0.9, color="#111", label="Experimental")
    ax1.plot(res["tth_fit"], res["fitted"],     lw=0.9, color="#e63946", ls="--", label="Fitted")
    for j, pid in enumerate(res["active_ids"]):
        nm = res["id_to_name"].get(pid, pid)
        c  = res["results"].loc[res["results"]["phase_id"]==pid, "concentration"].values
        ax1.fill_between(res["tth_fit"], 0, res["ref_fit"][:,j]*res["coeffs"][j],
                         alpha=0.3, color=cmap(j),
                         label=f"{nm} ({c[0]:.1f} wt-%)" if len(c) else nm)
    ax1.set_ylabel("Counts", fontsize=9)
    ax1.set_title(f"{name}  ·  Rwp={res['rwp']:.4f}  R={res['r']:.4f}",
                  fontsize=9, fontfamily="monospace")
    ax1.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax1.spines[["top","right"]].set_visible(False)
    ax2.plot(res["tth_fit"], res["counts_fit"]-res["fitted"], lw=0.7, color="#555")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("Residual", fontsize=8); ax2.set_xlabel("2θ (°)", fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=2.5)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.11)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# File picker row widget
# ═══════════════════════════════════════════════════════════════════════════
class FilePicker(QWidget):
    def __init__(self, label, multi=False, parent=None):
        super().__init__(parent)
        self.multi = multi
        self.paths = []
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)
        self.lbl = QLabel("No file selected"); self.lbl.setObjectName("file_label")
        self.lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        btn = QPushButton("Browse…"); btn.setObjectName("secondary_btn")
        btn.setFixedWidth(90); btn.clicked.connect(self.browse)
        lay.addWidget(self.lbl); lay.addWidget(btn)

    def browse(self):
        if self.multi:
            paths, _ = QFileDialog.getOpenFileNames(self, "Select sample CSVs", "", "CSV (*.csv)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", "", "CSV (*.csv)")
            paths = [path] if path else []
        if paths:
            self.paths = paths
            if self.multi:
                self.lbl.setText(f"{len(paths)} file(s) selected")
            else:
                self.lbl.setText(Path(paths[0]).name)
            self.lbl.setProperty("set", "true")
            self.lbl.style().unpolish(self.lbl); self.lbl.style().polish(self.lbl)

    def get(self):
        return self.paths if self.multi else (self.paths[0] if self.paths else None)


# ═══════════════════════════════════════════════════════════════════════════
# Results tab widget (one per sample)
# ═══════════════════════════════════════════════════════════════════════════
class SampleResultWidget(QWidget):
    def __init__(self, name, res, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)

        # Metric bar
        metrics = QHBoxLayout(); metrics.setSpacing(12)
        for val, lbl in [(f"{res['rwp']:.4f}", "Rwp"),
                         (f"{res['r']:.4f}",   "R"),
                         (str(len(res['active_ids'])), "Phases retained")]:
            tile = QFrame(); tile.setFrameShape(QFrame.Shape.StyledPanel)
            tile.setStyleSheet("QFrame{background:#f0f7fa;border:1px solid #cce4ee;border-radius:6px;padding:6px;}")
            tl = QVBoxLayout(tile); tl.setContentsMargins(10,6,10,6); tl.setSpacing(2)
            v = QLabel(val); v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.setStyleSheet("font-family:monospace;font-size:18px;font-weight:700;color:#005f73;")
            l = QLabel(lbl); l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setStyleSheet("font-size:10px;color:#777;")
            tl.addWidget(v); tl.addWidget(l)
            metrics.addWidget(tile)
        lay.addLayout(metrics)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Plot
        fig = make_figure(res, name)
        canvas = FigureCanvas(fig)
        canvas.setMinimumWidth(480)
        splitter.addWidget(canvas)

        # Tables
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(8)

        rl.addWidget(QLabel("<b>Phase concentrations (wt-%)</b>"))
        grouped = res["grouped"].copy().round(3)
        t1 = self._make_table(grouped)
        rl.addWidget(t1)

        rl.addWidget(QLabel("<b>Detailed results</b>"))
        detail = res["results"][["phase_id","phase_name","concentration","lod"]].copy().round(3)
        t2 = self._make_table(detail)
        rl.addWidget(t2)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2); splitter.setStretchFactor(1, 3)
        splitter.setSizes([480, 720])
        lay.addWidget(splitter)

        # Save plot button
        self._fig = fig
        self._name = name
        save_btn = QPushButton("⬇  Save plot as PNG…"); save_btn.setObjectName("secondary_btn")
        save_btn.clicked.connect(self.save_plot)
        lay.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignLeft)

    def _make_table(self, df):
        t = QTableWidget(len(df), len(df.columns))
        t.setHorizontalHeaderLabels([c.replace("_"," ").title() for c in df.columns])
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setMaximumHeight(180)
        for r, row in enumerate(df.itertuples(index=False)):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                t.setItem(r, c, item)
        return t

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", f"{self._name}.png",
                                               "PNG (*.png);;PDF (*.pdf)")
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")


# ═══════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("powdR  ·  XRPD Quantitative Analysis")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)
        self._all_rows = []
        self._worker   = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # Main splitter — left panel is now resizable by dragging
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setHandleWidth(5)
        self._main_splitter.setStyleSheet("QSplitter::handle{background:#c0cdd8;width:5px;}")
        root.addWidget(self._main_splitter)

        # ── Left panel (settings) ──────────────────────────────────────────
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(240)
        left_scroll.setStyleSheet("QScrollArea{background:#f0f4f8;border:none;}")
        left_inner = QWidget()
        left_scroll.setWidget(left_inner)
        ll = QVBoxLayout(left_inner); ll.setContentsMargins(14,14,14,14); ll.setSpacing(14)

        # Header
        hdr = QLabel("⬡  powdR"); hdr.setStyleSheet(
            "font-size:20px;font-weight:700;color:#005f73;font-family:monospace;")
        sub = QLabel("XRPD Quantitative Analysis"); sub.setStyleSheet("color:#4a6278;font-size:11px;")
        ll.addWidget(hdr); ll.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#d0dce8;"); ll.addWidget(sep)

        # Files group
        fg = QGroupBox("Input Files"); fl = QVBoxLayout(fg); fl.setSpacing(6)
        fl.addWidget(QLabel("Reference patterns CSV"))
        self.ref_picker = FilePicker("ref"); fl.addWidget(self.ref_picker)
        fl.addWidget(QLabel("Phases CSV"))
        self.phases_picker = FilePicker("phases"); fl.addWidget(self.phases_picker)
        fl.addWidget(QLabel("Sample CSV(s)"))
        self.samples_picker = FilePicker("samples", multi=True); fl.addWidget(self.samples_picker)
        ll.addWidget(fg)

        # Settings group
        sg = QGroupBox("Algorithm Settings"); sl = QGridLayout(sg); sl.setSpacing(8)

        sl.addWidget(QLabel("Internal standard ID"), 0, 0)
        self.std_id = QLineEdit("COR"); sl.addWidget(self.std_id, 0, 1)

        sl.addWidget(QLabel("Std conc (wt-%, 0=sum to 100%)"), 1, 0)
        self.std_conc = QDoubleSpinBox()
        self.std_conc.setRange(0, 100); self.std_conc.setValue(0); self.std_conc.setSingleStep(0.5)
        sl.addWidget(self.std_conc, 1, 1)

        sl.addWidget(QLabel("Solver"), 2, 0)
        self.solver = QComboBox(); self.solver.addItems(["BFGS","Nelder-Mead","CG"])
        sl.addWidget(self.solver, 2, 1)

        sl.addWidget(QLabel("Objective"), 3, 0)
        self.obj = QComboBox(); self.obj.addItems(["Rwp","R","Delta"])
        sl.addWidget(self.obj, 3, 1)

        sl.addWidget(QLabel("Max alignment (°)"), 4, 0)
        self.align = QDoubleSpinBox(); self.align.setRange(0,2); self.align.setValue(0.2)
        self.align.setSingleStep(0.05); sl.addWidget(self.align, 4, 1)

        sl.addWidget(QLabel("LOD (wt-%)"), 5, 0)
        self.lod = QDoubleSpinBox(); self.lod.setRange(0,5); self.lod.setValue(0.5)
        self.lod.setSingleStep(0.1); sl.addWidget(self.lod, 5, 1)

        sl.addWidget(QLabel("2θ min"), 6, 0)
        self.tth_min = QDoubleSpinBox(); self.tth_min.setRange(0,180); self.tth_min.setValue(5.0)
        sl.addWidget(self.tth_min, 6, 1)

        sl.addWidget(QLabel("2θ max"), 7, 0)
        self.tth_max = QDoubleSpinBox(); self.tth_max.setRange(0,180); self.tth_max.setValue(70.0)
        sl.addWidget(self.tth_max, 7, 1)

        self.omit_std = QCheckBox("Omit std from output")
        sl.addWidget(self.omit_std, 8, 0, 1, 2)
        ll.addWidget(sg)

        # Run button + progress
        self.run_btn = QPushButton("▶   Run Analysis"); self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self.run_analysis)
        ll.addWidget(self.run_btn)

        self.progress = QProgressBar(); self.progress.setValue(0)
        self.progress.setVisible(False); ll.addWidget(self.progress)

        self.export_btn = QPushButton("⬇  Export summary CSV"); self.export_btn.setObjectName("secondary_btn")
        self.export_btn.setEnabled(False); self.export_btn.clicked.connect(self.export_csv)
        ll.addWidget(self.export_btn)

        ll.addStretch()

        # ── Right panel (results tabs) ─────────────────────────────────────
        right_container = QWidget()
        rl = QVBoxLayout(right_container); rl.setContentsMargins(16,16,16,16)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(False)

        # Welcome tab
        welcome = QWidget(); wl = QVBoxLayout(welcome); wl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl.addStretch()
        wl_title = QLabel("Welcome to powdR"); wl_title.setStyleSheet(
            "font-size:24px;font-weight:700;color:#005f73;font-family:monospace;")
        wl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl_sub = QLabel("Load your files in the panel on the left, configure settings,\nthen click Run Analysis.")
        wl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl_sub.setStyleSheet("color:#4a6278;font-size:14px;line-height:1.6;")
        steps = QLabel("1  Upload reference patterns CSV\n2  Upload phases CSV\n3  Upload sample CSV(s)\n4  Click ▶ Run Analysis")
        steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        steps.setStyleSheet(
            "background:#f0f7fa;border:1px solid #cce4ee;border-radius:8px;"
            "padding:20px 40px;color:#1a2332;font-size:13px;line-height:2;")
        wl.addWidget(wl_title); wl.addSpacing(12); wl.addWidget(wl_sub)
        wl.addSpacing(24); wl.addWidget(steps, alignment=Qt.AlignmentFlag.AlignCenter)
        wl.addStretch()
        self.tabs.addTab(welcome, "Home")

        rl.addWidget(self.tabs)
        self._main_splitter.addWidget(left_scroll)
        self._main_splitter.addWidget(right_container)
        self._main_splitter.setSizes([500, 980])
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)

        # Status bar
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    # ── Run ────────────────────────────────────────────────────────────────
    def run_analysis(self):
        ref_path    = self.ref_picker.get()
        phases_path = self.phases_picker.get()
        sample_paths = self.samples_picker.get()

        if not ref_path:
            QMessageBox.warning(self, "Missing file", "Please select a reference patterns CSV.")
            return
        if not phases_path:
            QMessageBox.warning(self, "Missing file", "Please select a phases CSV.")
            return
        if not sample_paths:
            QMessageBox.warning(self, "Missing file", "Please select at least one sample CSV.")
            return

        try:
            ref_df    = pd.read_csv(ref_path, index_col=0)
            ref_df.index = ref_df.index.astype(float)
            phases_df = pd.read_csv(phases_path)
            phases_df["rir"] = phases_df["rir"].astype(float)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e)); return

        std_id = self.std_id.text().strip()
        if std_id not in phases_df["phase_id"].values:
            QMessageBox.critical(self, "Error", f"Standard ID '{std_id}' not in phases CSV."); return
        if std_id not in ref_df.columns:
            QMessageBox.critical(self, "Error", f"Standard ID '{std_id}' not in reference CSV."); return

        samples = []
        for p in sample_paths:
            try:
                df = pd.read_csv(p)
                if list(df.columns[:2]) != ["tth","counts"]:
                    df = df.iloc[:,:2]; df.columns = ["tth","counts"]
                samples.append((Path(p).stem, df["tth"].to_numpy(float), df["counts"].to_numpy(float)))
            except Exception as e:
                QMessageBox.warning(self, "Load warning", f"Could not load {p}:\n{e}")

        if not samples: return

        std_conc_val = self.std_conc.value()
        cfg = {
            "std_id":   std_id,
            "std_conc": None if std_conc_val == 0 else std_conc_val,
            "solver":   self.solver.currentText(),
            "obj":      self.obj.currentText(),
            "align":    self.align.value(),
            "lod":      self.lod.value(),
            "tth_min":  self.tth_min.value(),
            "tth_max":  self.tth_max.value(),
            "omit_std": self.omit_std.isChecked(),
        }

        # Remove old result tabs (keep Home tab at index 0)
        while self.tabs.count() > 1:
            self.tabs.removeTab(1)
        self._all_rows = []

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True); self.progress.setValue(0)
        self.export_btn.setEnabled(False)
        self.status.showMessage("Running analysis…")

        self._worker = AnalysisWorker(samples, ref_df, phases_df, cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.sample_done.connect(self._on_sample_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.status.showMessage(msg)

    def _on_sample_done(self, name, res):
        w = SampleResultWidget(name, res)
        self.tabs.addTab(w, name)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)

    def _on_finished(self, rows):
        self._all_rows = rows
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.export_btn.setEnabled(bool(rows))
        self.status.showMessage(f"Complete — {len(rows)} sample(s) processed.")

        if rows:
            # Summary tab
            summary_df = pd.DataFrame(rows).fillna(0)
            sw = QWidget(); sl2 = QVBoxLayout(sw); sl2.setContentsMargins(12,12,12,12)
            sl2.addWidget(QLabel("<b>Batch summary (wt-%)</b>"))
            t = QTableWidget(len(summary_df), len(summary_df.columns))
            t.setHorizontalHeaderLabels([c.replace("_"," ").title() for c in summary_df.columns])
            t.setAlternatingRowColors(True); t.verticalHeader().setVisible(False)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for r, row in enumerate(summary_df.itertuples(index=False)):
                for c, val in enumerate(row):
                    item = QTableWidgetItem(str(round(val,4) if isinstance(val,float) else val))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    t.setItem(r, c, item)
            sl2.addWidget(t)
            self.tabs.addTab(sw, "Summary")
            self.tabs.setCurrentWidget(sw)

    def _on_error(self, name, tb):
        QMessageBox.warning(self, f"Error — {name}",
                            f"Analysis failed for {name}:\n\n{tb[:600]}")

    def export_csv(self):
        if not self._all_rows: return
        path, _ = QFileDialog.getSaveFileName(self, "Save summary CSV",
                                               "powdr_results.csv", "CSV (*.csv)")
        if path:
            pd.DataFrame(self._all_rows).fillna(0).to_csv(path, index=False)
            self.status.showMessage(f"Saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setApplicationName("powdR")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
