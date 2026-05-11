"""
plotting.py
-----------
Plotting utilities for afps() batch analysis.

Each sample produces one PNG file saved to PLOTS_FOLDER.
After all samples are processed, all PNGs are opened automatically
using the OS default image viewer.
"""

import os
import sys
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_fit_plot(tth_fit, counts_fit, fitted, ref_fit, coeffs,
                  concentrations, active_ids, id_to_name,
                  rwp_val, r_val, sample_name, plots_folder):
    """
    Save a fitted-vs-experimental pattern plot for one sample as PNG.

    Returns the absolute path to the saved PNG, or None if saving failed.
    """
    try:
        # Use absolute path to avoid WinError 2 when opening later
        plots_folder = os.path.abspath(plots_folder)
        os.makedirs(plots_folder, exist_ok=True)

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in sample_name
        )
        plot_path = os.path.join(plots_folder, f"{safe_name}.png")

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(tth_fit, counts_fit, lw=1.0, color="black",
                label="Experimental")
        ax.plot(tth_fit, fitted, lw=1.0, color="red",
                linestyle="--", label="Fitted")

        for j, pid in enumerate(active_ids):
            ax.fill_between(
                tth_fit, 0, ref_fit[:, j] * coeffs[j],
                alpha=0.35,
                label=f"{id_to_name.get(pid, pid)} "
                      f"({concentrations[j]:.1f} wt-%)"
            )

        ax.set_xlabel("2theta (deg)", fontsize=11)
        ax.set_ylabel("Counts", fontsize=11)
        ax.set_title(
            f"{sample_name}  -  Rwp = {rwp_val:.4f}  |  R = {r_val:.4f}",
            fontsize=11
        )
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        print(f"[PLOT]  Saved -> {plot_path}")
        return plot_path

    except Exception as e:
        import traceback
        print(f"[WARNING] Could not save plot for {sample_name}: {e}")
        traceback.print_exc()
        return None


def open_plots(plot_paths):
    """
    Open a list of PNG files using the OS default image viewer.
    All paths are resolved to absolute before opening.
    """
    if not plot_paths:
        return

    print(f"\n[PLOTS] Opening {len(plot_paths)} plot(s) ...")
    for p in plot_paths:
        p = os.path.abspath(p)
        if not os.path.isfile(p):
            print(f"[WARNING] Plot file not found: {p}")
            continue
        try:
            if sys.platform == "win32":
                os.startfile(p)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as e:
            print(f"[WARNING] Could not open {p}: {e}")
