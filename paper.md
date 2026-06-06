---
title: 'Pypowdr: A Python package for automated quantitative mineralogy using full pattern summation of X-ray powder diffraction data'
tags:
  - Python
  - X-ray powder diffraction
  - quantitative mineralogy
  - full pattern summation
  - geochemistry
authors:
  - name: Yi Xiang
    corresponding: true
    affiliation: 1
  - name: Jonathan Smith
    affiliation: 1
  - name: Juan Pablo Gevaudan
    corresponding: true
    affiliation: 1
affiliations:
  - name: The Pennsylvania State University, United States
    index: 1
date: 6 June 2026
bibliography: paper.bib
---

# Summary

`Pypowdr` is a Python package for automated quantitative mineral phase analysis from X-ray powder diffraction (XRPD) data using full pattern summation (FPS). The package reimplements the automated FPS algorithm (`afps`) described by @Butler:2021 — originally available only in the R package `powdR` — in Python using NumPy, SciPy, pandas, and matplotlib. `Pypowdr` provides a modular, scriptable analysis pipeline and a cross-platform graphical user interface (GUI) built with PyQt6, enabling researchers to load reference pattern libraries, fit sample diffractograms against known mineral phases, compute weight-percent concentrations via reference intensity ratios (RIRs) and an internal corundum standard, and export results as CSV files. The package has been validated against both the published RockJock synthetic mixture dataset [@Eberl:2003] and laboratory-prepared binary mineral mixtures, achieving quantitative agreement with the original R implementation.

# Statement of need

Quantitative mineral phase analysis by XRPD is fundamental to geoscience, environmental science, soil science, and materials characterization. Full pattern summation methods, which fit a measured diffractogram as a weighted linear combination of pure-phase reference patterns, offer a practical alternative to full Rietveld refinement when the goal is phase quantification rather than crystal structure determination [@Chipera:2002]. The `powdR` R package [@Butler:2021] implemented an automated FPS workflow that iteratively removes phases below a detection limit and re-optimizes until convergence, making FPS accessible to non-specialists.

However, the Python scientific ecosystem — anchored by NumPy, SciPy, pandas, and scikit-learn — is now the dominant environment for data-intensive geoscience workflows. Researchers who process XRPD data alongside spectroscopic, geochemical, or remote-sensing datasets increasingly work entirely in Python, and switching to R for a single analysis step introduces friction, complicates reproducibility, and prevents seamless integration with downstream machine-learning or statistical pipelines. No Python package currently provides an equivalent automated FPS workflow.

`Pypowdr` addresses this gap by providing a native Python implementation of the `afps` algorithm with identical default settings and validation benchmarks, a scriptable batch-processing interface for high-throughput laboratories, and a GUI that enables users without programming experience to perform quantitative XRPD analysis on Windows and macOS.

# State of the field

Several software tools address quantitative XRPD phase analysis. `RockJock` [@Eberl:2003] pioneered the FPS approach using an Excel-based interface and a large reference library, but it is no longer actively maintained and is limited to Windows. `FULLPAT` [@Chipera:2002] similarly performs full-pattern fitting but is distributed as a standalone program with limited scriptability. Full Rietveld refinement codes such as GSAS-II and TOPAS offer rigorous crystal-structure-based quantification with corrections for preferred orientation and microstructure, but they require crystal structure models for every phase and expert user input, making them impractical for routine high-throughput analysis of complex natural samples.

The R package `powdR` [@Butler:2021] filled an important niche by combining a curated reference library with an automated iterative algorithm, sensible defaults, and open-source availability. However, `powdR` is implemented exclusively in R, and no Python equivalent exists.

`Pypowdr` was built as a new implementation rather than a wrapper around `powdR` for three reasons. First, a native Python implementation avoids the `rpy2` bridge, which introduces installation complexity, version-coupling issues, and performance overhead. Second, a pure-Python codebase allows direct integration with the broader scientific Python stack — users can call `Pypowdr` functions inside Jupyter notebooks, embed them in automated laboratory pipelines, or extend them with custom preprocessing without leaving Python. Third, reimplementation allowed the addition of a PyQt6 GUI and platform-specific packaging (macOS `.app` bundle) that were not part of the original R workflow.

# Software design

`Pypowdr` is organized into six computational modules and one GUI module, mirroring the logical stages of the `afps` algorithm described by @Butler:2021. The `config` module centralizes all user-editable settings — file paths, phase definitions, RIR values, internal standard identity, solver choice, objective function, alignment range, and limit-of-detection (LOD) threshold — in a single location. The `preprocessing` module performs cubic spline interpolation of all patterns onto a common $2\theta$ grid and optionally aligns the sample pattern to the internal standard via cross-correlation within a user-specified maximum shift. The `fitting` module implements three objective functions (Delta, $R$, and $R_{wp}$) and supports BFGS, Nelder–Mead, and conjugate-gradient optimization with non-negative least-squares (NNLS) initialization. The `afps` module orchestrates the full iterative workflow: harmonize, align, NNLS seed, optimize, remove negative coefficients, compute concentrations and LODs via RIR-based equations, and iteratively remove below-LOD phases until convergence. The `plotting` module generates publication-quality fitted-versus-experimental pattern overlays with shaded phase contributions and residual plots. The `run_analysis` module provides a batch runner that discovers sample CSV files in a directory, calls `afps` for each, and assembles a combined results table.

A key design decision was the separation of the computational core from the interface layer. The GUI (`app_qt.py`) wraps the same validated pipeline in a PyQt6 front-end with file-selection dialogs, algorithm-setting controls, tabbed per-sample results with interactive plots, and CSV export — without duplicating or modifying any quantification logic. This separation ensures that script-based and GUI-based workflows produce identical results, and that the computational modules can be imported independently for programmatic use.

Default algorithm parameters — BFGS solver, $R_{wp}$ objective, $\pm 0.2°$ alignment, and 0.5 wt-% LOD — follow the recommendations of @Butler:2021 to enable direct comparison with the R implementation.

# Research impact statement

`Pypowdr` has been validated through two complementary approaches. First, the package was benchmarked against the RockJock synthetic mixture dataset [@Eberl:2003], which consists of eight XRPD measurements of seven-phase mineral mixtures at known concentrations. The Python implementation achieved an overall mean absolute error (MAE) of 0.98 wt-%, consistent with the 0.9 wt-% reported for the R `powdR` package, and $R_{wp}$ values across all eight mixtures (0.117–0.143) matched the published values within 0.003. Second, laboratory binary mixtures of calcite–corundum and hydromagnesite–corundum were prepared following an in-house wet-grinding protocol and measured on a Malvern Panalytical Empyrean diffractometer. The Python implementation successfully quantified both mixtures, with $R_{wp}$ values of 0.103 and 0.083 respectively.

The package is currently used in the authors' laboratory at The Pennsylvania State University for routine quantitative XRPD analysis of geological and environmental samples, and the validated workflow has been incorporated into a laboratory standard operating procedure for sample preparation and data analysis. The addition of a GUI has made the FPS method accessible to undergraduate and graduate students without Python programming experience.

# AI usage disclosure

Generative AI tools (Anthropic Claude, various versions) were used during the development of this project to assist with Python code generation, refactoring, documentation drafting, and preparation of this manuscript. All AI-generated code was reviewed, tested, and validated by the human authors. Core design decisions — including the modular architecture, choice of algorithm defaults, validation strategy, and GUI design — were made by the authors. All scientific results were independently verified against published benchmarks and laboratory measurements.

# Acknowledgements

We acknowledge the developers of the R `powdR` package, Benjamin M. Butler and Stephen Hillier, whose open-source implementation and published validation datasets made this work possible. We also acknowledge Dennis D. Eberl for the RockJock reference library and synthetic mixture data. This work is funded by the DOE (Department of Energy) Office of Nuclear Energy's Nuclear Energy University Programs Award number: 24-32112. Any opinions, findings, and conclusions or recommendations expressed in this dissertation are those of the author and do not necessarily reflect the views of the DOE.

# References
