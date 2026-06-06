# Contributing to Pypowdr

Thank you for your interest in contributing to Pypowdr! This document explains how to report bugs, suggest improvements, and submit changes.

## Reporting bugs

If you find a bug, please open an issue on [GitHub Issues](https://github.com/yiiiiiiiiix/Pypowdr/issues) with the following information:

- A clear description of the problem
- Steps to reproduce the issue
- Your Python version and operating system
- The versions of numpy, scipy, pandas, and matplotlib you are using
- Any error messages or tracebacks

## Suggesting features

Feature requests are welcome. Please open an issue with:

- A description of the feature and why it would be useful
- Any relevant references (papers, algorithms, existing implementations)

## Setting up a development environment

1. Clone the repository:
   ```
   git clone https://github.com/yiiiiiiiiix/Pypowdr.git
   cd Pypowdr
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\activate         # Windows
   ```

3. Install the package in development mode with test dependencies:
   ```
   pip install -e ".[dev]"
   ```

## Running tests

Run the full test suite with:

```
pytest tests/ -v
```

To run a specific test class:

```
pytest tests/test_afps.py::TestRockJockValidation -v
```

Please ensure all tests pass before submitting a pull request.

## Submitting changes

1. Fork the repository on GitHub.
2. Create a new branch for your changes (`git checkout -b my-feature`).
3. Make your changes and add or update tests as needed.
4. Run the test suite to confirm nothing is broken.
5. Commit your changes with a clear commit message.
6. Push to your fork and open a pull request against `main`.

## Coding conventions

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use descriptive variable names, especially for scientific quantities (e.g., `tth` for 2-theta, `rwp` for weighted profile R-factor).
- Add docstrings to all public functions and classes.
- Keep the computational modules (`afps.py`, `fitting.py`, `preprocessing.py`) independent of the GUI code (`app_qt.py`).

## Questions

If you have questions about the code or the FPS algorithm, feel free to open an issue or contact the maintainers at ybx5215@psu.edu.
