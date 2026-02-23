# pytafast

A high-performance Python wrapper for the `ta-lib` C library using `nanobind`.

## Project Overview

- **Core Technology:** Uses `nanobind` for efficient C++/Python bindings and `scikit-build-core` for the build system.
- **Architecture:** 
    - A C++ extension (`pytafast_ext`) wraps the raw `ta-lib` functions.
    - A Python layer (`pytafast`) provides a high-level API that supports both `numpy` arrays and `pandas` Series.
- **Key Features:**
    - Preservation of `pandas.Series` metadata (index, name).
    - Automatic `TA_Initialize()` and `TA_Shutdown()` management.
    - Performance-oriented, utilizing `nb::ndarray` for zero-copy data access where possible.

## Building and Running

### Prerequisites

- C++ compiler (supporting C++17 or later)
- CMake (3.15+)
- Python (3.11+)

### Installation

For development, it is recommended to use `uv` or `pip` in editable mode:

```bash
# Using uv (recommended)
uv pip install -v -e .

# Using pip
pip install -v -e .
```

The `-v` flag is helpful to see the CMake build output.

### Testing

Tests are managed with `pytest`.

```bash
# Run all tests
pytest

# Run tests with benchmarking
pytest tests/test_benchmark.py
```

## Development Conventions

### Project Structure

- `src/pytafast_ext.cpp`: The main C++ binding file. All `ta-lib` function wrappings should be added here.
- `src/pytafast/__init__.py`: The public Python API. New functions added to the C++ extension should be exposed here with appropriate documentation and type handling.
- `third_party/`: Contains git submodules for `ta-lib` and `nanobind`.
- `tests/`: Contains `pytest` test cases.

### Coding Standards

- **C++:**
    - Use `nanobind` (`nb::ndarray`) for array passing.
    - Use `check_ta_retcode` to handle `TA_RetCode` and raise appropriate Python exceptions.
    - Ensure memory safety by using `nb::capsule` for data ownership when returning arrays allocated in C++.
- **Python:**
    - Support both `numpy.ndarray` and `pandas.Series` in public functions.
    - Achieve true zero-copy data passing by ensuring array continuity (`np.ascontiguousarray`) and explicitly typing C++ arguments as `const double` references (`nb::ndarray<nb::numpy, const double, nb::c_contig>`) which safely accepts Pandas' read-only memory views without enforcing deep copies.
- **Testing:**
    - Always compare results against the official `TA-Lib` Python package for verification.
    - Include benchmarks for new functions to ensure performance remains competitive.
