# pytafast

A high-performance Python wrapper for the `ta-lib` C library using `nanobind`.

## Project Overview

- **Core Technology:** Uses `nanobind` for efficient C++/Python bindings and `scikit-build-core` for the build system.
- **Architecture:** 
    - A C++ extension (`pytafast_ext`) wraps the raw `ta-lib` functions, organized into category-specific source files.
    - A Python layer (`pytafast`) provides a high-level API that supports both `numpy` arrays and `pandas` Series.
    - An async layer (`pytafast.aio`) provides `asyncio`-compatible wrappers using `asyncio.to_thread`.
- **Key Features:**
    - Preservation of `pandas.Series` metadata (index, name).
    - Automatic `TA_Initialize()` and `TA_Shutdown()` management.
    - Performance-oriented, utilizing `nb::ndarray` for zero-copy data access where possible.
    - GIL release in all C++ functions for true parallelism.

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

Tests are managed with `pytest`. The test suite compares all function outputs against the official `TA-Lib` Python package.

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v
```

## Development Conventions

### Project Structure

```
src/
├── common.h               # Shared types (DoubleArrayIN/OUT), constants, helpers (check_ta_retcode, alloc_output)
├── pytafast_ext.cpp        # nanobind module definition, forward declarations, and bindings
├── overlap.cpp             # Overlap Studies: SMA, EMA, BBANDS, DEMA, KAMA, MA, T3, TEMA, TRIMA, WMA, MIDPOINT, SAR
├── momentum.cpp            # Momentum: RSI, MACD, ADX, CCI, ROC, STOCH, MOM, WILLR, NATR, MFI, CMO, DX, etc.
├── volatility.cpp          # Volatility: ATR, NATR, TRANGE
├── price_transform.cpp     # Price Transform: AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, MIDPRICE
├── volume.cpp              # Volume: OBV, AD, ADOSC
├── statistic.cpp           # Statistics: BETA, CORREL, LINEARREG*, TSF, VAR, AVGDEV, MIN, MAX, SUM, MINMAX, MINMAXINDEX
├── math_operator.cpp       # Math Operators: ADD, SUB, MULT, DIV
├── math_transform.cpp      # Math Transforms: ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH
├── cycle.cpp               # Cycle: HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE
├── candlestick.cpp         # Candlestick Patterns: 61 CDL* functions (macros for standard + penetration variants)
└── pytafast/
    ├── __init__.py          # Public Python API with pandas/numpy support
    └── aio.py               # Async wrappers using asyncio.to_thread
```

- `third_party/`: Contains git submodules for `ta-lib` and `nanobind`.
- `tests/`: Contains `pytest` test cases.

### Coding Standards

- **C++:**
    - Use `nanobind` (`nb::ndarray`) for array passing.
    - Use `check_ta_retcode` to handle `TA_RetCode` and raise appropriate Python exceptions.
    - Ensure memory safety by using `nb::capsule` for data ownership when returning arrays allocated in C++.
    - Use `nb::gil_scoped_release` in compute sections for parallelism.
    - Use macros (`CDL_FUNC`, `MATH_TRANSFORM_FUNC`) to reduce boilerplate for repetitive function patterns.
- **Python:**
    - Support both `numpy.ndarray` and `pandas.Series` in public functions.
    - Achieve true zero-copy data passing by ensuring array continuity (`np.ascontiguousarray`) and explicitly typing C++ arguments as `const double` references.
    - Use factory functions (`_make_cdl_standard`, `_make_math_transform`) for generating repetitive wrappers.
- **Testing:**
    - Always compare results against the official `TA-Lib` Python package for verification.
    - Use `pytest.mark.parametrize` for testing groups of similar functions.
