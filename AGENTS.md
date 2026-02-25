# pytafast

A high-performance Python wrapper for the `ta-lib` C library using `nanobind`.

## Project Overview

- **Core Technology:** Uses `nanobind` for efficient C++/Python bindings and `scikit-build-core` for the build system.
- **Architecture:**
    - A C++ extension (`pytafast_ext`) wraps the raw `ta-lib` functions, organized into category-specific source files.
    - A Python layer (`pytafast`) provides a high-level API that supports both `numpy` arrays and `pandas` Series, using factory functions to minimize boilerplate.
    - An async namespace (`pytafast.aio`) is built inline as a virtual submodule using `types.ModuleType`, providing `asyncio`-compatible wrappers via `asyncio.to_thread`. No separate file — all generated from sync functions at import time.
- **Key Features:**
    - Preservation of `pandas.Series` metadata (index).
    - Automatic `TA_Initialize()` and `TA_Shutdown()` management via `atexit`.
    - Performance-oriented, utilizing `nb::ndarray` for zero-copy data access where possible.
    - GIL release in all C++ functions for true parallelism.
    - Input array length validation in C++ to prevent segfaults.
    - `_ensure_array` fast path skips `np.ascontiguousarray` when input is already `float64` C-contiguous.

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

# Run benchmarks
uv run pytest tests/test_benchmark.py -v
```

## Development Conventions

### Project Structure

```
src/
├── common.h               # Shared types (DoubleArrayIN/OUT), constants, helpers (check_ta_retcode, alloc_output)
├── pytafast_ext.cpp        # nanobind module definition, forward declarations, and bindings
├── overlap.cpp             # Overlap Studies: SMA, EMA, BBANDS, DEMA, KAMA, MA, T3, TEMA, TRIMA, WMA, MIDPOINT, SAR
├── momentum.cpp            # Momentum: RSI, MACD, ADX, CCI, ROC, STOCH, MOM, WILLR, MFI, CMO, DX, APO, PPO, etc.
├── volatility.cpp          # Volatility: ATR, NATR, TRANGE, STDDEV
├── price_transform.cpp     # Price Transform: AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, MIDPRICE
├── volume.cpp              # Volume: OBV, AD, ADOSC
├── statistic.cpp           # Statistics: BETA, CORREL, LINEARREG*, TSF, VAR, AVGDEV, MIN, MAX, SUM, MINMAX, MINMAXINDEX
├── math_operator.cpp       # Math Operators: ADD, SUB, MULT, DIV
├── math_transform.cpp      # Math Transforms: ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH
├── cycle.cpp               # Cycle: HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE
├── candlestick.cpp         # Candlestick Patterns: 61 CDL* functions (macros for standard + penetration variants)
└── pytafast/
    └── __init__.py          # Public Python API with pandas/numpy support + inline aio async namespace
```

- `third_party/`: Contains git submodules for `ta-lib` and `nanobind`.
- `tests/`: Contains `pytest` test cases and benchmarks.

### Coding Standards

- **C++:**
    - Use `nanobind` (`nb::ndarray`) for array passing.
    - Use `check_ta_retcode` to handle `TA_RetCode` and raise appropriate Python exceptions.
    - Ensure memory safety by using `nb::capsule` for data ownership when returning arrays allocated in C++.
    - Use `nb::gil_scoped_release` in compute sections for parallelism.
    - Use macros (`CDL_FUNC`, `CDL_FUNC_PEN`, `MATH_TRANSFORM_FUNC`) to reduce boilerplate for repetitive function patterns.
    - Validate input array lengths match for multi-input functions (throw `std::runtime_error` on mismatch).
- **Python:**
    - Module-level pandas detection (`_HAS_PANDAS`), no try/except per function call.
    - `_ensure_array` for efficient input conversion with fast path for already-compatible arrays.
    - Factory functions for generating repetitive wrappers:
        - `_make_single(name, default_timeperiod)` — single-input + timeperiod
        - `_make_single_no_params(name)` — single-input, no params
        - `_make_hlc(name, default_timeperiod)` — HLC + timeperiod
        - `_make_hl(name, default_timeperiod)` — HL + timeperiod
        - `_make_dual(name, default_timeperiod)` — dual-input + timeperiod
        - `_make_dual_no_params(name)` — dual-input, no params
        - `_make_cdl_standard(name)` — OHLC candlestick pattern
        - `_make_cdl_penetration(name, default_pen)` — OHLC + penetration param
        - `_make_math_transform(name)` — single-input math function
    - Async wrappers auto-generated via `_make_async(sync_fn)` and attached to a `types.ModuleType("pytafast.aio")` object, registered in `sys.modules`. No separate `aio.py` file.
- **Testing:**
    - Always compare results against the official `TA-Lib` Python package for verification.
    - Use `pytest.mark.parametrize` for testing groups of similar functions.
    - Benchmark tests in `tests/test_benchmark.py` using `pytest-benchmark`.
