<div align="center">

# pytafast

[![PyPI](https://img.shields.io/pypi/v/pytafast?color=blue)](https://pypi.org/project/pytafast/)
[![Python](https://img.shields.io/pypi/pyversions/pytafast)](https://pypi.org/project/pytafast/)
[![Codecov](https://img.shields.io/codecov/c/github/twn39/pytafast)](https://codecov.io/gh/twn39/pytafast)
[![License](https://img.shields.io/pypi/l/pytafast)](https://github.com/twn39/pytafast/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/pytafast)](https://pypi.org/project/pytafast/)
[![CI](https://img.shields.io/github/actions/workflow/status/twn39/pytafast/build.yml?label=CI)](https://github.com/twn39/pytafast/actions)
[![GitHub Stars](https://img.shields.io/github/stars/twn39/pytafast?style=flat)](https://github.com/twn39/pytafast)

English | [中文](README_CN.md)

</div>

A high-performance Python wrapper for [TA-Lib](https://ta-lib.org/) and [R TTR](https://github.com/joshuaulrich/TTR) built with [nanobind](https://github.com/wjakob/nanobind). Provides **170+ technical analysis functions** with interactive plotting, pandas support, and async capabilities.

## Features

- 🚀 **High Performance** — C++ bindings via nanobind with GIL release for true parallelism.
- 📈 **R TTR Consistency** — Migrated 14+ R-native indicators (ALMA, ZigZag, GMMA, KST, etc.) with 100% numerical alignment.
- 📊 **Interactive Plotting** — Built-in `quantmod`-style visualization engine powered by Plotly.
- 🐼 **Pandas Native** — Seamless support for both `numpy.ndarray` and `pandas.Series` (preserves index).
- ⚡ **Async Support** — All functions available as async via `pytafast.aio`.
- 🔒 **GSL Powered Safety** — Uses Microsoft GSL (`gsl::span`) to prevent buffer overflows in C++.
- 📦 **Drop-in Replacement** — Same API as [ta-lib-python](https://github.com/TA-Lib/ta-lib-python).

## Installation

```bash
pip install pytafast
```

### Optional Dependencies
```bash
# For static image export (PNG/PDF)
pip install kaleido
```

## Quick Start

### Interactive Plotting (quantmod style)

`pytafast` provides a powerful chaining API for professional financial charts:

```python
import pandas as pd
import pytafast

df = pd.read_csv("data.csv")

# Create a professional interactive chart in one chain
chart = (pytafast.Chart(df)
         .add_candlestick(name="NASDAQ 100")
         .add_sma(n=20, color='orange')
         .add_bbands(n=20)
         .add_zigzag(change=2.0)
         .add_patterns()  # Automatically label 60+ candlestick patterns
         .add_volume()
         .add_macd()
         .add_rsi())

chart.show()  # Opens interactive Plotly chart
chart.save_image("analysis.png")  # Saves as high-res static image
```

### R-native Indicator Computation

```python
import pytafast

# Arnaud Legoux Moving Average (ALMA) - superior smoothness
alma = pytafast.ALMA(close, timeperiod=9, offset=0.85, sigma=6.0)

# Zero Lag Exponential Moving Average (ZLEMA)
zlema = pytafast.ZLEMA(close, timeperiod=30)

# Chaikin Money Flow (CMF)
cmf = pytafast.CMF(high, low, close, volume, timeperiod=20)

# Know Sure Thing (KST)
kst, signal = pytafast.KST(close)
```

## Advanced Calculation Examples

### 1. Multi-MA Strategy Setup
```python
import pytafast
# Combine traditional MAs with high-performance R-style smoothers
df["sma"] = pytafast.SMA(df["close"], 20)
df["alma"] = pytafast.ALMA(df["close"], timeperiod=9, offset=0.85, sigma=6)
df["zlema"] = pytafast.ZLEMA(df["close"], 30)
```

### 2. Volatility & Squeeze Detection
```python
upper_bb, mid_bb, lower_bb = pytafast.BBANDS(df["close"], 20)
upper_kc, mid_kc, lower_kc = pytafast.keltnerChannels(df["high"], df["low"], df["close"], 20)
# Detect if Bollinger Bands are inside Keltner Channels (Squeeze)
df["squeeze"] = (upper_bb < upper_kc) & (lower_bb > lower_kc)
```

## Indicators Portfolio

| Category | Indicators (Partial List) |
|:---|:---|
| **R-Native (New)** | **ALMA, ZLEMA, EVWMA, ZIGZAG, HMA, Donchian, Keltner, CMF, KST, SMI, VHF, SNR** |
| **Overlap** | SMA, EMA, WMA, DEMA, TEMA, KAMA, BBANDS, SAR, MIDPOINT... |
| **Momentum** | RSI, MACD, STOCH, ADX, CCI, ROC, MOM, WILLR, MFI... |
| **Volatility** | ATR, NATR, TRANGE, STDDEV |
| **Pattern** | 61 patterns: CDLHAMMER, CDLENGULFING, CDLDOJI, CDLMORNINGSTAR... |

## Performance

pytafast achieves **superior throughput** via C++ GIL release. Scalability is near-linear with CPU cores.

| Multi-threaded (4 Threads) | Official Wrapper | pytafast | **Speedup** |
|:---|:---|:---|:---|
| **SMA** Concurrency | 20.1 ms | 6.3 ms | **3.15x** |
| **MACD** Concurrency | 75.0 ms | 20.7 ms | **3.62x** |

## Cross-Verification

We maintain a rigorous cross-verification suite against R `TTR`. You can run the numerical alignment report locally:

```bash
# Compare 150+ indicators against R TTR results
./scripts/run_comparison.sh data/berkshire_1y.csv
```
Detailed results are documented in [docs/comparison_report.md](docs/comparison_report.md).

## License

MIT License. Includes statically linked [TA-Lib](https://ta-lib.org/) (BSD).
Copyright (c) 1999-2026, Curry Tang & Mario Fortier.
