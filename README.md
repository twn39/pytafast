<div align="center">

# pytafast

[![PyPI](https://img.shields.io/pypi/v/pytafast?color=blue)](https://pypi.org/project/pytafast/)
[![Python](https://img.shields.io/pypi/pyversions/pytafast)](https://pypi.org/project/pytafast/)
[![Codecov](https://img.shields.io/codecov/c/github/twn39/pytafast)](https://codecov.io/gh/twn39/pytafast)
[![License](https://img.shields.io/pypi/l/pytafast)](https://github.com/twn39/pytafast/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/pytafast)](https://pypi.org/project/pytafast/)
[![CI](https://img.shields.io/github/actions/workflow/status/twn39/pytafast/build.yml?label=CI)](https://github.com/twn39/pytafast/actions)
[![GitHub Stars](https://img.shields.io/github/stars/twn39/pytafast?style=flat)](https://github.com/twn39/pytafast)

</div>

A high-performance Python wrapper for [TA-Lib](https://ta-lib.org/) built with [nanobind](https://github.com/wjakob/nanobind). Provides **150+ technical analysis functions** with pandas/numpy support and async capabilities.

## Features

- 🚀 **High Performance** — C++ bindings via nanobind with GIL release for true parallelism
- 📊 **Full TA-Lib Coverage** — 150+ indicators including overlaps, momentum, volatility, volume, statistics, cycle indicators, and 61 candlestick patterns
- 🐼 **Pandas Native** — Seamless support for both `numpy.ndarray` and `pandas.Series` (preserves index)
- ⚡ **Async Support** — All functions available as async via `pytafast.aio`
- 🔒 **GSL Powered Safety** — Uses Microsoft GSL (`gsl::not_null`, `gsl::span`) to prevent buffer overflows and null pointer dereferences in C++
- 📦 **Drop-in Replacement** — Same API as [ta-lib-python](https://github.com/TA-Lib/ta-lib-python), easy migration

## Installation

```bash
pip install pytafast
```

### Build from Source

```bash
git clone --recursive https://github.com/twn39/pytafast.git
cd pytafast
pip install -v -e .
```

## Quick Start

### Basic Usage with NumPy

```python
import numpy as np
import pytafast

# Generate sample price data
np.random.seed(42)
close = np.cumsum(np.random.randn(200)) + 100

# Moving Averages
sma = pytafast.SMA(close, timeperiod=20)
ema = pytafast.EMA(close, timeperiod=12)
wma = pytafast.WMA(close, timeperiod=10)
```

### Momentum Indicators

```python
# RSI — Relative Strength Index
rsi = pytafast.RSI(close, timeperiod=14)

# MACD — returns 3 arrays: macd line, signal line, histogram
macd, signal, hist = pytafast.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

# Stochastic Oscillator — requires high, low, close
high = close + np.abs(np.random.randn(200)) * 2
low = close - np.abs(np.random.randn(200)) * 2
slowk, slowd = pytafast.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)

# ADX — Average Directional Index
adx = pytafast.ADX(high, low, close, timeperiod=14)
```

### Bollinger Bands & Volatility

```python
# Bollinger Bands — returns upper, middle, lower bands
upper, middle, lower = pytafast.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)

# ATR — Average True Range
atr = pytafast.ATR(high, low, close, timeperiod=14)
```

### Volume Indicators

```python
volume = np.random.random(200) * 1_000_000

# On-Balance Volume
obv = pytafast.OBV(close, volume)

# Chaikin A/D Line
ad = pytafast.AD(high, low, close, volume)

# Money Flow Index
mfi = pytafast.MFI(high, low, close, volume, timeperiod=14)
```

### Statistics & Math

```python
# Linear Regression
linreg = pytafast.LINEARREG(close, timeperiod=14)
slope = pytafast.LINEARREG_SLOPE(close, timeperiod=14)

# Correlation between two series
beta = pytafast.BETA(close, ema, timeperiod=5)
correl = pytafast.CORREL(close, ema, timeperiod=30)

# Standard Deviation
stddev = pytafast.STDDEV(close, timeperiod=20, nbdev=1.0)
```

### Candlestick Pattern Recognition

```python
# Generate OHLC data
open_ = close + np.random.randn(200) * 0.5

# Detect patterns — returns integer array (100=bullish, -100=bearish, 0=none)
engulfing = pytafast.CDLENGULFING(open_, high, low, close)
doji = pytafast.CDLDOJI(open_, high, low, close)
hammer = pytafast.CDLHAMMER(open_, high, low, close)
morning_star = pytafast.CDLMORNINGSTAR(open_, high, low, close, penetration=0.3)

# Find bullish signals
bullish_idx = np.where(engulfing == 100)[0]
```

### Pandas Support

```python
import pandas as pd

# Create a DataFrame of stock prices
df = pd.DataFrame({
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
}, index=pd.date_range("2024-01-01", periods=200, freq="D"))

# All functions accept and return pd.Series, preserving the DatetimeIndex
df["sma_20"] = pytafast.SMA(df["close"], timeperiod=20)
df["rsi_14"] = pytafast.RSI(df["close"], timeperiod=14)
df["atr_14"] = pytafast.ATR(df["high"], df["low"], df["close"], timeperiod=14)

upper, middle, lower = pytafast.BBANDS(df["close"], timeperiod=20)
df["bb_upper"] = upper
df["bb_lower"] = lower
```

### Async Support

```python
import asyncio
import pytafast

async def compute_indicators(close, high, low, volume):
    """Compute multiple indicators concurrently."""
    sma, rsi, macd_result, atr = await asyncio.gather(
        pytafast.aio.SMA(close, timeperiod=20),
        pytafast.aio.RSI(close, timeperiod=14),
        pytafast.aio.MACD(close),
        pytafast.aio.ATR(high, low, close, timeperiod=14),
    )
    macd, signal, hist = macd_result
    return sma, rsi, macd, atr

# asyncio.run(compute_indicators(close, high, low, volume))
```

### Cycle Indicators

```python
# Hilbert Transform indicators for cycle analysis
ht_period = pytafast.HT_DCPERIOD(close)
ht_trendline = pytafast.HT_TRENDLINE(close)
sine, leadsine = pytafast.HT_SINE(close)
trend_mode = pytafast.HT_TRENDMODE(close)  # 1 = trend, 0 = cycle
```

## API Reference

All functions accept `numpy.ndarray` (float64) or `pandas.Series`. When a `pandas.Series` is provided, the function returns a `pandas.Series` with the same index.

### Overlap Studies

| Function | Signature | Description |
|:---|:---|:---|
| **SMA** | `SMA(real, timeperiod=30)` | Simple Moving Average |
| **EMA** | `EMA(real, timeperiod=30)` | Exponential Moving Average |
| **WMA** | `WMA(real, timeperiod=30)` | Weighted Moving Average |
| **DEMA** | `DEMA(real, timeperiod=30)` | Double Exponential Moving Average |
| **TEMA** | `TEMA(real, timeperiod=30)` | Triple Exponential Moving Average |
| **TRIMA** | `TRIMA(real, timeperiod=30)` | Triangular Moving Average |
| **KAMA** | `KAMA(real, timeperiod=30)` | Kaufman Adaptive Moving Average |
| **MAMA** | `MAMA(real, fastlimit=0.5, slowlimit=0.05)` | MESA Adaptive Moving Average. Returns `(mama, fama)` |
| **T3** | `T3(real, timeperiod=5, vfactor=0.7)` | Triple Exponential Moving Average (T3) |
| **BBANDS** | `BBANDS(real, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0)` | Bollinger Bands. Returns `(upper, middle, lower)` |
| **MA** | `MA(real, timeperiod=30, matype=0)` | Moving Average (generic) |
| **SAR** | `SAR(high, low, acceleration=0.02, maximum=0.2)` | Parabolic SAR |
| **MIDPOINT** | `MIDPOINT(real, timeperiod=14)` | Midpoint over period |
| **MIDPRICE** | `MIDPRICE(high, low, timeperiod=14)` | Midprice over period |

### Momentum Indicators

| Function | Signature | Description |
|:---|:---|:---|
| **RSI** | `RSI(real, timeperiod=14)` | Relative Strength Index |
| **MACD** | `MACD(real, fastperiod=12, slowperiod=26, signalperiod=9)` | Returns `(macd, signal, hist)` |
| **MACDEXT** | `MACDEXT(real, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)` | MACD with controllable MA types. Returns `(macd, signal, hist)` |
| **MACDFIX** | `MACDFIX(real, signalperiod=9)` | MACD Fix 12/26. Returns `(macd, signal, hist)` |
| **STOCH** | `STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)` | Returns `(slowk, slowd)` |
| **STOCHF** | `STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)` | Returns `(fastk, fastd)` |
| **STOCHRSI** | `STOCHRSI(real, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)` | Returns `(fastk, fastd)` |
| **ADX** | `ADX(high, low, close, timeperiod=14)` | Average Directional Index |
| **ADXR** | `ADXR(high, low, close, timeperiod=14)` | Average Directional Index Rating |
| **CCI** | `CCI(high, low, close, timeperiod=14)` | Commodity Channel Index |
| **MFI** | `MFI(high, low, close, volume, timeperiod=14)` | Money Flow Index |
| **ROC** | `ROC(real, timeperiod=10)` | Rate of change : ((price/prevPrice)-1)*100 |
| **ROCP** | `ROCP(real, timeperiod=10)` | Rate of change Percentage: (price-prevPrice)/prevPrice |
| **ROCR** | `ROCR(real, timeperiod=10)` | Rate of change ratio: (price/prevPrice) |
| **ROCR100** | `ROCR100(real, timeperiod=10)` | Rate of change ratio 100: (price/prevPrice)*100 |
| **MOM** | `MOM(real, timeperiod=10)` | Momentum |
| **WILLR** | `WILLR(high, low, close, timeperiod=14)` | Williams' %R |
| **CMO** | `CMO(real, timeperiod=14)` | Chande Momentum Oscillator |
| **DX** | `DX(high, low, close, timeperiod=14)` | Directional Movement Index |
| **MINUS_DI** | `MINUS_DI(high, low, close, timeperiod=14)` | Minus Directional Indicator |
| **MINUS_DM** | `MINUS_DM(high, low, timeperiod=14)` | Minus Directional Movement |
| **PLUS_DI** | `PLUS_DI(high, low, close, timeperiod=14)` | Plus Directional Indicator |
| **PLUS_DM** | `PLUS_DM(high, low, timeperiod=14)` | Plus Directional Movement |
| **APO** | `APO(real, fastperiod=12, slowperiod=26, matype=0)` | Absolute Price Oscillator |
| **PPO** | `PPO(real, fastperiod=12, slowperiod=26, matype=0)` | Percentage Price Oscillator |
| **AROON** | `AROON(high, low, timeperiod=14)` | Returns `(aroondown, aroonup)` |
| **AROONOSC** | `AROONOSC(high, low, timeperiod=14)` | Aroon Oscillator |
| **ULTOSC** | `ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)` | Ultimate Oscillator |
| **BOP** | `BOP(open, high, low, close)` | Balance Of Power |
| **TRIX** | `TRIX(real, timeperiod=30)` | 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA |

### Volatility Indicators

| Function | Signature | Description |
|:---|:---|:---|
| **ATR** | `ATR(high, low, close, timeperiod=14)` | Average True Range |
| **NATR** | `NATR(high, low, close, timeperiod=14)` | Normalized Average True Range |
| **TRANGE** | `TRANGE(high, low, close)` | True Range |

### Volume Indicators

| Function | Signature | Description |
|:---|:---|:---|
| **OBV** | `OBV(real, volume)` | On Balance Volume |
| **AD** | `AD(high, low, close, volume)` | Chaikin A/D Line |
| **ADOSC** | `ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)` | Chaikin A/D Oscillator |

### Price Transform

| Function | Signature | Description |
|:---|:---|:---|
| **AVGPRICE** | `AVGPRICE(open, high, low, close)` | Average Price |
| **MEDPRICE** | `MEDPRICE(high, low)` | Median Price |
| **TYPPRICE** | `TYPPRICE(high, low, close)` | Typical Price |
| **WCLPRICE** | `WCLPRICE(high, low, close)` | Weighted Close Price |

### Statistics

| Function | Signature | Description |
|:---|:---|:---|
| **STDDEV** | `STDDEV(real, timeperiod=5, nbdev=1.0)` | Standard Deviation |
| **VAR** | `VAR(real, timeperiod=5, nbdev=1.0)` | Variance |
| **BETA** | `BETA(real0, real1, timeperiod=5)` | Beta |
| **CORREL** | `CORREL(real0, real1, timeperiod=30)` | Pearson's Correlation Coefficient (r) |
| **LINEARREG** | `LINEARREG(real, timeperiod=14)` | Linear Regression |
| **LINEARREG_ANGLE** | `LINEARREG_ANGLE(real, timeperiod=14)` | Linear Regression Angle |
| **LINEARREG_INTERCEPT**| `LINEARREG_INTERCEPT(real, timeperiod=14)`| Linear Regression Intercept |
| **LINEARREG_SLOPE** | `LINEARREG_SLOPE(real, timeperiod=14)` | Linear Regression Slope |
| **TSF** | `TSF(real, timeperiod=14)` | Time Series Forecast |
| **AVGDEV** | `AVGDEV(real, timeperiod=14)` | Average Deviation |
| **MAX** | `MAX(real, timeperiod=30)` | Highest value over a specified period |
| **MIN** | `MIN(real, timeperiod=30)` | Lowest value over a specified period |
| **SUM** | `SUM(real, timeperiod=30)` | Summation |
| **MINMAX** | `MINMAX(real, timeperiod=30)` | Returns `(min, max)` |
| **MINMAXINDEX** | `MINMAXINDEX(real, timeperiod=30)` | Returns `(minidx, maxidx)` |

### Cycle Indicators

| Function | Signature | Description |
|:---|:---|:---|
| **HT_DCPERIOD** | `HT_DCPERIOD(real)` | Hilbert Transform - Dominant Cycle Period |
| **HT_DCPHASE** | `HT_DCPHASE(real)` | Hilbert Transform - Dominant Cycle Phase |
| **HT_PHASOR** | `HT_PHASOR(real)` | Hilbert Transform - Phasor Components. Returns `(inphase, quadrature)` |
| **HT_SINE** | `HT_SINE(real)` | Hilbert Transform - SineWave. Returns `(sine, leadsine)` |
| **HT_TRENDLINE** | `HT_TRENDLINE(real)` | Hilbert Transform - Instantaneous Trendline |
| **HT_TRENDMODE** | `HT_TRENDMODE(real)` | Hilbert Transform - Trend vs Cycle Mode |

### Math Operators

| Function | Signature | Description |
|:---|:---|:---|
| **ADD** | `ADD(real0, real1)` | Vector Arithmetic Addition |
| **SUB** | `SUB(real0, real1)` | Vector Arithmetic Subtraction |
| **MULT** | `MULT(real0, real1)` | Vector Arithmetic Multiplication |
| **DIV** | `DIV(real0, real1)` | Vector Arithmetic Division |

### Math Transforms

All take a single `real` array:
`ACOS`, `ASIN`, `ATAN`, `CEIL`, `COS`, `COSH`, `EXP`, `FLOOR`, `LN`, `LOG10`, `SIN`, `SINH`, `SQRT`, `TAN`, `TANH`

### Candlestick Patterns

All candlestick functions accept `(open, high, low, close)` and return an integer array (100, -100, or 0).

**Standard patterns:**
`CDL2CROWS`, `CDL3BLACKCROWS`, `CDL3INSIDE`, `CDL3LINESTRIKE`, `CDL3OUTSIDE`, `CDL3STARSINSOUTH`, `CDL3WHITESOLDIERS`, `CDLADVANCEBLOCK`, `CDLBELTHOLD`, `CDLBREAKAWAY`, `CDLCLOSINGMARUBOZU`, `CDLCONCEALBABYSWALL`, `CDLCOUNTERATTACK`, `CDLDOJI`, `CDLDOJISTAR`, `CDLDRAGONFLYDOJI`, `CDLENGULFING`, `CDLGAPSIDESIDEWHITE`, `CDLGRAVESTONEDOJI`, `CDLHAMMER`, `CDLHANGINGMAN`, `CDLHARAMI`, `CDLHARAMICROSS`, `CDLHIGHWAVE`, `CDLHIKKAKE`, `CDLHIKKAKEMOD`, `CDLHOMINGPIGEON`, `CDLIDENTICAL3CROWS`, `CDLINNECK`, `CDLINVERTEDHAMMER`, `CDLKICKING`, `CDLKICKINGBYLENGTH`, `CDLLADDERBOTTOM`, `CDLLONGLEGGEDDOJI`, `CDLLONGLINE`, `CDLMARUBOZU`, `CDLMATCHINGLOW`, `CDLONNECK`, `CDLPIERCING`, `CDLRICKSHAWMAN`, `CDLRISEFALL3METHODS`, `CDLSEPARATINGLINES`, `CDLSHOOTINGSTAR`, `CDLSHORTLINE`, `CDLSPINNINGTOP`, `CDLSTALLEDPATTERN`, `CDLSTICKSANDWICH`, `CDLTAKURI`, `CDLTASUKIGAP`, `CDLTHRUSTING`, `CDLTRISTAR`, `CDLUNIQUE3RIVER`, `CDLUPSIDEGAP2CROWS`, `CDLXSIDEGAP3METHODS`

**Patterns with optional `penetration` parameter:**
`CDLABANDONEDBABY(..., penetration=0.3)`, `CDLDARKCLOUDCOVER(..., penetration=0.5)`, `CDLEVENINGDOJISTAR(..., penetration=0.3)`, `CDLEVENINGSTAR(..., penetration=0.3)`, `CDLMATHOLD(..., penetration=0.5)`, `CDLMORNINGDOJISTAR(..., penetration=0.3)`, `CDLMORNINGSTAR(..., penetration=0.3)`

## Performance

pytafast achieves **superior performance** compared to the official [ta-lib-python](https://github.com/TA-Lib/ta-lib-python) in Pandas, multi-threaded, and complex calculation scenarios. Benchmarked on 100,000 data points:

### Multi-threading Throughput

Thanks to efficient GIL release in C++, `pytafast` scales near-linearly with CPU cores. Benchmark results in a 4-thread concurrent environment:

| Benchmark (4 Threads) | Official Wrapper | pytafast | **Throughput Increase** |
|:---|:---|:---|:---|
| **SMA** Concurrency | 20.1 ms | 6.3 ms | **3.15x** |
| **RSI** Concurrency | 56.5 ms | 16.3 ms | **3.46x** |
| **MACD** Concurrency | 75.0 ms | 20.7 ms | **3.62x** |

### Single-call Efficiency

| Category | Representative Indicators | Avg Speedup |
|----------|-----------|-------------|
| **Pandas Integration** | SMA, EMA, MOM, RSI (Series input) | **1.05x - 1.30x** |
| **Complex Indicators** | ADX, ADXR, ULTOSC, MAMA | **1.08x - 1.12x** |
| **Overlap Studies** | SMA, EMA, BBANDS, KAMA, T3 | **1.02x** |
| **Momentum** | RSI, MACD, STOCH, CCI, ROC | **1.03x** |
| **Statistics & Math** | LINEARREG, STDDEV, BETA, ADD | **1.04x** |
| **Overall Consistency** | All 150+ functions | **100% Match** |

**Key Highlights:**
*   **MOM (Pandas)**: pytafast is **30% faster** than official wrapper.
*   **ADX (NumPy)**: pytafast is **11% faster** due to lower binding overhead.
*   **Safety**: Integrated **Microsoft GSL** ensures robust memory management for large-scale financial data.

> **Recommendation**: Use `pytafast` for multi-threaded backtesting or high-concurrency web servers to unlock full CPU potential.

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for full details.

## API Compatibility

pytafast follows the same function signatures as the official [TA-Lib Python wrapper](https://github.com/TA-Lib/ta-lib-python), making it a drop-in replacement in most cases.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project includes and statically links to the [TA-Lib](https://ta-lib.org/) C library, which is distributed under a BSD License.
Copyright (c) 1999-2025, Mario Fortier. See `third_party/ta-lib/LICENSE` for details.
