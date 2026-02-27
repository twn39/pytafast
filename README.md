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
- 🔒 **Memory Safe** — Zero-copy data access with proper ownership management
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

## Supported Indicators

### Overlap Studies
`SMA`, `EMA`, `BBANDS`, `DEMA`, `KAMA`, `MA`, `T3`, `TEMA`, `TRIMA`, `WMA`, `MIDPOINT`, `SAR`

### Momentum Indicators
`RSI`, `MACD`, `MACDEXT`, `MACDFIX`, `ADX`, `ADXR`, `CCI`, `ROC`, `ROCP`, `ROCR`, `ROCR100`, `STOCH`, `STOCHF`, `STOCHRSI`, `MOM`, `WILLR`, `MFI`, `CMO`, `DX`, `MINUS_DI`, `MINUS_DM`, `PLUS_DI`, `PLUS_DM`, `APO`, `AROON`, `AROONOSC`, `PPO`, `TRIX`, `ULTOSC`, `BOP`

### Volatility
`ATR`, `NATR`, `TRANGE`

### Volume
`OBV`, `AD`, `ADOSC`

### Price Transform
`AVGPRICE`, `MEDPRICE`, `TYPPRICE`, `WCLPRICE`, `MIDPRICE`

### Statistics
`STDDEV`, `BETA`, `CORREL`, `LINEARREG`, `LINEARREG_ANGLE`, `LINEARREG_INTERCEPT`, `LINEARREG_SLOPE`, `TSF`, `VAR`, `AVGDEV`, `MIN`, `MAX`, `SUM`, `MINMAX`, `MINMAXINDEX`

### Math Operators
`ADD`, `SUB`, `MULT`, `DIV`

### Math Transforms
`ACOS`, `ASIN`, `ATAN`, `CEIL`, `COS`, `COSH`, `EXP`, `FLOOR`, `LN`, `LOG10`, `SIN`, `SINH`, `SQRT`, `TAN`, `TANH`

### Cycle Indicators
`HT_DCPERIOD`, `HT_DCPHASE`, `HT_PHASOR`, `HT_SINE`, `HT_TRENDLINE`, `HT_TRENDMODE`

### Candlestick Patterns (61 patterns)
`CDL2CROWS`, `CDL3BLACKCROWS`, `CDL3INSIDE`, `CDL3LINESTRIKE`, `CDL3OUTSIDE`, `CDL3STARSINSOUTH`, `CDL3WHITESOLDIERS`, `CDLABANDONEDBABY`, `CDLADVANCEBLOCK`, `CDLBELTHOLD`, `CDLBREAKAWAY`, `CDLCLOSINGMARUBOZU`, `CDLCONCEALBABYSWALL`, `CDLCOUNTERATTACK`, `CDLDARKCLOUDCOVER`, `CDLDOJI`, `CDLDOJISTAR`, `CDLDRAGONFLYDOJI`, `CDLENGULFING`, `CDLEVENINGDOJISTAR`, `CDLEVENINGSTAR`, `CDLGAPSIDESIDEWHITE`, `CDLGRAVESTONEDOJI`, `CDLHAMMER`, `CDLHANGINGMAN`, `CDLHARAMI`, `CDLHARAMICROSS`, `CDLHIGHWAVE`, `CDLHIKKAKE`, `CDLHIKKAKEMOD`, `CDLHOMINGPIGEON`, `CDLIDENTICAL3CROWS`, `CDLINNECK`, `CDLINVERTEDHAMMER`, `CDLKICKING`, `CDLKICKINGBYLENGTH`, `CDLLADDERBOTTOM`, `CDLLONGLEGGEDDOJI`, `CDLLONGLINE`, `CDLMARUBOZU`, `CDLMATCHINGLOW`, `CDLMATHOLD`, `CDLMORNINGDOJISTAR`, `CDLMORNINGSTAR`, `CDLONNECK`, `CDLPIERCING`, `CDLRICKSHAWMAN`, `CDLRISEFALL3METHODS`, `CDLSEPARATINGLINES`, `CDLSHOOTINGSTAR`, `CDLSHORTLINE`, `CDLSPINNINGTOP`, `CDLSTALLEDPATTERN`, `CDLSTICKSANDWICH`, `CDLTAKURI`, `CDLTASUKIGAP`, `CDLTHRUSTING`, `CDLTRISTAR`, `CDLUNIQUE3RIVER`, `CDLUPSIDEGAP2CROWS`, `CDLXSIDEGAP3METHODS`

## Performance

pytafast achieves **near-parity** with the official [ta-lib-python](https://github.com/TA-Lib/ta-lib-python) — both wrap the same underlying TA-Lib C library. Benchmarked across **62 indicators** on 100,000 data points:

| Category | Indicators | Avg Speedup |
|----------|-----------|-------------|
| Overlap Studies | SMA, EMA, BBANDS, DEMA, KAMA, TEMA, TRIMA, T3, SAR, MIDPOINT | **1.03x** |
| Momentum | RSI, MACD, ADX, CCI, STOCH, ROC, MOM, WILLR, PPO, BOP, MFI… | **1.03x** |
| Volatility | ATR, NATR, TRANGE | **0.99x** |
| Volume | OBV, AD, ADOSC | **1.00x** |
| Price Transform | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE | **0.98x** |
| Statistics | STDDEV, BETA, CORREL, LINEARREG, VAR, TSF | **1.07x** |
| Math Operators | ADD, SUB, MULT, DIV | **0.97x** |
| Math Transforms | SIN, COS, SQRT, LN, EXP | **1.00x** |
| Cycle | HT_DCPERIOD, HT_DCPHASE, HT_TRENDLINE, HT_TRENDMODE | **1.01x** |
| Candlestick | CDLENGULFING, CDLDOJI, CDL3WHITESOLDIERS, CDLHAMMER… | **0.97x** |

**Overall**: 39/62 indicators equal or faster · Average speedup **1.01x** · Notable: BETA 1.42x, BBANDS 1.27x, CCI 1.13x

> pytafast's key advantages are **pandas native support**, **async capabilities**, and **GIL release** for true parallelism — not raw single-call speed (both use the same C core).

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for full details.

## API Compatibility

pytafast follows the same function signatures as the official [TA-Lib Python wrapper](https://github.com/TA-Lib/ta-lib-python), making it a drop-in replacement in most cases.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project includes and statically links to the [TA-Lib](https://ta-lib.org/) C library, which is distributed under a BSD License.
Copyright (c) 1999-2025, Mario Fortier. See `third_party/ta-lib/LICENSE` for details.
