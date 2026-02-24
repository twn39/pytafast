# pytafast

[![PyPI](https://img.shields.io/pypi/v/pytafast)](https://pypi.org/project/pytafast/)
[![Python](https://img.shields.io/pypi/pyversions/pytafast)](https://pypi.org/project/pytafast/)

A high-performance Python wrapper for [TA-Lib](https://ta-lib.org/) built with [nanobind](https://github.com/wjakob/nanobind). Provides **150+ technical analysis functions** with pandas/numpy support and async capabilities.

## Features

- 🚀 **High Performance** — C++ bindings via nanobind with GIL release for true parallelism
- 📊 **Full TA-Lib Coverage** — 150+ indicators including overlaps, momentum, volatility, volume, statistics, cycle indicators, and 61 candlestick patterns
- 🐼 **Pandas Native** — Seamless support for both `numpy.ndarray` and `pandas.Series` (preserves index and name)
- ⚡ **Async Support** — All functions available as async via `pytafast.aio`
- 🔒 **Memory Safe** — Zero-copy data access with proper ownership management

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

```python
import numpy as np
import pytafast

# Using numpy arrays
close = np.random.random(100) * 100

sma = pytafast.SMA(close, timeperiod=20)
rsi = pytafast.RSI(close, timeperiod=14)
upper, middle, lower = pytafast.BBANDS(close, timeperiod=20)
macd, signal, hist = pytafast.MACD(close)
```

```python
# Using pandas Series
import pandas as pd

close = pd.Series(np.random.random(100) * 100, name="close")
sma = pytafast.SMA(close, timeperiod=20)  # Returns pd.Series with preserved index
```

```python
# Async support
import pytafast.aio as aio

async def compute():
    sma = await aio.SMA(close, timeperiod=20)
    rsi = await aio.RSI(close, timeperiod=14)
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

## API Compatibility

pytafast follows the same function signatures as the official [TA-Lib Python wrapper](https://github.com/TA-Lib/ta-lib-python), making it a drop-in replacement in most cases.

## License

MIT
