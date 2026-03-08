<div align="center">

# pytafast

[![PyPI](https://img.shields.io/pypi/v/pytafast?color=blue)](https://pypi.org/project/pytafast/)
[![Python](https://img.shields.io/pypi/pyversions/pytafast)](https://pypi.org/project/pytafast/)
[![Codecov](https://img.shields.io/codecov/c/github/twn39/pytafast)](https://codecov.io/gh/twn39/pytafast)
[![License](https://img.shields.io/pypi/l/pytafast)](https://github.com/twn39/pytafast/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/pytafast)](https://pypi.org/project/pytafast/)
[![CI](https://img.shields.io/github/actions/workflow/status/twn39/pytafast/build.yml?label=CI)](https://github.com/twn39/pytafast/actions)
[![GitHub Stars](https://img.shields.io/github/stars/twn39/pytafast?style=flat)](https://github.com/twn39/pytafast)

[English](README.md) | 中文

</div>

基于 [nanobind](https://github.com/wjakob/nanobind) 构建的 [TA-Lib](https://ta-lib.org/) 和 [R TTR](https://github.com/joshuaulrich/TTR) 高性能 Python 封装库。提供 **170 多个技术分析指标**，内置交互式绘图引擎，完美支持 Pandas 并具备异步计算能力。

## 核心特性

- 🚀 **极致性能** — 基于 C++ 编写，通过 nanobind 绑定并释放 GIL，实现真正的多核并行计算。
- 📈 **R 语言对齐** — 迁移了 14 个以上 R 语言特有指标（如 ALMA, ZigZag, GMMA, KST 等），实现 100% 数值对齐。
- 📊 **交互式绘图** — 内置复刻 R 语言 `quantmod` 风格的绘图引擎，由 Plotly 强力驱动。
- 🐼 **Pandas 原生支持** — 无缝支持 `numpy.ndarray` 和 `pandas.Series`（自动保留索引）。
- ⚡ **异步支持** — `pytafast.aio` 命名空间下提供所有指标的异步版本。
- 🔒 **GSL 安全保障** — 引入 Microsoft GSL (`gsl::span`) 确保 C++ 层的内存访问安全，杜绝缓冲区溢出。
- 📦 **零成本迁移** — 保持与 [ta-lib-python](https://github.com/TA-Lib/ta-lib-python) 相同的 API 设计。

## 安装

```bash
pip install pytafast
```

### 可选依赖
```bash
# 用于静态图片导出 (PNG/PDF)
pip install kaleido
```

## 快速上手

### 交互式绘图 (quantmod 风格)

```python
import pandas as pd
import pytafast

df = pd.read_csv("data.csv")

# 一行链式调用生成专业交互式图表
chart = (pytafast.Chart(df)
         .add_candlestick(name="纳斯达克 100")
         .add_sma(n=20, color='orange')
         .add_bbands(n=20)
         .add_zigzag(change=2.0)
         .add_patterns()  # 自动标注 60 多种蜡烛图形态
         .add_volume()
         .add_macd()
         .add_rsi())

chart.show()  # 打开交互式 Plotly 图表
chart.save_image("analysis.png")  # 保存为高分辨率静态图片
```

### R 风格指标计算

```python
import pytafast

# Arnaud Legoux 移动平均线 (ALMA) - 极致平滑
alma = pytafast.ALMA(close, timeperiod=9, offset=0.85, sigma=6.0)

# 零滞后指数移动平均线 (ZLEMA)
zlema = pytafast.ZLEMA(close, timeperiod=30)

# 蔡金资金流量 (CMF)
cmf = pytafast.CMF(high, low, close, volume, timeperiod=20)

# 确定的事 (KST)
kst, signal = pytafast.KST(close)
```

## 常用计算示例

### 1. 策略均线组合
```python
import pytafast
# 传统均线与 R 风格高性能均线组合
df["sma"] = pytafast.SMA(df["close"], 20)
df["alma"] = pytafast.ALMA(df["close"], timeperiod=9, offset=0.85, sigma=6)
df["zlema"] = pytafast.ZLEMA(df["close"], 30)
```

### 2. 波动率与挤压检测
```python
upper_bb, mid_bb, lower_bb = pytafast.BBANDS(df["close"], 20)
upper_kc, mid_kc, lower_kc = pytafast.keltnerChannels(df["high"], df["low"], df["close"], 20)
# 检测布林带是否进入肯特纳通道内部（挤压状态）
df["squeeze"] = (upper_bb < upper_kc) & (lower_bb > lower_kc)
```

## 指标详细参考手册

> **注意：** 所有指标均同时支持 `numpy.ndarray` 和 `pandas.Series` 输入。当输入为 Series 时，会自动保留元数据（索引和名称）。

### 重叠研究 (Overlap Studies)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **ALMA** | `(inReal, timeperiod=9, offset=0.85, sigma=6.0)` | Arnaud Legoux Moving Average. |
| **BBANDS** | `(inReal, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=MAType.SMA)` | Bollinger Bands. Returns: (upperband, middleband, lowerband) |
| **DEMA** | `(inReal, timeperiod=30)` | DEMA indicator. |
| **EMA** | `(inReal, timeperiod=30)` | EMA indicator. |
| **EVWMA** | `(inReal, inVolume, timeperiod=30)` | Elastic Volume Weighted Moving Average. |
| **HMA** | `(inReal, timeperiod=20)` | Hull Moving Average. |
| **KAMA** | `(inReal, timeperiod=30)` | KAMA indicator. |
| **MA** | `(inReal, timeperiod=30, matype=0)` | Moving Average (generic). |
| **MAMA** | `(inReal, fastlimit=0.5, slowlimit=0.05)` | MESA Adaptive Moving Average. Returns: (mama, fama) |
| **MIDPOINT** | `(inReal, timeperiod=14)` | MIDPOINT indicator. |
| **MIDPRICE** | `(inHigh, inLow, timeperiod=14)` | MIDPRICE indicator. |
| **SAR** | `(inHigh, inLow, acceleration=0.02, maximum=0.2)` | Parabolic SAR. |
| **SMA** | `(inReal, timeperiod=30)` | SMA indicator. |
| **T3** | `(inReal, timeperiod=5, vfactor=0.7)` | Triple Exponential Moving Average (T3). |
| **TEMA** | `(inReal, timeperiod=30)` | TEMA indicator. |
| **TRIMA** | `(inReal, timeperiod=30)` | TRIMA indicator. |
| **WMA** | `(inReal, timeperiod=30)` | WMA indicator. |
| **ZLEMA** | `(inReal, timeperiod=30)` | Zero Lag Exponential Moving Average. |


### 动量指标 (Momentum Indicators)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **ADX** | `(inHigh, inLow, inClose, timeperiod=14)` | ADX indicator. |
| **ADXR** | `(inHigh, inLow, inClose, timeperiod=14)` | ADXR indicator. |
| **APO** | `(inReal, fastperiod=12, slowperiod=26, matype=0)` | Absolute Price Oscillator. |
| **AROON** | `(inHigh, inLow, timeperiod=14)` | Aroon. Returns: (aroondown, aroonup) |
| **AROONOSC** | `(inHigh, inLow, timeperiod=14)` | AROONOSC indicator. |
| **BOP** | `(inOpen, inHigh, inLow, inClose)` | Balance Of Power. |
| **CCI** | `(inHigh, inLow, inClose, timeperiod=14)` | CCI indicator. |
| **CMO** | `(inReal, timeperiod=14)` | CMO indicator. |
| **DX** | `(inHigh, inLow, inClose, timeperiod=14)` | DX indicator. |
| **KST** | `(inReal, nROC1=10, nROC2=15, nROC3=20, nROC4=30, nAvg1=10, nAvg2=10, nAvg3=10, nAvg4=15, nSig=9)` | Know Sure Thing (KST). Returns: (kst, signal) |
| **MACD** | `(inReal, fastperiod=12, slowperiod=26, signalperiod=9)` | Moving Average Convergence/Divergence. Returns: (macd, signal, hist) |
| **MACDEXT** | `(inReal, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)` | MACD with controllable MA type. |
| **MACDFIX** | `(inReal, signalperiod=9)` | MACD Fix 12/26. |
| **MFI** | `(inHigh, inLow, inClose, inVolume, timeperiod=14)` | Money Flow Index. |
| **MINUS_DI** | `(inHigh, inLow, inClose, timeperiod=14)` | MINUS_DI indicator. |
| **MINUS_DM** | `(inHigh, inLow, timeperiod=14)` | MINUS_DM indicator. |
| **MOM** | `(inReal, timeperiod=10)` | MOM indicator. |
| **PLUS_DI** | `(inHigh, inLow, inClose, timeperiod=14)` | PLUS_DI indicator. |
| **PLUS_DM** | `(inHigh, inLow, timeperiod=14)` | PLUS_DM indicator. |
| **PPO** | `(inReal, fastperiod=12, slowperiod=26, matype=0)` | Percentage Price Oscillator. |
| **ROC** | `(inReal, timeperiod=10)` | ROC indicator. |
| **ROCP** | `(inReal, timeperiod=10)` | ROCP indicator. |
| **ROCR** | `(inReal, timeperiod=10)` | ROCR indicator. |
| **ROCR100** | `(inReal, timeperiod=10)` | ROCR100 indicator. |
| **RSI** | `(inReal, timeperiod=14)` | RSI indicator. |
| **SMI** | `(inHigh, inLow, inClose, n=13, nFast=2, nSlow=25, nSig=9)` | Stochastic Momentum Index. Returns: (smi, signal) |
| **STOCH** | `(inHigh, inLow, inClose, fastk_period=5, slowk_period=3, slowk_matype=MAType.SMA, slowd_period=3, slowd_matype=MAType.SMA)` | Stochastic. Returns: (slowk, slowd) |
| **STOCHF** | `(inHigh, inLow, inClose, fastk_period=5, fastd_period=3, fastd_matype=0)` | Stochastic Fast. |
| **STOCHRSI** | `(inReal, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)` | Stochastic RSI. |
| **TRIX** | `(inReal, timeperiod=30)` | TRIX indicator. |
| **ULTOSC** | `(inHigh, inLow, inClose, timeperiod1=7, timeperiod2=14, timeperiod3=28)` | Ultimate Oscillator. |
| **WILLR** | `(inHigh, inLow, inClose, timeperiod=14)` | WILLR indicator. |


### 波动率指标 (Volatility Indicators)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **ATR** | `(inHigh, inLow, inClose, timeperiod=14)` | ATR indicator. |
| **DonchianChannel** | `(inHigh, inLow, timeperiod=10)` | Donchian Channel. Returns: (upper, middle, lower) |
| **NATR** | `(inHigh, inLow, inClose, timeperiod=14)` | NATR indicator. |
| **STDDEV** | `(inReal, timeperiod=5, nbdev=1.0)` | Standard Deviation. |
| **TRANGE** | `(inHigh, inLow, inClose)` | True Range. |
| **keltnerChannels** | `(inHigh, inLow, inClose, timeperiod=20, atr_mult=2.0)` | Keltner Channels. Returns: (upper, middle, lower) |


### 成交量指标 (Volume Indicators)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **AD** | `(inHigh, inLow, inClose, inVolume)` | Chaikin A/D Line. |
| **ADOSC** | `(inHigh, inLow, inClose, inVolume, fastperiod=3, slowperiod=10)` | Chaikin A/D Oscillator. |
| **CMF** | `(inHigh, inLow, inClose, inVolume, timeperiod=20)` | Chaikin Money Flow. |
| **EMV** | `(inHigh, inLow, inVolume, timeperiod=9, vol_divisor=10000.0)` | Arms' Ease of Movement Value. Returns: (emv, smoothed_emv) |
| **OBV** | `(inReal0, inReal1)` | OBV indicator. |


### 价格转换 (Price Transform)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **AVGPRICE** | `(inOpen, inHigh, inLow, inClose)` | Average Price. |
| **MEDPRICE** | `(inReal0, inReal1)` | MEDPRICE indicator. |
| **TYPPRICE** | `(inHigh, inLow, inClose)` | Typical Price. |
| **WCLPRICE** | `(inHigh, inLow, inClose)` | Weighted Close Price. |


### 周期指标 (Cycle Indicators)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **HT_DCPERIOD** | `(inReal)` | HT_DCPERIOD indicator. |
| **HT_DCPHASE** | `(inReal)` | HT_DCPHASE indicator. |
| **HT_PHASOR** | `(inReal)` | Hilbert Transform - Phasor Components. |
| **HT_SINE** | `(inReal)` | Hilbert Transform - SineWave. |
| **HT_TRENDLINE** | `(inReal)` | HT_TRENDLINE indicator. |
| **HT_TRENDMODE** | `(inReal)` | HT_TRENDMODE indicator. |


### 统计函数 (Statistics Functions)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **AVGDEV** | `(inReal, timeperiod=14)` | AVGDEV indicator. |
| **BETA** | `(inReal0, inReal1, timeperiod=5)` | BETA indicator. |
| **CORREL** | `(inReal0, inReal1, timeperiod=30)` | CORREL indicator. |
| **LINEARREG** | `(inReal, timeperiod=14)` | LINEARREG indicator. |
| **LINEARREG_ANGLE** | `(inReal, timeperiod=14)` | LINEARREG_ANGLE indicator. |
| **LINEARREG_INTERCEPT** | `(inReal, timeperiod=14)` | LINEARREG_INTERCEPT indicator. |
| **LINEARREG_SLOPE** | `(inReal, timeperiod=14)` | LINEARREG_SLOPE indicator. |
| **MAX** | `(inReal, timeperiod=30)` | MAX indicator. |
| **MIN** | `(inReal, timeperiod=30)` | MIN indicator. |
| **MINMAX** | `(inReal, timeperiod=30)` | Lowest and highest values over a specified period. |
| **MINMAXINDEX** | `(inReal, timeperiod=30)` | Indexes of lowest and highest values over a specified period. |
| **SUM** | `(inReal, timeperiod=30)` | SUM indicator. |
| **TSF** | `(inReal, timeperiod=14)` | TSF indicator. |
| **VAR** | `(inReal, timeperiod=5, nbdev=1.0)` | Variance. |


### 数学算子与变换 (Math Operators & Transforms)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **ACOS** | `(inReal)` | Vector ACOS. |
| **ADD** | `(inReal0, inReal1)` | ADD indicator. |
| **ASIN** | `(inReal)` | Vector ASIN. |
| **ATAN** | `(inReal)` | Vector ATAN. |
| **CEIL** | `(inReal)` | Vector CEIL. |
| **COS** | `(inReal)` | Vector COS. |
| **COSH** | `(inReal)` | Vector COSH. |
| **DIV** | `(inReal0, inReal1)` | DIV indicator. |
| **EXP** | `(inReal)` | Vector EXP. |
| **FLOOR** | `(inReal)` | Vector FLOOR. |
| **LN** | `(inReal)` | Vector LN. |
| **LOG10** | `(inReal)` | Vector LOG10. |
| **MULT** | `(inReal0, inReal1)` | MULT indicator. |
| **SIN** | `(inReal)` | Vector SIN. |
| **SINH** | `(inReal)` | Vector SINH. |
| **SQRT** | `(inReal)` | Vector SQRT. |
| **SUB** | `(inReal0, inReal1)` | SUB indicator. |
| **TAN** | `(inReal)` | Vector TAN. |
| **TANH** | `(inReal)` | Vector TANH. |


### 自定义与 R 语言原生 (Custom & R-Native)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **DPO** | `(inReal, timeperiod=10)` | Detrended Price Oscillator. |
| **GMMA** | `(inReal)` | Guppy Multiple Moving Average. Returns a tuple of 12 EMA series. |
| **SNR** | `(inHigh, inLow, inClose, timeperiod=14)` | Signal to Noise Ratio. |
| **VHF** | `(inReal, timeperiod=28)` | Vertical Horizontal Filter. |
| **ZIGZAG** | `(inHigh, inLow, change=10.0, percent=True)` | ZigZag indicator. |


### 蜡烛图形态 (Candlestick Patterns)

| 指标名称 | 参数签名 | 描述 |
| :--- | :--- | :--- |
| **CDL2CROWS** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDL2CROWS |
| **CDL3BLACKCROWS** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDL3BLACKCROWS |
| **CDL3INSIDE** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDL3INSIDE |
| **CDL3LINESTRIKE** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDL3LINESTRIKE |
| **CDL3OUTSIDE** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDL3OUTSIDE |
| **CDL3STARSINSOUTH** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDL3STARSINSOUTH |
| **CDL3WHITESOLDIERS** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDL3WHITESOLDIERS |
| **CDLABANDONEDBABY** | `(inOpen, inHigh, inLow, inClose, penetration=0.3)` | Candlestick Pattern: CDLABANDONEDBABY |
| **CDLADVANCEBLOCK** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLADVANCEBLOCK |
| **CDLBELTHOLD** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLBELTHOLD |
| **CDLBREAKAWAY** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLBREAKAWAY |
| **CDLCLOSINGMARUBOZU** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLCLOSINGMARUBOZU |
| **CDLCONCEALBABYSWALL** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLCONCEALBABYSWALL |
| **CDLCOUNTERATTACK** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLCOUNTERATTACK |
| **CDLDARKCLOUDCOVER** | `(inOpen, inHigh, inLow, inClose, penetration=0.5)` | Candlestick Pattern: CDLDARKCLOUDCOVER |
| **CDLDOJI** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLDOJI |
| **CDLDOJISTAR** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLDOJISTAR |
| **CDLDRAGONFLYDOJI** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLDRAGONFLYDOJI |
| **CDLENGULFING** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLENGULFING |
| **CDLEVENINGDOJISTAR** | `(inOpen, inHigh, inLow, inClose, penetration=0.3)` | Candlestick Pattern: CDLEVENINGDOJISTAR |
| **CDLEVENINGSTAR** | `(inOpen, inHigh, inLow, inClose, penetration=0.3)` | Candlestick Pattern: CDLEVENINGSTAR |
| **CDLGAPSIDESIDEWHITE** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLGAPSIDESIDEWHITE |
| **CDLGRAVESTONEDOJI** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLGRAVESTONEDOJI |
| **CDLHAMMER** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLHAMMER |
| **CDLHANGINGMAN** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLHANGINGMAN |
| **CDLHARAMI** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLHARAMI |
| **CDLHARAMICROSS** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLHARAMICROSS |
| **CDLHIGHWAVE** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLHIGHWAVE |
| **CDLHIKKAKE** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLHIKKAKE |
| **CDLHIKKAKEMOD** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLHIKKAKEMOD |
| **CDLHOMINGPIGEON** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLHOMINGPIGEON |
| **CDLIDENTICAL3CROWS** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLIDENTICAL3CROWS |
| **CDLINNECK** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLINNECK |
| **CDLINVERTEDHAMMER** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLINVERTEDHAMMER |
| **CDLKICKING** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLKICKING |
| **CDLKICKINGBYLENGTH** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLKICKINGBYLENGTH |
| **CDLLADDERBOTTOM** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLLADDERBOTTOM |
| **CDLLONGLEGGEDDOJI** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLLONGLEGGEDDOJI |
| **CDLLONGLINE** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLLONGLINE |
| **CDLMARUBOZU** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLMARUBOZU |
| **CDLMATCHINGLOW** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLMATCHINGLOW |
| **CDLMATHOLD** | `(inOpen, inHigh, inLow, inClose, penetration=0.5)` | Candlestick Pattern: CDLMATHOLD |
| **CDLMORNINGDOJISTAR** | `(inOpen, inHigh, inLow, inClose, penetration=0.3)` | Candlestick Pattern: CDLMORNINGDOJISTAR |
| **CDLMORNINGSTAR** | `(inOpen, inHigh, inLow, inClose, penetration=0.3)` | Candlestick Pattern: CDLMORNINGSTAR |
| **CDLONNECK** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLONNECK |
| **CDLPIERCING** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLPIERCING |
| **CDLRICKSHAWMAN** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLRICKSHAWMAN |
| **CDLRISEFALL3METHODS** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLRISEFALL3METHODS |
| **CDLSEPARATINGLINES** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLSEPARATINGLINES |
| **CDLSHOOTINGSTAR** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLSHOOTINGSTAR |
| **CDLSHORTLINE** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLSHORTLINE |
| **CDLSPINNINGTOP** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLSPINNINGTOP |
| **CDLSTALLEDPATTERN** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLSTALLEDPATTERN |
| **CDLSTICKSANDWICH** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLSTICKSANDWICH |
| **CDLTAKURI** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLTAKURI |
| **CDLTASUKIGAP** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLTASUKIGAP |
| **CDLTHRUSTING** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLTHRUSTING |
| **CDLTRISTAR** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLTRISTAR |
| **CDLUNIQUE3RIVER** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLUNIQUE3RIVER |
| **CDLUPSIDEGAP2CROWS** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLUPSIDEGAP2CROWS |
| **CDLXSIDEGAP3METHODS** | `(inOpen, inHigh, inLow, inClose)` | Candlestick Pattern: CDLXSIDEGAP3METHODS |

## 性能表现

`pytafast` 通过释放 GIL 实现了卓越的吞吐量，在多核环境下几乎呈线性扩展。

| 4 线程并发测试 | 官方 Python 包装器 | pytafast | **加速比** |
|:---|:---|:---|:---|
| **SMA** 并发 | 20.1 ms | 6.3 ms | **3.15x** |
| **MACD** 并发 | 75.0 ms | 20.7 ms | **3.62x** |

## 交叉验证

我们针对 R 语言 `TTR` 包维持着严格的数值对齐校验体系：

```bash
# 运行 150+ 指标的自动化对齐报告
./scripts/run_comparison.sh data/berkshire_1y.csv
```
详细对齐分析参见 [docs/comparison_report.md](docs/comparison_report.md)。

## 开源协议

MIT License. 静态链接 [TA-Lib](https://ta-lib.org/) (BSD)。
Copyright (c) 1999-2026, Curry Tang & Mario Fortier.
