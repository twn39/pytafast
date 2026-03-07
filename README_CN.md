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

## 指标库概览

| 类别 | 包含指标 (部分列举) |
|:---|:---|
| **R-Native (新)** | **ALMA, ZLEMA, EVWMA, ZIGZAG, HMA, Donchian, Keltner, CMF, KST, SMI, VHF, SNR** |
| **Overlap** | SMA, EMA, WMA, DEMA, TEMA, KAMA, BBANDS, SAR, MIDPOINT... |
| **Momentum** | RSI, MACD, STOCH, ADX, CCI, ROC, MOM, WILLR, MFI... |
| **Volatility** | ATR, NATR, TRANGE, STDDEV |
| **Pattern** | 61 种形态: CDLHAMMER, CDLENGULFING, CDLDOJI, CDLMORNINGSTAR... |

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
