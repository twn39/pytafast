# pytafast 金融指标全量参考手册

本手册涵盖了 `pytafast` 中所有 170+ 个高性能金融指标。所有函数均支持 `numpy.ndarray` 和 `pandas.Series` 输入。

---

## 1. R 语言原生与迁移指标 (R-Native / TTR)
这些指标复刻了 R 语言 `TTR` 和 `quantmod` 的核心算法，具备高度的跨平台一致性。

| 名称 | 解释 | 入参 | 返回值 |
| :--- | :--- | :--- | :--- |
| **ALMA** | Arnaud Legoux MA | `real, n=9, offset=0.85, sigma=6` | 阿诺勒古平滑序列 |
| **ZLEMA** | Zero Lag EMA | `real, n=30` | 零滞后指数均线 |
| **EVWMA** | Elastic Volume WMA | `real, volume, n=30` | 弹性成交量加权均线 |
| **ZIGZAG** | 之字转向 | `high, low, change=10, percent=T` | 过滤噪音后的趋势折线 |
| **HMA** | Hull MA | `real, n=20` | 赫尔移动平均线 |
| **DonchianChannel** | 唐奇安通道 | `high, low, n=10` | `(upper, mid, lower)` |
| **keltnerChannels** | 肯特纳通道 | `high, low, close, n=20` | `(upper, mid, lower)` |
| **CMF** | Chaikin Money Flow | `high, low, close, vol, n=20` | 蔡金资金流量 |
| **KST** | Know Sure Thing | `real, ...` | `(kst, signal)` |
| **SMI** | Stochastic Momentum | `high, low, close, ...` | `(smi, signal)` |
| **GMMA** | 顾比复合均线 | `real` | 12 条 EMA 组成的元组 |
| **VHF** | 垂直水平过滤器 | `real, n=28` | 趋势/震荡强度判断 |
| **SNR** | 信噪比 | `high, low, close, n=14` | 信号质量评估 |
| **DPO** | 去趋势价格震荡 | `real, n=10` | 消除长期趋势后的价格波 |

---

## 2. 重叠研究 (Overlap Studies)
直接叠加在主图上的趋势指标。

*   **SMA**: 简单移动平均线。入参: `real, n=30`。
*   **EMA**: 指数移动平均线。入参: `real, n=30`。
*   **WMA**: 加权移动平均线。入参: `real, n=30`。
*   **DEMA**: 双指数移动平均线。入参: `real, n=30`。
*   **TEMA**: 三指数移动平均线。入参: `real, n=30`。
*   **TRIMA**: 三角移动平均线。入参: `real, n=30`。
*   **KAMA**: 考夫曼自适应均线。入参: `real, n=30`。
*   **MAMA**: MESA自适应均线。入参: `real, fast=0.5, slow=0.05`。返回: `(mama, fama)`。
*   **T3**: 三重指数均线 (T3)。入参: `real, n=5, vfactor=0.7`。
*   **BBANDS**: 布林带。入参: `real, n=5, up=2, dn=2, matype=0`。返回: `(upper, mid, lower)`。
*   **MA**: 通用移动平均线。支持切换 `matype`。
*   **SAR**: 抛物线转向。入参: `high, low, accel=0.02, max=0.2`。
*   **MIDPOINT**: 周期中点。入参: `real, n=14`。
*   **MIDPRICE**: 周期价格中点。入参: `high, low, n=14`。

---

## 3. 动量指标 (Momentum Indicators)
测量价格变动速率的指标。

*   **RSI**: 相对强弱指标。入参: `real, n=14`。
*   **MACD**: 指数平滑异同移动平均线。返回: `(macd, signal, hist)`。
*   **MACDEXT / MACDFIX**: 增强版/固定参数版 MACD。
*   **STOCH**: 随机指标 (KD)。返回: `(slowk, slowd)`。
*   **STOCHF**: 快速随机指标。返回: `(fastk, fastd)`。
*   **STOCHRSI**: 随机 RSI。返回: `(fastk, fastd)`。
*   **ADX / ADXR**: 平均趋向指标。入参: `high, low, close, n=14`。
*   **CCI**: 顺势指标。入参: `high, low, close, n=14`。
*   **MFI**: 资金流量指标。入参: `high, low, close, vol, n=14`。
*   **WILLR**: 威廉指标。入参: `high, low, close, n=14`。范围: `[-100, 0]`。
*   **ROC / ROCP / ROCR / ROCR100**: 各种变体的价格变动率。
*   **MOM**: 动量指标。即 `Price - PrevPrice`。
*   **CMO**: 钱德动量摆动指标。入参: `real, n=14`。
*   **DX / MINUS_DI / PLUS_DI / MINUS_DM / PLUS_DM**: 方向性运动指标系列。
*   **APO / PPO**: 绝对/百分比价格振荡器。
*   **AROON / AROONOSC**: 阿隆指标。返回: `(down, up)` 或单个振荡值。
*   **ULTOSC**: 终极震荡指标。入参: `high, low, close, t1=7, t2=14, t3=28`。
*   **BOP**: 均势指标。入参: `open, high, low, close`。
*   **TRIX**: 三重指数平滑平均线。入参: `real, n=30`。

---

## 4. 波动率与成交量 (Volatility & Volume)

*   **ATR**: 平均真实波幅。
*   **NATR**: 归一化平均真实波幅。
*   **TRANGE**: 真实波幅。
*   **STDDEV**: 标准差。注：`pytafast` 使用总体标准差 (除以 N)。
*   **OBV**: 能量潮。入参: `real, volume`。
*   **AD**: 蔡金累积/派发线。
*   **ADOSC**: 蔡金摆动指标。

---

## 5. 统计与数学运算 (Stats & Math)

*   **BETA / CORREL**: 贝塔系数与相关系数。
*   **LINEARREG / LINEARREG_SLOPE / LINEARREG_ANGLE / LINEARREG_INTERCEPT**: 线性回归系列。
*   **TSF**: 时间序列预测。
*   **VAR**: 方差。
*   **AVGDEV**: 平均绝对偏差。
*   **MAX / MIN / SUM**: 周期内最大、最小、总和。
*   **MINMAX / MINMAXINDEX**: 周期极值及其索引。返回: `(min, max)`。
*   **ADD / SUB / MULT / DIV**: 向量四则运算。
*   **ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH**: 向量数学函数。

---

## 6. 周期指标 (Cycle Indicators)
基于希尔伯特变换的周期分析。

*   **HT_DCPERIOD**: 主轴周期。
*   **HT_DCPHASE**: 主轴相位。
*   **HT_PHASOR**: 相量分量。返回: `(inphase, quadrature)`。
*   **HT_SINE**: 正弦波。返回: `(sine, leadsine)`。
*   **HT_TRENDLINE**: 瞬时趋势线。
*   **HT_TRENDMODE**: 趋势 vs 周期模式。

---

## 7. 蜡烛图模式 (Candlestick Patterns)
共 61 种。全部入参均为 `(open, high, low, close)`，返回值为整数序列（100 = 多，-100 = 空，0 = 无）。

**常用列表预览**:
- `CDL2CROWS`, `CDL3BLACKCROWS`, `CDL3INSIDE`, `CDL3LINESTRIKE`, `CDL3OUTSIDE`, `CDL3STARSINSOUTH`, `CDL3WHITESOLDIERS`
- `CDLADVANCEBLOCK`, `CDLBELTHOLD`, `CDLBREAKAWAY`, `CDLCLOSINGMARUBOZU`, `CDLCONCEALBABYSWALL`
- `CDLCOUNTERATTACK`, `CDLDOJI`, `CDLDOJISTAR`, `CDLDRAGONFLYDOJI`, `CDLENGULFING`
- `CDLHAMMER`, `CDLHANGINGMAN`, `CDLHARAMI`, `CDLMORNINGSTAR`, `CDLEVENINGSTAR`
- `CDLPIERCING`, `CDLSHOOTINGSTAR`, `CDLSPINNINGTOP`, `CDLTRISTAR`
- (完整列表参考 `dir(pytafast)` 以 CDL 开头的函数)

---

## 8. 参数对照表 (MAType)
| 值 | 映射 | 描述 |
| :--- | :--- | :--- |
| 0 | MAType.SMA | 简单移动平均 |
| 1 | MAType.EMA | 指数移动平均 |
| 2 | MAType.WMA | 加权移动平均 |
| 3 | MAType.DEMA | 双指数移动平均 |
| 4 | MAType.TEMA | 三指数移动平均 |
| 5 | MAType.TRIMA | 三角移动平均 |
| 6 | MAType.KAMA | 考夫曼自适应均线 |
| 7 | MAType.MAMA | MESA自适应均线 |
| 8 | MAType.T3 | 三重指数均线 (T3) |
