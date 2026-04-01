# Detailed Indicators Reference

> **Note:** All indicators support both `numpy.ndarray` and `pandas.Series` as input. When a Series is provided, metadata (index and name) is preserved.

### Overlap Studies

| Name | Signature | Description |
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


### Momentum Indicators

| Name | Signature | Description |
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


### Volatility Indicators

| Name | Signature | Description |
| :--- | :--- | :--- |
| **ATR** | `(inHigh, inLow, inClose, timeperiod=14)` | ATR indicator. |
| **DonchianChannel** | `(inHigh, inLow, timeperiod=10)` | Donchian Channel. Returns: (upper, middle, lower) |
| **NATR** | `(inHigh, inLow, inClose, timeperiod=14)` | NATR indicator. |
| **STDDEV** | `(inReal, timeperiod=5, nbdev=1.0)` | Standard Deviation. |
| **TRANGE** | `(inHigh, inLow, inClose)` | True Range. |
| **keltnerChannels** | `(inHigh, inLow, inClose, timeperiod=20, atr_mult=2.0)` | Keltner Channels. Returns: (upper, middle, lower) |


### Volume Indicators

| Name | Signature | Description |
| :--- | :--- | :--- |
| **AD** | `(inHigh, inLow, inClose, inVolume)` | Chaikin A/D Line. |
| **ADOSC** | `(inHigh, inLow, inClose, inVolume, fastperiod=3, slowperiod=10)` | Chaikin A/D Oscillator. |
| **CMF** | `(inHigh, inLow, inClose, inVolume, timeperiod=20)` | Chaikin Money Flow. |
| **EMV** | `(inHigh, inLow, inVolume, timeperiod=9, vol_divisor=10000.0)` | Arms' Ease of Movement Value. Returns: (emv, smoothed_emv) |
| **OBV** | `(inReal0, inReal1)` | OBV indicator. |


### Price Transform

| Name | Signature | Description |
| :--- | :--- | :--- |
| **AVGPRICE** | `(inOpen, inHigh, inLow, inClose)` | Average Price. |
| **MEDPRICE** | `(inReal0, inReal1)` | MEDPRICE indicator. |
| **TYPPRICE** | `(inHigh, inLow, inClose)` | Typical Price. |
| **WCLPRICE** | `(inHigh, inLow, inClose)` | Weighted Close Price. |


### Cycle Indicators

| Name | Signature | Description |
| :--- | :--- | :--- |
| **HT_DCPERIOD** | `(inReal)` | HT_DCPERIOD indicator. |
| **HT_DCPHASE** | `(inReal)` | HT_DCPHASE indicator. |
| **HT_PHASOR** | `(inReal)` | Hilbert Transform - Phasor Components. |
| **HT_SINE** | `(inReal)` | Hilbert Transform - SineWave. |
| **HT_TRENDLINE** | `(inReal)` | HT_TRENDLINE indicator. |
| **HT_TRENDMODE** | `(inReal)` | HT_TRENDMODE indicator. |


### Statistics Functions

| Name | Signature | Description |
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


### Math Operators & Transforms

| Name | Signature | Description |
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


### Custom & R-Native

| Name | Signature | Description |
| :--- | :--- | :--- |
| **DPO** | `(inReal, timeperiod=10)` | Detrended Price Oscillator. |
| **GMMA** | `(inReal)` | Guppy Multiple Moving Average. Returns a tuple of 12 EMA series. |
| **SNR** | `(inHigh, inLow, inClose, timeperiod=14)` | Signal to Noise Ratio. |
| **VHF** | `(inReal, timeperiod=28)` | Vertical Horizontal Filter. |
| **ZIGZAG** | `(inHigh, inLow, change=10.0, percent=True)` | ZigZag indicator. |


### Candlestick Patterns

| Name | Signature | Description |
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


