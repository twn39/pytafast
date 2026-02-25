import numpy as np
import atexit

# We import the compiled extension module
from . import pytafast_ext
from .pytafast_ext import MAType

__version__ = "0.3.0"

# --- Module-level pandas detection (optimization #1) ---
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    _HAS_PANDAS = False


def _is_pandas_series(obj):
    return _HAS_PANDAS and isinstance(obj, pd.Series)


def _ensure_array(x):
    """Fast-path: skip np.ascontiguousarray when already float64 C-contiguous."""
    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags['C_CONTIGUOUS']:
        return x
    return np.ascontiguousarray(x, dtype=np.float64)


# ===================================================================
# Factory functions — eliminate ~700 lines of repetitive wrappers
# ===================================================================

def _make_single(name, default_timeperiod):
    """Factory for single-input indicators: f(inReal, timeperiod=N)"""
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inReal, timeperiod=default_timeperiod):
        is_series = _is_pandas_series(inReal)
        arr = _ensure_array(inReal)
        out = ext_fn(arr, timeperiod)
        if is_series:
            return pd.Series(out, index=inReal.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"{name} indicator."
    return wrapper


def _make_single_no_params(name):
    """Factory for single-input, no-param indicators: f(inReal)"""
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inReal):
        is_series = _is_pandas_series(inReal)
        arr = _ensure_array(inReal)
        out = ext_fn(arr)
        if is_series:
            return pd.Series(out, index=inReal.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"{name} indicator."
    return wrapper


def _make_hlc(name, default_timeperiod):
    """Factory for HLC indicators: f(inHigh, inLow, inClose, timeperiod=N)"""
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inHigh, inLow, inClose, timeperiod=default_timeperiod):
        is_series = _is_pandas_series(inClose)
        h = _ensure_array(inHigh)
        l = _ensure_array(inLow)
        c = _ensure_array(inClose)
        out = ext_fn(h, l, c, timeperiod)
        if is_series:
            return pd.Series(out, index=inClose.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"{name} indicator."
    return wrapper


def _make_hl(name, default_timeperiod):
    """Factory for HL indicators: f(inHigh, inLow, timeperiod=N)"""
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inHigh, inLow, timeperiod=default_timeperiod):
        is_series = _is_pandas_series(inHigh)
        h = _ensure_array(inHigh)
        l = _ensure_array(inLow)
        out = ext_fn(h, l, timeperiod)
        if is_series:
            return pd.Series(out, index=inHigh.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"{name} indicator."
    return wrapper


def _make_dual(name, default_timeperiod):
    """Factory for dual-input indicators: f(inReal0, inReal1, timeperiod=N)"""
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inReal0, inReal1, timeperiod=default_timeperiod):
        is_series = _is_pandas_series(inReal0)
        a0 = _ensure_array(inReal0)
        a1 = _ensure_array(inReal1)
        out = ext_fn(a0, a1, timeperiod)
        if is_series:
            return pd.Series(out, index=inReal0.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"{name} indicator."
    return wrapper


def _make_dual_no_params(name):
    """Factory for dual-input, no-param: f(inReal0, inReal1)"""
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inReal0, inReal1):
        is_series = _is_pandas_series(inReal0)
        a0 = _ensure_array(inReal0)
        a1 = _ensure_array(inReal1)
        out = ext_fn(a0, a1)
        if is_series:
            return pd.Series(out, index=inReal0.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"{name} indicator."
    return wrapper



# ===================================================================
# Overlap Studies
# ===================================================================

SMA = _make_single("SMA", 30)
EMA = _make_single("EMA", 30)
DEMA = _make_single("DEMA", 30)
KAMA = _make_single("KAMA", 30)
TEMA = _make_single("TEMA", 30)
TRIMA = _make_single("TRIMA", 30)
WMA = _make_single("WMA", 30)
MIDPOINT = _make_single("MIDPOINT", 14)


def MA(inReal, timeperiod=30, matype=0):
    """Moving Average (generic)."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out = pytafast_ext.MA(arr, timeperiod, matype)
    if is_series:
        return pd.Series(out, index=inReal.index, name="MA")
    return out


def T3(inReal, timeperiod=5, vfactor=0.7):
    """Triple Exponential Moving Average (T3)."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out = pytafast_ext.T3(arr, timeperiod, vfactor)
    if is_series:
        return pd.Series(out, index=inReal.index, name="T3")
    return out


def BBANDS(inReal, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=MAType.SMA):
    """Bollinger Bands. Returns: (upperband, middleband, lowerband)"""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    ma_int = int(matype.value) if hasattr(matype, 'value') else int(matype)
    upper, middle, lower = pytafast_ext.BBANDS(arr, timeperiod, nbdevup, nbdevdn, ma_int)
    if is_series:
        return (
            pd.Series(upper, index=inReal.index, name="UpperBand"),
            pd.Series(middle, index=inReal.index, name="MiddleBand"),
            pd.Series(lower, index=inReal.index, name="LowerBand"),
        )
    return upper, middle, lower


def SAR(inHigh, inLow, acceleration=0.02, maximum=0.2):
    """Parabolic SAR."""
    is_series = _is_pandas_series(inHigh)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    out = pytafast_ext.SAR(h, l, acceleration, maximum)
    if is_series:
        return pd.Series(out, index=inHigh.index, name="SAR")
    return out


MIDPRICE = _make_hl("MIDPRICE", 14)


# ===================================================================
# Momentum Indicators
# ===================================================================

RSI = _make_single("RSI", 14)
MOM = _make_single("MOM", 10)
ROC = _make_single("ROC", 10)
ROCP = _make_single("ROCP", 10)
ROCR = _make_single("ROCR", 10)
ROCR100 = _make_single("ROCR100", 10)
CMO = _make_single("CMO", 14)
TRIX = _make_single("TRIX", 30)


def APO(inReal, fastperiod=12, slowperiod=26, matype=0):
    """Absolute Price Oscillator."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out = pytafast_ext.APO(arr, fastperiod, slowperiod, matype)
    if is_series:
        return pd.Series(out, index=inReal.index, name="APO")
    return out


def PPO(inReal, fastperiod=12, slowperiod=26, matype=0):
    """Percentage Price Oscillator."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out = pytafast_ext.PPO(arr, fastperiod, slowperiod, matype)
    if is_series:
        return pd.Series(out, index=inReal.index, name="PPO")
    return out


def MACD(inReal, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence/Divergence. Returns: (macd, signal, hist)"""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    macd, signal, hist = pytafast_ext.MACD(arr, fastperiod, slowperiod, signalperiod)
    if is_series:
        return (
            pd.Series(macd, index=inReal.index, name="MACD"),
            pd.Series(signal, index=inReal.index, name="MACD_Signal"),
            pd.Series(hist, index=inReal.index, name="MACD_Hist"),
        )
    return macd, signal, hist


def MACDEXT(inReal, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0,
            signalperiod=9, signalmatype=0):
    """MACD with controllable MA type."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    macd, signal, hist = pytafast_ext.MACDEXT(
        arr, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype)
    if is_series:
        idx = inReal.index
        return (pd.Series(macd, index=idx, name="MACD"),
                pd.Series(signal, index=idx, name="MACDSignal"),
                pd.Series(hist, index=idx, name="MACDHist"))
    return macd, signal, hist


def MACDFIX(inReal, signalperiod=9):
    """MACD Fix 12/26."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    macd, signal, hist = pytafast_ext.MACDFIX(arr, signalperiod)
    if is_series:
        idx = inReal.index
        return (pd.Series(macd, index=idx, name="MACD"),
                pd.Series(signal, index=idx, name="MACDSignal"),
                pd.Series(hist, index=idx, name="MACDHist"))
    return macd, signal, hist


def STOCH(inHigh, inLow, inClose, fastk_period=5, slowk_period=3,
          slowk_matype=MAType.SMA, slowd_period=3, slowd_matype=MAType.SMA):
    """Stochastic. Returns: (slowk, slowd)"""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    sk_t = int(slowk_matype.value) if hasattr(slowk_matype, 'value') else int(slowk_matype)
    sd_t = int(slowd_matype.value) if hasattr(slowd_matype, 'value') else int(slowd_matype)
    slowk, slowd = pytafast_ext.STOCH(h, l, c, fastk_period, slowk_period, sk_t, slowd_period, sd_t)
    if is_series:
        return (pd.Series(slowk, index=inClose.index, name="SlowK"),
                pd.Series(slowd, index=inClose.index, name="SlowD"))
    return slowk, slowd


def STOCHF(inHigh, inLow, inClose, fastk_period=5, fastd_period=3, fastd_matype=0):
    """Stochastic Fast."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    fastk, fastd = pytafast_ext.STOCHF(h, l, c, fastk_period, fastd_period, fastd_matype)
    if is_series:
        idx = inClose.index
        return (pd.Series(fastk, index=idx, name="FastK"),
                pd.Series(fastd, index=idx, name="FastD"))
    return fastk, fastd


def STOCHRSI(inReal, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    """Stochastic RSI."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    fastk, fastd = pytafast_ext.STOCHRSI(arr, timeperiod, fastk_period, fastd_period, fastd_matype)
    if is_series:
        idx = inReal.index
        return (pd.Series(fastk, index=idx, name="FastK"),
                pd.Series(fastd, index=idx, name="FastD"))
    return fastk, fastd


ADX = _make_hlc("ADX", 14)
ADXR = _make_hlc("ADXR", 14)
CCI = _make_hlc("CCI", 14)
DX = _make_hlc("DX", 14)
MINUS_DI = _make_hlc("MINUS_DI", 14)
PLUS_DI = _make_hlc("PLUS_DI", 14)
WILLR = _make_hlc("WILLR", 14)

MINUS_DM = _make_hl("MINUS_DM", 14)
PLUS_DM = _make_hl("PLUS_DM", 14)

AROONOSC = _make_hl("AROONOSC", 14)


def AROON(inHigh, inLow, timeperiod=14):
    """Aroon. Returns: (aroondown, aroonup)"""
    is_series = _is_pandas_series(inHigh)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    down, up = pytafast_ext.AROON(h, l, timeperiod)
    if is_series:
        return (pd.Series(down, index=inHigh.index, name="AROON_DOWN"),
                pd.Series(up, index=inHigh.index, name="AROON_UP"))
    return down, up


def MFI(inHigh, inLow, inClose, inVolume, timeperiod=14):
    """Money Flow Index."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    v = _ensure_array(inVolume)
    out = pytafast_ext.MFI(h, l, c, v, timeperiod)
    if is_series:
        return pd.Series(out, index=inClose.index, name="MFI")
    return out


def ULTOSC(inHigh, inLow, inClose, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    """Ultimate Oscillator."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.ULTOSC(h, l, c, timeperiod1, timeperiod2, timeperiod3)
    if is_series:
        return pd.Series(out, index=inClose.index, name="ULTOSC")
    return out


def BOP(inOpen, inHigh, inLow, inClose):
    """Balance Of Power."""
    is_series = _is_pandas_series(inClose)
    o = _ensure_array(inOpen)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.BOP(o, h, l, c)
    if is_series:
        return pd.Series(out, index=inClose.index, name="BOP")
    return out


# ===================================================================
# Volatility
# ===================================================================

ATR = _make_hlc("ATR", 14)
NATR = _make_hlc("NATR", 14)


def TRANGE(inHigh, inLow, inClose):
    """True Range."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.TRANGE(h, l, c)
    if is_series:
        return pd.Series(out, index=inClose.index, name="TRANGE")
    return out


def STDDEV(inReal, timeperiod=5, nbdev=1.0):
    """Standard Deviation."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out = pytafast_ext.STDDEV(arr, timeperiod, nbdev)
    if is_series:
        return pd.Series(out, index=inReal.index, name="STDDEV")
    return out


# ===================================================================
# Volume
# ===================================================================

OBV = _make_dual_no_params("OBV")


def AD(inHigh, inLow, inClose, inVolume):
    """Chaikin A/D Line."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    v = _ensure_array(inVolume)
    out = pytafast_ext.AD(h, l, c, v)
    if is_series:
        return pd.Series(out, index=inClose.index, name="AD")
    return out


def ADOSC(inHigh, inLow, inClose, inVolume, fastperiod=3, slowperiod=10):
    """Chaikin A/D Oscillator."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    v = _ensure_array(inVolume)
    out = pytafast_ext.ADOSC(h, l, c, v, fastperiod, slowperiod)
    if is_series:
        return pd.Series(out, index=inClose.index, name="ADOSC")
    return out


# ===================================================================
# Price Transform
# ===================================================================

def AVGPRICE(inOpen, inHigh, inLow, inClose):
    """Average Price."""
    is_series = _is_pandas_series(inClose)
    o = _ensure_array(inOpen)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.AVGPRICE(o, h, l, c)
    if is_series:
        return pd.Series(out, index=inClose.index, name="AVGPRICE")
    return out


MEDPRICE = _make_dual_no_params("MEDPRICE")


def TYPPRICE(inHigh, inLow, inClose):
    """Typical Price."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.TYPPRICE(h, l, c)
    if is_series:
        return pd.Series(out, index=inClose.index, name="TYPPRICE")
    return out


def WCLPRICE(inHigh, inLow, inClose):
    """Weighted Close Price."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    l = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.WCLPRICE(h, l, c)
    if is_series:
        return pd.Series(out, index=inClose.index, name="WCLPRICE")
    return out


# ===================================================================
# Statistics
# ===================================================================

BETA = _make_dual("BETA", 5)
CORREL = _make_dual("CORREL", 30)
LINEARREG = _make_single("LINEARREG", 14)
LINEARREG_ANGLE = _make_single("LINEARREG_ANGLE", 14)
LINEARREG_INTERCEPT = _make_single("LINEARREG_INTERCEPT", 14)
LINEARREG_SLOPE = _make_single("LINEARREG_SLOPE", 14)
TSF = _make_single("TSF", 14)
AVGDEV = _make_single("AVGDEV", 14)
MAX = _make_single("MAX", 30)
MIN = _make_single("MIN", 30)
SUM = _make_single("SUM", 30)


def VAR(inReal, timeperiod=5, nbdev=1.0):
    """Variance."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out = pytafast_ext.VAR(arr, timeperiod, nbdev)
    if is_series:
        return pd.Series(out, index=inReal.index, name="VAR")
    return out


def MINMAX(inReal, timeperiod=30):
    """Lowest and highest values over a specified period."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out_min, out_max = pytafast_ext.MINMAX(arr, timeperiod)
    if is_series:
        return (pd.Series(out_min, index=inReal.index, name="min"),
                pd.Series(out_max, index=inReal.index, name="max"))
    return out_min, out_max


def MINMAXINDEX(inReal, timeperiod=30):
    """Indexes of lowest and highest values over a specified period."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out_minidx, out_maxidx = pytafast_ext.MINMAXINDEX(arr, timeperiod)
    if is_series:
        return (pd.Series(out_minidx, index=inReal.index, name="minidx"),
                pd.Series(out_maxidx, index=inReal.index, name="maxidx"))
    return out_minidx, out_maxidx


# ===================================================================
# Math Operators
# ===================================================================

ADD = _make_dual_no_params("ADD")
SUB = _make_dual_no_params("SUB")
MULT = _make_dual_no_params("MULT")
DIV = _make_dual_no_params("DIV")


# ===================================================================
# Math Transforms (factory, same as before)
# ===================================================================

def _make_math_transform(name):
    """Factory for single-input math transform wrappers."""
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inReal):
        is_series = _is_pandas_series(inReal)
        arr = _ensure_array(inReal)
        out = ext_fn(arr)
        if is_series:
            return pd.Series(out, index=inReal.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"Vector {name}."
    return wrapper


ACOS = _make_math_transform("ACOS")
ASIN = _make_math_transform("ASIN")
ATAN = _make_math_transform("ATAN")
CEIL = _make_math_transform("CEIL")
COS = _make_math_transform("COS")
COSH = _make_math_transform("COSH")
EXP = _make_math_transform("EXP")
FLOOR = _make_math_transform("FLOOR")
LN = _make_math_transform("LN")
LOG10 = _make_math_transform("LOG10")
SIN = _make_math_transform("SIN")
SINH = _make_math_transform("SINH")
SQRT = _make_math_transform("SQRT")
TAN = _make_math_transform("TAN")
TANH = _make_math_transform("TANH")


# ===================================================================
# Cycle Indicators
# ===================================================================

HT_DCPERIOD = _make_single_no_params("HT_DCPERIOD")
HT_DCPHASE = _make_single_no_params("HT_DCPHASE")
HT_TRENDLINE = _make_single_no_params("HT_TRENDLINE")
HT_TRENDMODE = _make_single_no_params("HT_TRENDMODE")


def HT_PHASOR(inReal):
    """Hilbert Transform - Phasor Components."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    inphase, quadrature = pytafast_ext.HT_PHASOR(arr)
    if is_series:
        return (pd.Series(inphase, index=inReal.index, name="inphase"),
                pd.Series(quadrature, index=inReal.index, name="quadrature"))
    return inphase, quadrature


def HT_SINE(inReal):
    """Hilbert Transform - SineWave."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    sine, leadsine = pytafast_ext.HT_SINE(arr)
    if is_series:
        return (pd.Series(sine, index=inReal.index, name="sine"),
                pd.Series(leadsine, index=inReal.index, name="leadsine"))
    return sine, leadsine


# ===================================================================
# Candlestick Patterns
# ===================================================================

_CDL_STANDARD = [
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE",
    "CDL3OUTSIDE", "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS",
    "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK",
    "CDLDOJI", "CDLDOJISTAR", "CDLDRAGONFLYDOJI", "CDLENGULFING",
    "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI", "CDLHAMMER",
    "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS", "CDLHIGHWAVE",
    "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS", "CDLINNECK", "CDLINVERTEDHAMMER",
    "CDLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI", "CDLLONGLINE", "CDLMARUBOZU",
    "CDLMATCHINGLOW", "CDLONNECK", "CDLPIERCING", "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR",
    "CDLSHORTLINE", "CDLSPINNINGTOP", "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH", "CDLTAKURI", "CDLTASUKIGAP", "CDLTHRUSTING",
    "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]

_CDL_PENETRATION = {
    "CDLABANDONEDBABY": 0.3, "CDLDARKCLOUDCOVER": 0.5,
    "CDLEVENINGDOJISTAR": 0.3, "CDLEVENINGSTAR": 0.3,
    "CDLMATHOLD": 0.5, "CDLMORNINGDOJISTAR": 0.3, "CDLMORNINGSTAR": 0.3,
}


def _make_cdl_standard(name):
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inOpen, inHigh, inLow, inClose):
        is_series = _is_pandas_series(inClose)
        o = _ensure_array(inOpen)
        h = _ensure_array(inHigh)
        l = _ensure_array(inLow)
        c = _ensure_array(inClose)
        out = ext_fn(o, h, l, c)
        if is_series:
            return pd.Series(out, index=inClose.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"Candlestick Pattern: {name}"
    return wrapper


def _make_cdl_penetration(name, default_pen):
    ext_fn = getattr(pytafast_ext, name)
    def wrapper(inOpen, inHigh, inLow, inClose, penetration=default_pen):
        is_series = _is_pandas_series(inClose)
        o = _ensure_array(inOpen)
        h = _ensure_array(inHigh)
        l = _ensure_array(inLow)
        c = _ensure_array(inClose)
        out = ext_fn(o, h, l, c, penetration)
        if is_series:
            return pd.Series(out, index=inClose.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"Candlestick Pattern: {name}"
    return wrapper


for _name in _CDL_STANDARD:
    globals()[_name] = _make_cdl_standard(_name)

for _name, _pen in _CDL_PENETRATION.items():
    globals()[_name] = _make_cdl_penetration(_name, _pen)


# ===================================================================
# Async wrappers — built as a virtual submodule `pytafast.aio`
# ===================================================================

import asyncio as _asyncio
import sys as _sys
import types as _types

def _make_async(sync_fn):
    async def wrapper(*args, **kwargs):
        return await _asyncio.to_thread(sync_fn, *args, **kwargs)
    wrapper.__name__ = sync_fn.__name__
    wrapper.__doc__ = sync_fn.__doc__
    return wrapper

# Build the aio namespace as a proper module object
aio = _types.ModuleType("pytafast.aio")
aio.__doc__ = """Async wrappers for all pytafast functions via asyncio.to_thread.

Usage:
    import pytafast
    result = await pytafast.aio.SMA(close, timeperiod=20)
"""

# Auto-generate async versions of all public indicator functions
_ALL_FUNCTIONS = [
    # Overlap
    "SMA", "EMA", "DEMA", "KAMA", "MA", "T3", "TEMA", "TRIMA", "WMA",
    "BBANDS", "SAR", "MIDPOINT", "MIDPRICE",
    # Momentum
    "RSI", "MACD", "MACDEXT", "MACDFIX", "MOM", "ROC", "ROCP", "ROCR",
    "ROCR100", "CMO", "APO", "PPO", "TRIX", "ADX", "ADXR", "CCI", "DX",
    "MINUS_DI", "MINUS_DM", "PLUS_DI", "PLUS_DM", "WILLR", "MFI",
    "STOCH", "STOCHF", "STOCHRSI", "AROON", "AROONOSC", "ULTOSC", "BOP",
    # Volatility
    "ATR", "NATR", "TRANGE", "STDDEV",
    # Volume
    "OBV", "AD", "ADOSC",
    # Price Transform
    "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE",
    # Statistics
    "BETA", "CORREL", "LINEARREG", "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT", "LINEARREG_SLOPE", "TSF", "VAR", "AVGDEV",
    "MAX", "MIN", "SUM", "MINMAX", "MINMAXINDEX",
    # Math Operators
    "ADD", "SUB", "MULT", "DIV",
    # Math Transforms
    "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH", "EXP", "FLOOR",
    "LN", "LOG10", "SIN", "SINH", "SQRT", "TAN", "TANH",
    # Cycle
    "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE",
    "HT_TRENDLINE", "HT_TRENDMODE",
]

for _fn_name in _ALL_FUNCTIONS:
    setattr(aio, _fn_name, _make_async(globals()[_fn_name]))

# Candlestick patterns
for _fn_name in _CDL_STANDARD + list(_CDL_PENETRATION.keys()):
    setattr(aio, _fn_name, _make_async(globals()[_fn_name]))

# Register as a proper submodule so `import pytafast.aio` also works
_sys.modules["pytafast.aio"] = aio


# ===================================================================
# Initialize TA-Lib context
# ===================================================================

pytafast_ext.initialize()
atexit.register(pytafast_ext.shutdown)
