import asyncio as _asyncio
import atexit
import sys as _sys
import types as _types
from typing import Any, cast

import numpy as np

# We import the compiled extension module
from . import pytafast_ext
from .plotting import Chart as Chart
from .pytafast_ext import MAType

__version__ = "0.4.0"

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
    if _HAS_PANDAS and isinstance(x, pd.Series):
        x = x.to_numpy()

    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags["C_CONTIGUOUS"]:
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
        low_val = _ensure_array(inLow)
        c = _ensure_array(inClose)
        out = ext_fn(h, low_val, c, timeperiod)
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
        low_val = _ensure_array(inLow)
        out = ext_fn(h, low_val, timeperiod)
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


def MAMA(inReal, fastlimit=0.5, slowlimit=0.05):
    """MESA Adaptive Moving Average. Returns: (mama, fama)"""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    mama_out, fama_out = pytafast_ext.MAMA(arr, fastlimit, slowlimit)
    if is_series:
        return (
            pd.Series(mama_out, index=inReal.index, name="MAMA"),
            pd.Series(fama_out, index=inReal.index, name="FAMA"),
        )
    return mama_out, fama_out


def BBANDS(inReal, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=MAType.SMA):
    """Bollinger Bands. Returns: (upperband, middleband, lowerband)"""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    # Handle both enum member and raw int
    if hasattr(matype, "value"):
        ma_int = int(cast(Any, matype).value)
    else:
        ma_int = int(matype)
    upper, middle, lower = pytafast_ext.BBANDS(
        arr, timeperiod, nbdevup, nbdevdn, ma_int
    )

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
    low_val = _ensure_array(inLow)
    out = pytafast_ext.SAR(h, low_val, acceleration, maximum)
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


def MACDEXT(
    inReal,
    fastperiod=12,
    fastmatype=0,
    slowperiod=26,
    slowmatype=0,
    signalperiod=9,
    signalmatype=0,
):
    """MACD with controllable MA type."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    macd, signal, hist = pytafast_ext.MACDEXT(
        arr,
        fastperiod,
        fastmatype,
        slowperiod,
        slowmatype,
        signalperiod,
        signalmatype,
    )
    if is_series:
        idx = inReal.index
        return (
            pd.Series(macd, index=idx, name="MACD"),
            pd.Series(signal, index=idx, name="MACDSignal"),
            pd.Series(hist, index=idx, name="MACDHist"),
        )
    return macd, signal, hist


def MACDFIX(inReal, signalperiod=9):
    """MACD Fix 12/26."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    macd, signal, hist = pytafast_ext.MACDFIX(arr, signalperiod)
    if is_series:
        idx = inReal.index
        return (
            pd.Series(macd, index=idx, name="MACD"),
            pd.Series(signal, index=idx, name="MACDSignal"),
            pd.Series(hist, index=idx, name="MACDHist"),
        )
    return macd, signal, hist


def STOCH(
    inHigh,
    inLow,
    inClose,
    fastk_period=5,
    slowk_period=3,
    slowk_matype=MAType.SMA,
    slowd_period=3,
    slowd_matype=MAType.SMA,
):
    """Stochastic. Returns: (slowk, slowd)"""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    if hasattr(slowk_matype, "value"):
        sk_t = int(cast(Any, slowk_matype).value)
    else:
        sk_t = int(slowk_matype)

    if hasattr(slowd_matype, "value"):
        sd_t = int(cast(Any, slowd_matype).value)
    else:
        sd_t = int(slowd_matype)

    slowk, slowd = pytafast_ext.STOCH(
        h, low_val, c, fastk_period, slowk_period, sk_t, slowd_period, sd_t
    )
    if is_series:
        return (
            pd.Series(slowk, index=inClose.index, name="SlowK"),
            pd.Series(slowd, index=inClose.index, name="SlowD"),
        )
    return slowk, slowd


def STOCHF(inHigh, inLow, inClose, fastk_period=5, fastd_period=3, fastd_matype=0):
    """Stochastic Fast."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    fastk, fastd = pytafast_ext.STOCHF(
        h, low_val, c, fastk_period, fastd_period, fastd_matype
    )
    if is_series:
        idx = inClose.index
        return (
            pd.Series(fastk, index=idx, name="FastK"),
            pd.Series(fastd, index=idx, name="FastD"),
        )
    return fastk, fastd


def STOCHRSI(inReal, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    """Stochastic RSI."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    fastk, fastd = pytafast_ext.STOCHRSI(
        arr, timeperiod, fastk_period, fastd_period, fastd_matype
    )
    if is_series:
        idx = inReal.index
        return (
            pd.Series(fastk, index=idx, name="FastK"),
            pd.Series(fastd, index=idx, name="FastD"),
        )
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
    low_val = _ensure_array(inLow)
    down, up = pytafast_ext.AROON(h, low_val, timeperiod)
    if is_series:
        return (
            pd.Series(down, index=inHigh.index, name="AROON_DOWN"),
            pd.Series(up, index=inHigh.index, name="AROON_UP"),
        )
    return down, up


def MFI(inHigh, inLow, inClose, inVolume, timeperiod=14):
    """Money Flow Index."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    v = _ensure_array(inVolume)
    out = pytafast_ext.MFI(h, low_val, c, v, timeperiod)
    if is_series:
        return pd.Series(out, index=inClose.index, name="MFI")
    return out


def ULTOSC(inHigh, inLow, inClose, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    """Ultimate Oscillator."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.ULTOSC(h, low_val, c, timeperiod1, timeperiod2, timeperiod3)
    if is_series:
        return pd.Series(out, index=inClose.index, name="ULTOSC")
    return out


def BOP(inOpen, inHigh, inLow, inClose):
    """Balance Of Power."""
    is_series = _is_pandas_series(inClose)
    o = _ensure_array(inOpen)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.BOP(o, h, low_val, c)
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
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.TRANGE(h, low_val, c)
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
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    v = _ensure_array(inVolume)
    out = pytafast_ext.AD(h, low_val, c, v)
    if is_series:
        return pd.Series(out, index=inClose.index, name="AD")
    return out


def ADOSC(inHigh, inLow, inClose, inVolume, fastperiod=3, slowperiod=10):
    """Chaikin A/D Oscillator."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    v = _ensure_array(inVolume)
    out = pytafast_ext.ADOSC(h, low_val, c, v, fastperiod, slowperiod)
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
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.AVGPRICE(o, h, low_val, c)
    if is_series:
        return pd.Series(out, index=inClose.index, name="AVGPRICE")
    return out


MEDPRICE = _make_dual_no_params("MEDPRICE")


def TYPPRICE(inHigh, inLow, inClose):
    """Typical Price."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.TYPPRICE(h, low_val, c)
    if is_series:
        return pd.Series(out, index=inClose.index, name="TYPPRICE")
    return out


def WCLPRICE(inHigh, inLow, inClose):
    """Weighted Close Price."""
    is_series = _is_pandas_series(inClose)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    c = _ensure_array(inClose)
    out = pytafast_ext.WCLPRICE(h, low_val, c)
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
        return (
            pd.Series(out_min, index=inReal.index, name="min"),
            pd.Series(out_max, index=inReal.index, name="max"),
        )
    return out_min, out_max


def MINMAXINDEX(inReal, timeperiod=30):
    """Indexes of lowest and highest values over a specified period."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out_minidx, out_maxidx = pytafast_ext.MINMAXINDEX(arr, timeperiod)
    if is_series:
        return (
            pd.Series(out_minidx, index=inReal.index, name="minidx"),
            pd.Series(out_maxidx, index=inReal.index, name="maxidx"),
        )
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
        return (
            pd.Series(inphase, index=inReal.index, name="inphase"),
            pd.Series(quadrature, index=inReal.index, name="quadrature"),
        )
    return inphase, quadrature


def HT_SINE(inReal):
    """Hilbert Transform - SineWave."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    sine, leadsine = pytafast_ext.HT_SINE(arr)
    if is_series:
        return (
            pd.Series(sine, index=inReal.index, name="sine"),
            pd.Series(leadsine, index=inReal.index, name="leadsine"),
        )
    return sine, leadsine


# ===================================================================
# Candlestick Patterns
# ===================================================================

_CDL_STANDARD = [
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]

_CDL_PENETRATION = {
    "CDLABANDONEDBABY": 0.3,
    "CDLDARKCLOUDCOVER": 0.5,
    "CDLEVENINGDOJISTAR": 0.3,
    "CDLEVENINGSTAR": 0.3,
    "CDLMATHOLD": 0.5,
    "CDLMORNINGDOJISTAR": 0.3,
    "CDLMORNINGSTAR": 0.3,
}


def _make_cdl_standard(name):
    ext_fn = getattr(pytafast_ext, name)

    def wrapper(inOpen, inHigh, inLow, inClose):
        is_series = _is_pandas_series(inClose)
        o = _ensure_array(inOpen)
        h = _ensure_array(inHigh)
        low_val = _ensure_array(inLow)
        c = _ensure_array(inClose)
        out = ext_fn(o, h, low_val, c)
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
        low_val = _ensure_array(inLow)
        c = _ensure_array(inClose)
        out = ext_fn(o, h, low_val, c, penetration)
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


def ZIGZAG(inHigh, inLow, change=10.0, percent=True):
    """ZigZag indicator."""
    is_series = _is_pandas_series(inHigh)
    h = _ensure_array(inHigh)
    low_val = _ensure_array(inLow)
    out = pytafast_ext.ZIGZAG(h, low_val, change, percent)
    if is_series:
        return pd.Series(out, index=inHigh.index, name="ZIGZAG")
    return out


def ALMA(inReal, timeperiod=9, offset=0.85, sigma=6.0):
    """Arnaud Legoux Moving Average."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out = pytafast_ext.ALMA(arr, timeperiod, offset, sigma)
    if is_series:
        return pd.Series(out, index=inReal.index, name="ALMA")
    return out


def EVWMA(inReal, inVolume, timeperiod=30):
    """Elastic Volume Weighted Moving Average."""
    is_series = _is_pandas_series(inReal)
    r = _ensure_array(inReal)
    v = _ensure_array(inVolume)
    out = pytafast_ext.EVWMA(r, v, timeperiod)
    if is_series:
        return pd.Series(out, index=inReal.index, name="EVWMA")
    return out


# ===================================================================
# R-consistent Indicators (Migrated from TTR/quantmod)
# ===================================================================


def DonchianChannel(inHigh, inLow, timeperiod=10):
    """Donchian Channel. Returns: (upper, middle, lower)"""
    upper = MAX(inHigh, timeperiod)
    lower = MIN(inLow, timeperiod)
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def GMMA(inReal):
    """Guppy Multiple Moving Average. Returns a tuple of 12 EMA series."""
    periods = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
    return tuple(EMA(inReal, timeperiod=p) for p in periods)


def KST(
    inReal,
    nROC1=10,
    nROC2=15,
    nROC3=20,
    nROC4=30,
    nAvg1=10,
    nAvg2=10,
    nAvg3=10,
    nAvg4=15,
    nSig=9,
):
    """Know Sure Thing (KST). Returns: (kst, signal)"""

    def _smooth_roc(n_roc, n_avg):
        r = ROC(inReal, n_roc)
        is_s = _is_pandas_series(r)
        if is_s:
            valid_r = r.dropna()
            if len(valid_r) < n_avg:
                return r * np.nan
            res = SMA(valid_r, n_avg).reindex(r.index)
        else:
            first_idx = n_roc
            if first_idx >= len(r):
                return np.full_like(r, np.nan)
            valid_r = r[first_idx:]
            s = SMA(valid_r, n_avg)
            res = np.full_like(r, np.nan)
            res[first_idx:] = s
        return res

    sr1 = _smooth_roc(nROC1, nAvg1)
    sr2 = _smooth_roc(nROC2, nAvg2)
    sr3 = _smooth_roc(nROC3, nAvg3)
    sr4 = _smooth_roc(nROC4, nAvg4)
    kst = sr1 * 1 + sr2 * 2 + sr3 * 3 + sr4 * 4

    if _is_pandas_series(kst):
        signal = SMA(kst.dropna(), nSig).reindex(kst.index)
    else:
        first_valid = (
            max(nROC1 + nAvg1, nROC2 + nAvg2, nROC3 + nAvg3, nROC4 + nAvg4) - 1
        )
        if first_valid >= len(kst):
            signal = np.full_like(kst, np.nan)
        else:
            s = SMA(kst[first_valid:], nSig)
            signal = np.full_like(kst, np.nan)
            signal[first_valid:] = s
    return kst, signal


def ZLEMA(inReal, timeperiod=30):
    """Zero Lag Exponential Moving Average."""
    is_series = _is_pandas_series(inReal)
    arr = _ensure_array(inReal)
    out = pytafast_ext.ZLEMA(arr, timeperiod)
    if is_series:
        return pd.Series(out, index=inReal.index, name="ZLEMA")
    return out


def HMA(inReal, timeperiod=20):
    """Hull Moving Average."""
    half_n = timeperiod // 2
    sqrt_n = int(np.sqrt(timeperiod))
    w1 = WMA(inReal, half_n)
    w2 = WMA(inReal, timeperiod)
    diff = 2 * w1 - w2
    # Handle NaNs from WMAs
    if _is_pandas_series(diff):
        return WMA(diff.dropna(), sqrt_n).reindex(inReal.index)
    else:
        res = np.full_like(inReal, np.nan)
        first_valid = timeperiod - 1
        if len(diff) <= first_valid:
            return res
        w = WMA(diff[first_valid:], sqrt_n)
        res[first_valid:] = w
        return res


def keltnerChannels(inHigh, inLow, inClose, timeperiod=20, atr_mult=2.0):
    """Keltner Channels. Returns: (upper, middle, lower)"""
    tp = (inHigh + inLow + inClose) / 3.0
    middle = EMA(tp, timeperiod)
    atr = ATR(inHigh, inLow, inClose, timeperiod)
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    return upper, middle, lower


def CMF(inHigh, inLow, inClose, inVolume, timeperiod=20):
    """Chaikin Money Flow."""
    # CLV = [(close - low) - (high - close)] / (high - low)
    # Simplified: (2*close - low - high) / (high - low)
    num = 2 * inClose - inLow - inHigh
    den = inHigh - inLow
    # Avoid div by zero
    clv = np.where(den != 0, num / den, 0.0)
    vol_clv = clv * inVolume
    return SUM(vol_clv, timeperiod) / SUM(inVolume, timeperiod)


def DPO(inReal, timeperiod=10):
    """Detrended Price Oscillator."""
    shift = timeperiod // 2 + 1
    ma = SMA(inReal, timeperiod)
    if _is_pandas_series(ma):
        shifted_ma = ma.shift(-shift)
        return inReal - shifted_ma
    else:
        res = np.full_like(inReal, np.nan)
        if len(ma) <= shift:
            return res
        res[:-shift] = inReal[:-shift] - ma[shift:]
        return res


def EMV(inHigh, inLow, inVolume, timeperiod=9, vol_divisor=10000.0):
    """Arms' Ease of Movement Value. Returns: (emv, smoothed_emv)"""
    mid = (inHigh + inLow) / 2.0
    # mid_move = mid - mid_prev
    is_s = _is_pandas_series(mid)
    mid_move = mid.diff() if is_s else np.diff(mid, prepend=np.nan)

    box_ratio = (inVolume / vol_divisor) / (inHigh - inLow)
    # R TTR replaces box_ratio infinites with NA
    box_ratio = np.where((inHigh - inLow) == 0, np.nan, box_ratio)

    emv = np.where(np.isnan(box_ratio), np.nan, mid_move / box_ratio)
    if is_s:
        # TTR EMV ma uses SMA natively
        with np.errstate(invalid='ignore'):
             ma_emv = SMA(emv, timeperiod)
        return pd.Series(emv, index=inHigh.index, name="emv"), pd.Series(ma_emv, index=inHigh.index, name="maEMV")
    else:
        ma_emv = np.full_like(emv, np.nan)
        valid_mask = ~np.isnan(emv)
        if valid_mask.sum() >= timeperiod:
             ma_emv[valid_mask] = SMA(emv[valid_mask], timeperiod)
    return emv, ma_emv


def VHF(inReal, timeperiod=28):
    """Vertical Horizontal Filter."""
    hmax = MAX(inReal, timeperiod)
    lmin = MIN(inReal, timeperiod)
    # Diff = abs(price - price_prev)
    is_series = _is_pandas_series(inReal)
    diff = inReal.diff().abs() if is_series else np.abs(np.diff(inReal, prepend=np.nan))

    if is_series:
        vol = SUM(diff.dropna(), timeperiod).reindex(inReal.index)
    else:
        vol = np.full_like(diff, np.nan)
        if len(diff) > 1:
            s = SUM(diff[1:], timeperiod)
            vol[1:] = s

    return (hmax - lmin) / vol


def SNR(inHigh, inLow, inClose, timeperiod=14):
    """Signal to Noise Ratio."""
    is_series = _is_pandas_series(inClose)
    # Change = abs(C - C_prev_n)
    change = (
        inClose.diff(timeperiod).abs()
        if is_series
        else np.abs(inClose - np.roll(inClose, timeperiod))
    )
    if not is_series:
        change[:timeperiod] = np.nan
    atr = ATR(inHigh, inLow, inClose, timeperiod)
    return change / atr


def SMI(inHigh, inLow, inClose, n=13, nFast=2, nSlow=25, nSig=9):
    """Stochastic Momentum Index. Returns: (smi, signal)"""
    hmax = MAX(inHigh, n)
    lmin = MIN(inLow, n)

    hl_diff = hmax - lmin
    c_diff = inClose - (hmax + lmin) / 2.0

    def _double_smooth(x):
        is_s = _is_pandas_series(x)
        if is_s:
            s1 = EMA(x.dropna(), nSlow)
            s2 = EMA(s1.dropna(), nFast).reindex(x.index)
            return s2
        else:
            res = np.full_like(x, np.nan)
            valid = x[~np.isnan(x)]
            if len(valid) < nSlow:
                return res
            s1 = EMA(valid, nSlow)
            valid2 = s1[~np.isnan(s1)]
            if len(valid2) < nFast:
                return res
            s2 = EMA(valid2, nFast)
            res[len(x) - len(s2) :] = s2
            return res

    num = _double_smooth(c_diff)
    den = _double_smooth(hl_diff)

    # Avoid div by zero
    smi = 100.0 * np.where(den != 0, num / (den / 2.0), 0.0)
    if _is_pandas_series(inClose):
        smi = pd.Series(smi, index=inClose.index, name="SMI")
        signal = EMA(smi.dropna(), nSig).reindex(smi.index)
    else:
        signal = np.full_like(smi, np.nan)
        valid = smi[~np.isnan(smi)]
        if len(valid) >= nSig:
            s = EMA(valid, nSig)
            signal[len(smi) - len(s) :] = s
    return smi, signal


# ===================================================================
# Async wrappers — built as a virtual submodule `pytafast.aio`
# ===================================================================


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
    "SMA",
    "EMA",
    "DEMA",
    "KAMA",
    "MA",
    "T3",
    "MAMA",
    "TEMA",
    "TRIMA",
    "WMA",
    "BBANDS",
    "SAR",
    "MIDPOINT",
    "MIDPRICE",
    # Momentum
    "RSI",
    "MACD",
    "MACDEXT",
    "MACDFIX",
    "MOM",
    "ROC",
    "ROCP",
    "ROCR",
    "ROCR100",
    "CMO",
    "APO",
    "PPO",
    "TRIX",
    "ADX",
    "ADXR",
    "CCI",
    "DX",
    "MINUS_DI",
    "MINUS_DM",
    "PLUS_DI",
    "PLUS_DM",
    "WILLR",
    "MFI",
    "STOCH",
    "STOCHF",
    "STOCHRSI",
    "AROON",
    "AROONOSC",
    "ULTOSC",
    "BOP",
    # Volatility
    "ATR",
    "NATR",
    "TRANGE",
    "STDDEV",
    # Volume
    "OBV",
    "AD",
    "ADOSC",
    # Price Transform
    "AVGPRICE",
    "MEDPRICE",
    "TYPPRICE",
    "WCLPRICE",
    # Statistics
    "BETA",
    "CORREL",
    "LINEARREG",
    "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT",
    "LINEARREG_SLOPE",
    "TSF",
    "VAR",
    "AVGDEV",
    "MAX",
    "MIN",
    "SUM",
    "MINMAX",
    "MINMAXINDEX",
    # Math Operators
    "ADD",
    "SUB",
    "MULT",
    "DIV",
    # Math Transforms
    "ACOS",
    "ASIN",
    "ATAN",
    "CEIL",
    "COS",
    "COSH",
    "EXP",
    "FLOOR",
    "LN",
    "LOG10",
    "SIN",
    "SINH",
    "SQRT",
    "TAN",
    "TANH",
    # Cycle
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR",
    "HT_SINE",
    "HT_TRENDLINE",
    "HT_TRENDMODE",
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
