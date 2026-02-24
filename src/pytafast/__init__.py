import numpy as np
import atexit

# We import the compiled extension module
from . import pytafast_ext
from .pytafast_ext import MAType

__version__ = "0.1.0"

def _is_pandas_series(obj):
    try:
        import pandas as pd
        return isinstance(obj, pd.Series)
    except ImportError:
        return False

def SMA(inReal, timeperiod=30):
    """
    Simple Moving Average.
    
    Args:
        inReal: 1D array-like (numpy ndarray, pandas Series, list, etc.) of floats.
        timeperiod: Integer time period.
        
    Returns:
        A numpy array or pandas Series (matching input) with the SMA values.
        The leading (timeperiod - 1) elements will be NaN.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.SMA(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name=inReal.name)
    return out_arr

def EMA(inReal, timeperiod=30):
    """
    Exponential Moving Average.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.EMA(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name=inReal.name)
    return out_arr

def RSI(inReal, timeperiod=14):
    """
    Relative Strength Index.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.RSI(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name=inReal.name)
    return out_arr

def MACD(inReal, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Moving Average Convergence/Divergence.
    Returns: (macd, macdsignal, macdhist)
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    macd, macdsignal, macdhist = pytafast_ext.MACD(arr, fastperiod, slowperiod, signalperiod)
    if is_series:
        import pandas as pd
        return (
            pd.Series(macd, index=inReal.index, name="MACD"),
            pd.Series(macdsignal, index=inReal.index, name="MACD_Signal"),
            pd.Series(macdhist, index=inReal.index, name="MACD_Hist")
        )
    return macd, macdsignal, macdhist

def BBANDS(inReal, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=MAType.SMA):
    """
    Bollinger Bands.
    Returns: (upperband, middleband, lowerband)
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    upper, middle, lower = pytafast_ext.BBANDS(arr, timeperiod, nbdevup, nbdevdn, int(matype.value) if hasattr(matype, 'value') else int(matype))
    
    if is_series:
        import pandas as pd
        return (
            pd.Series(upper, index=inReal.index, name="UpperBand"),
            pd.Series(middle, index=inReal.index, name="MiddleBand"),
            pd.Series(lower, index=inReal.index, name="LowerBand")
        )
    return upper, middle, lower

def ATR(inHigh, inLow, inClose, timeperiod=14):
    """
    Average True Range.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.ATR(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="ATR")
    return out_arr

def ADX(inHigh, inLow, inClose, timeperiod=14):
    """
    Average Directional Movement Index.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.ADX(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="ADX")
    return out_arr

def CCI(inHigh, inLow, inClose, timeperiod=14):
    """
    Commodity Channel Index.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.CCI(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="CCI")
    return out_arr

def OBV(inReal, inVolume):
    """
    On Balance Volume.
    """
    is_series = _is_pandas_series(inReal)
    arr_c = np.ascontiguousarray(inReal, dtype=np.float64)
    arr_v = np.ascontiguousarray(inVolume, dtype=np.float64)
    out_arr = pytafast_ext.OBV(arr_c, arr_v)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="OBV")
    return out_arr

def ROC(inReal, timeperiod=10):
    """
    Rate of change : ((price/prevPrice)-1)*100
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.ROC(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="ROC")
    return out_arr

def STOCH(inHigh, inLow, inClose, fastk_period=5, slowk_period=3, slowk_matype=MAType.SMA, slowd_period=3, slowd_matype=MAType.SMA):
    """
    Stochastic.
    Returns: (slowk, slowd)
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    
    sk_t = int(slowk_matype.value) if hasattr(slowk_matype, 'value') else int(slowk_matype)
    sd_t = int(slowd_matype.value) if hasattr(slowd_matype, 'value') else int(slowd_matype)
    
    slowk, slowd = pytafast_ext.STOCH(arr_h, arr_l, arr_c, fastk_period, slowk_period, sk_t, slowd_period, sd_t)
    
    if is_series:
        import pandas as pd
        return (
            pd.Series(slowk, index=inClose.index, name="SlowK"),
            pd.Series(slowd, index=inClose.index, name="SlowD")
        )
    return slowk, slowd

def MOM(inReal, timeperiod=10):
    """
    Momentum.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.MOM(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="MOM")
    return out_arr

def STDDEV(inReal, timeperiod=5, nbdev=1.0):
    """
    Standard Deviation.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.STDDEV(arr, timeperiod, nbdev)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="STDDEV")
    return out_arr

def WILLR(inHigh, inLow, inClose, timeperiod=14):
    """
    Williams %R.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.WILLR(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="WILLR")
    return out_arr

def NATR(inHigh, inLow, inClose, timeperiod=14):
    """
    Normalized Average True Range.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.NATR(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="NATR")
    return out_arr

def MFI(inHigh, inLow, inClose, inVolume, timeperiod=14):
    """
    Money Flow Index.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    arr_v = np.ascontiguousarray(inVolume, dtype=np.float64)
    out_arr = pytafast_ext.MFI(arr_h, arr_l, arr_c, arr_v, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="MFI")
    return out_arr

def CMO(inReal, timeperiod=14):
    """
    Chande Momentum Oscillator.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.CMO(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="CMO")
    return out_arr

def DX(inHigh, inLow, inClose, timeperiod=14):
    """
    Directional Movement Index.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.DX(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="DX")
    return out_arr

def MINUS_DI(inHigh, inLow, inClose, timeperiod=14):
    """
    Minus Directional Indicator.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.MINUS_DI(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="MINUS_DI")
    return out_arr

def MINUS_DM(inHigh, inLow, timeperiod=14):
    """
    Minus Directional Movement.
    """
    is_series = _is_pandas_series(inHigh)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    out_arr = pytafast_ext.MINUS_DM(arr_h, arr_l, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inHigh.index, name="MINUS_DM")
    return out_arr

def PLUS_DI(inHigh, inLow, inClose, timeperiod=14):
    """
    Plus Directional Indicator.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.PLUS_DI(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="PLUS_DI")
    return out_arr

def PLUS_DM(inHigh, inLow, timeperiod=14):
    """
    Plus Directional Movement.
    """
    is_series = _is_pandas_series(inHigh)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    out_arr = pytafast_ext.PLUS_DM(arr_h, arr_l, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inHigh.index, name="PLUS_DM")
    return out_arr

def APO(inReal, fastperiod=12, slowperiod=26, matype=0):
    """
    Absolute Price Oscillator.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.APO(arr, fastperiod, slowperiod, matype)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="APO")
    return out_arr

def AROON(inHigh, inLow, timeperiod=14):
    """
    Aroon.
    """
    is_series = _is_pandas_series(inHigh)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    out_down, out_up = pytafast_ext.AROON(arr_h, arr_l, timeperiod)
    if is_series:
        import pandas as pd
        return (
            pd.Series(out_down, index=inHigh.index, name="AROON_DOWN"),
            pd.Series(out_up, index=inHigh.index, name="AROON_UP")
        )
    return out_down, out_up

def AROONOSC(inHigh, inLow, timeperiod=14):
    """
    Aroon Oscillator.
    """
    is_series = _is_pandas_series(inHigh)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    out_arr = pytafast_ext.AROONOSC(arr_h, arr_l, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inHigh.index, name="AROONOSC")
    return out_arr

def PPO(inReal, fastperiod=12, slowperiod=26, matype=0):
    """
    Percentage Price Oscillator.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.PPO(arr, fastperiod, slowperiod, matype)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="PPO")
    return out_arr

def TRIX(inReal, timeperiod=30):
    """
    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.TRIX(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="TRIX")
    return out_arr

def ULTOSC(inHigh, inLow, inClose, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    """
    Ultimate Oscillator.
    """
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.ULTOSC(arr_h, arr_l, arr_c, timeperiod1, timeperiod2, timeperiod3)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="ULTOSC")
    return out_arr

def DEMA(inReal, timeperiod=30):
    """Double Exponential Moving Average."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.DEMA(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="DEMA")
    return out_arr

def KAMA(inReal, timeperiod=30):
    """Kaufman Adaptive Moving Average."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.KAMA(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="KAMA")
    return out_arr

def MA(inReal, timeperiod=30, matype=0):
    """Moving Average (generic)."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.MA(arr, timeperiod, matype)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="MA")
    return out_arr

def T3(inReal, timeperiod=5, vfactor=0.7):
    """Triple Exponential Moving Average (T3)."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.T3(arr, timeperiod, vfactor)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="T3")
    return out_arr

def TEMA(inReal, timeperiod=30):
    """Triple Exponential Moving Average."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.TEMA(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="TEMA")
    return out_arr

def TRIMA(inReal, timeperiod=30):
    """Triangular Moving Average."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.TRIMA(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="TRIMA")
    return out_arr

def WMA(inReal, timeperiod=30):
    """Weighted Moving Average."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.WMA(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="WMA")
    return out_arr

def SAR(inHigh, inLow, acceleration=0.02, maximum=0.2):
    """Parabolic SAR."""
    is_series = _is_pandas_series(inHigh)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    out_arr = pytafast_ext.SAR(arr_h, arr_l, acceleration, maximum)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inHigh.index, name="SAR")
    return out_arr

def TRANGE(inHigh, inLow, inClose):
    """True Range."""
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.TRANGE(arr_h, arr_l, arr_c)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="TRANGE")
    return out_arr

def AVGPRICE(inOpen, inHigh, inLow, inClose):
    """Average Price."""
    is_series = _is_pandas_series(inClose)
    arr_o = np.ascontiguousarray(inOpen, dtype=np.float64)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.AVGPRICE(arr_o, arr_h, arr_l, arr_c)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="AVGPRICE")
    return out_arr

def MEDPRICE(inHigh, inLow):
    """Median Price."""
    is_series = _is_pandas_series(inHigh)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    out_arr = pytafast_ext.MEDPRICE(arr_h, arr_l)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inHigh.index, name="MEDPRICE")
    return out_arr

def TYPPRICE(inHigh, inLow, inClose):
    """Typical Price."""
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.TYPPRICE(arr_h, arr_l, arr_c)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="TYPPRICE")
    return out_arr

def WCLPRICE(inHigh, inLow, inClose):
    """Weighted Close Price."""
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.WCLPRICE(arr_h, arr_l, arr_c)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="WCLPRICE")
    return out_arr

def MIDPOINT(inReal, timeperiod=14):
    """MidPoint over period."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.MIDPOINT(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="MIDPOINT")
    return out_arr

def MIDPRICE(inHigh, inLow, timeperiod=14):
    """Midpoint Price over period."""
    is_series = _is_pandas_series(inHigh)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    out_arr = pytafast_ext.MIDPRICE(arr_h, arr_l, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inHigh.index, name="MIDPRICE")
    return out_arr

# Initialize TA-Lib context when the module is loaded
pytafast_ext.initialize()
atexit.register(pytafast_ext.shutdown)

def ADXR(inHigh, inLow, inClose, timeperiod=14):
    """Average Directional Movement Index Rating."""
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.ADXR(arr_h, arr_l, arr_c, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="ADXR")
    return out_arr

def BOP(inOpen, inHigh, inLow, inClose):
    """Balance Of Power."""
    is_series = _is_pandas_series(inClose)
    arr_o = np.ascontiguousarray(inOpen, dtype=np.float64)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    out_arr = pytafast_ext.BOP(arr_o, arr_h, arr_l, arr_c)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="BOP")
    return out_arr

def MACDEXT(inReal, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0,
            signalperiod=9, signalmatype=0):
    """MACD with controllable MA type."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    macd, signal, hist = pytafast_ext.MACDEXT(arr, fastperiod, fastmatype,
                                               slowperiod, slowmatype,
                                               signalperiod, signalmatype)
    if is_series:
        import pandas as pd
        idx = inReal.index
        return (pd.Series(macd, index=idx, name="MACD"),
                pd.Series(signal, index=idx, name="MACDSignal"),
                pd.Series(hist, index=idx, name="MACDHist"))
    return macd, signal, hist

def MACDFIX(inReal, signalperiod=9):
    """MACD Fix 12/26."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    macd, signal, hist = pytafast_ext.MACDFIX(arr, signalperiod)
    if is_series:
        import pandas as pd
        idx = inReal.index
        return (pd.Series(macd, index=idx, name="MACD"),
                pd.Series(signal, index=idx, name="MACDSignal"),
                pd.Series(hist, index=idx, name="MACDHist"))
    return macd, signal, hist

def STOCHF(inHigh, inLow, inClose, fastk_period=5, fastd_period=3,
           fastd_matype=0):
    """Stochastic Fast."""
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    fastk, fastd = pytafast_ext.STOCHF(arr_h, arr_l, arr_c, fastk_period,
                                        fastd_period, fastd_matype)
    if is_series:
        import pandas as pd
        idx = inClose.index
        return (pd.Series(fastk, index=idx, name="FastK"),
                pd.Series(fastd, index=idx, name="FastD"))
    return fastk, fastd

def STOCHRSI(inReal, timeperiod=14, fastk_period=5, fastd_period=3,
             fastd_matype=0):
    """Stochastic RSI."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    fastk, fastd = pytafast_ext.STOCHRSI(arr, timeperiod, fastk_period,
                                          fastd_period, fastd_matype)
    if is_series:
        import pandas as pd
        idx = inReal.index
        return (pd.Series(fastk, index=idx, name="FastK"),
                pd.Series(fastd, index=idx, name="FastD"))
    return fastk, fastd

def ROCP(inReal, timeperiod=10):
    """Rate of change Percentage."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.ROCP(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="ROCP")
    return out_arr

def ROCR(inReal, timeperiod=10):
    """Rate of change ratio."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.ROCR(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="ROCR")
    return out_arr

def ROCR100(inReal, timeperiod=10):
    """Rate of change ratio 100 scale."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.ROCR100(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="ROCR100")
    return out_arr

def AD(inHigh, inLow, inClose, inVolume):
    """Chaikin A/D Line."""
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    arr_v = np.ascontiguousarray(inVolume, dtype=np.float64)
    out_arr = pytafast_ext.AD(arr_h, arr_l, arr_c, arr_v)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="AD")
    return out_arr

def ADOSC(inHigh, inLow, inClose, inVolume, fastperiod=3, slowperiod=10):
    """Chaikin A/D Oscillator."""
    is_series = _is_pandas_series(inClose)
    arr_h = np.ascontiguousarray(inHigh, dtype=np.float64)
    arr_l = np.ascontiguousarray(inLow, dtype=np.float64)
    arr_c = np.ascontiguousarray(inClose, dtype=np.float64)
    arr_v = np.ascontiguousarray(inVolume, dtype=np.float64)
    out_arr = pytafast_ext.ADOSC(arr_h, arr_l, arr_c, arr_v, fastperiod, slowperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="ADOSC")
    return out_arr

def BETA(inReal0, inReal1, timeperiod=5):
    """Beta."""
    is_series = _is_pandas_series(inReal0)
    arr0 = np.ascontiguousarray(inReal0, dtype=np.float64)
    arr1 = np.ascontiguousarray(inReal1, dtype=np.float64)
    out_arr = pytafast_ext.BETA(arr0, arr1, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal0.index, name="BETA")
    return out_arr

def CORREL(inReal0, inReal1, timeperiod=30):
    """Pearson's Correlation Coefficient."""
    is_series = _is_pandas_series(inReal0)
    arr0 = np.ascontiguousarray(inReal0, dtype=np.float64)
    arr1 = np.ascontiguousarray(inReal1, dtype=np.float64)
    out_arr = pytafast_ext.CORREL(arr0, arr1, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal0.index, name="CORREL")
    return out_arr

def LINEARREG(inReal, timeperiod=14):
    """Linear Regression."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.LINEARREG(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="LINEARREG")
    return out_arr

def LINEARREG_ANGLE(inReal, timeperiod=14):
    """Linear Regression Angle."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.LINEARREG_ANGLE(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="LINEARREG_ANGLE")
    return out_arr

def LINEARREG_INTERCEPT(inReal, timeperiod=14):
    """Linear Regression Intercept."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.LINEARREG_INTERCEPT(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="LINEARREG_INTERCEPT")
    return out_arr

def LINEARREG_SLOPE(inReal, timeperiod=14):
    """Linear Regression Slope."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.LINEARREG_SLOPE(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="LINEARREG_SLOPE")
    return out_arr

def TSF(inReal, timeperiod=14):
    """Time Series Forecast."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.TSF(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="TSF")
    return out_arr

def VAR(inReal, timeperiod=5, nbdev=1.0):
    """Variance."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.VAR(arr, timeperiod, nbdev)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="VAR")
    return out_arr

def AVGDEV(inReal, timeperiod=14):
    """Average Deviation."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.AVGDEV(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="AVGDEV")
    return out_arr

def ADD(inReal0, inReal1):
    """Vector Arithmetic Add."""
    is_series = _is_pandas_series(inReal0)
    arr0 = np.ascontiguousarray(inReal0, dtype=np.float64)
    arr1 = np.ascontiguousarray(inReal1, dtype=np.float64)
    out_arr = pytafast_ext.ADD(arr0, arr1)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal0.index, name="ADD")
    return out_arr

def SUB(inReal0, inReal1):
    """Vector Arithmetic Subtraction."""
    is_series = _is_pandas_series(inReal0)
    arr0 = np.ascontiguousarray(inReal0, dtype=np.float64)
    arr1 = np.ascontiguousarray(inReal1, dtype=np.float64)
    out_arr = pytafast_ext.SUB(arr0, arr1)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal0.index, name="SUB")
    return out_arr

def MULT(inReal0, inReal1):
    """Vector Arithmetic Multiply."""
    is_series = _is_pandas_series(inReal0)
    arr0 = np.ascontiguousarray(inReal0, dtype=np.float64)
    arr1 = np.ascontiguousarray(inReal1, dtype=np.float64)
    out_arr = pytafast_ext.MULT(arr0, arr1)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal0.index, name="MULT")
    return out_arr

def DIV(inReal0, inReal1):
    """Vector Arithmetic Division."""
    is_series = _is_pandas_series(inReal0)
    arr0 = np.ascontiguousarray(inReal0, dtype=np.float64)
    arr1 = np.ascontiguousarray(inReal1, dtype=np.float64)
    out_arr = pytafast_ext.DIV(arr0, arr1)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal0.index, name="DIV")
    return out_arr

def _make_math_transform(name):
    """Factory for single-input math transform wrappers."""
    def wrapper(inReal):
        is_series = _is_pandas_series(inReal)
        arr = np.ascontiguousarray(inReal, dtype=np.float64)
        out_arr = getattr(pytafast_ext, name)(arr)
        if is_series:
            import pandas as pd
            return pd.Series(out_arr, index=inReal.index, name=name)
        return out_arr
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

def MAX(inReal, timeperiod=30):
    """Highest value over a specified period."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.MAX(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="MAX")
    return out_arr

def MIN(inReal, timeperiod=30):
    """Lowest value over a specified period."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.MIN(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="MIN")
    return out_arr

def SUM(inReal, timeperiod=30):
    """Summation."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.SUM(arr, timeperiod)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="SUM")
    return out_arr

def MINMAX(inReal, timeperiod=30):
    """Lowest and highest values over a specified period."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_min, out_max = pytafast_ext.MINMAX(arr, timeperiod)
    if is_series:
        import pandas as pd
        return (pd.Series(out_min, index=inReal.index, name="min"),
                pd.Series(out_max, index=inReal.index, name="max"))
    return out_min, out_max

def MINMAXINDEX(inReal, timeperiod=30):
    """Indexes of lowest and highest values over a specified period."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_minidx, out_maxidx = pytafast_ext.MINMAXINDEX(arr, timeperiod)
    if is_series:
        import pandas as pd
        return (pd.Series(out_minidx, index=inReal.index, name="minidx"),
                pd.Series(out_maxidx, index=inReal.index, name="maxidx"))
    return out_minidx, out_maxidx

def HT_DCPERIOD(inReal):
    """Hilbert Transform - Dominant Cycle Period."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.HT_DCPERIOD(arr)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="HT_DCPERIOD")
    return out_arr

def HT_DCPHASE(inReal):
    """Hilbert Transform - Dominant Cycle Phase."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.HT_DCPHASE(arr)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="HT_DCPHASE")
    return out_arr

def HT_PHASOR(inReal):
    """Hilbert Transform - Phasor Components."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    inphase, quadrature = pytafast_ext.HT_PHASOR(arr)
    if is_series:
        import pandas as pd
        return (pd.Series(inphase, index=inReal.index, name="inphase"),
                pd.Series(quadrature, index=inReal.index, name="quadrature"))
    return inphase, quadrature

def HT_SINE(inReal):
    """Hilbert Transform - SineWave."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    sine, leadsine = pytafast_ext.HT_SINE(arr)
    if is_series:
        import pandas as pd
        return (pd.Series(sine, index=inReal.index, name="sine"),
                pd.Series(leadsine, index=inReal.index, name="leadsine"))
    return sine, leadsine

def HT_TRENDLINE(inReal):
    """Hilbert Transform - Instantaneous Trendline."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.HT_TRENDLINE(arr)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="HT_TRENDLINE")
    return out_arr

def HT_TRENDMODE(inReal):
    """Hilbert Transform - Trend vs Cycle Mode."""
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(inReal, dtype=np.float64)
    out_arr = pytafast_ext.HT_TRENDMODE(arr)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name="HT_TRENDMODE")
    return out_arr

# --- Candlestick Patterns ---
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
    def wrapper(inOpen, inHigh, inLow, inClose):
        is_series = _is_pandas_series(inClose)
        o = np.ascontiguousarray(inOpen, dtype=np.float64)
        h = np.ascontiguousarray(inHigh, dtype=np.float64)
        l = np.ascontiguousarray(inLow, dtype=np.float64)
        c = np.ascontiguousarray(inClose, dtype=np.float64)
        out = getattr(pytafast_ext, name)(o, h, l, c)
        if is_series:
            import pandas as pd
            return pd.Series(out, index=inClose.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"Candlestick Pattern: {name}"
    return wrapper

def _make_cdl_penetration(name, default_pen):
    def wrapper(inOpen, inHigh, inLow, inClose, penetration=default_pen):
        is_series = _is_pandas_series(inClose)
        o = np.ascontiguousarray(inOpen, dtype=np.float64)
        h = np.ascontiguousarray(inHigh, dtype=np.float64)
        l = np.ascontiguousarray(inLow, dtype=np.float64)
        c = np.ascontiguousarray(inClose, dtype=np.float64)
        out = getattr(pytafast_ext, name)(o, h, l, c, penetration)
        if is_series:
            import pandas as pd
            return pd.Series(out, index=inClose.index, name=name)
        return out
    wrapper.__name__ = name
    wrapper.__doc__ = f"Candlestick Pattern: {name}"
    return wrapper

# Generate all standard CDL wrappers
for _name in _CDL_STANDARD:
    globals()[_name] = _make_cdl_standard(_name)

# Generate all penetration CDL wrappers
for _name, _pen in _CDL_PENETRATION.items():
    globals()[_name] = _make_cdl_penetration(_name, _pen)

from pytafast import aio


