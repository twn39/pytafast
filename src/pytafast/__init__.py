import numpy as np

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
    
    # We use copy=False and ascontiguousarray to achieve zero-copy whenever possible
    arr = np.asarray(inReal, dtype=np.float64)
    arr = np.ascontiguousarray(arr)
    
    # Call the C++ nanobind wrapper
    out_arr = pytafast_ext.SMA(arr, timeperiod)
    
    # Cast back to pandas Series if input was a Series
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name=inReal.name)
    
    return out_arr

def EMA(inReal, timeperiod=30):
    """
    Exponential Moving Average.
    """
    is_series = _is_pandas_series(inReal)
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
    # Convert Enum to int for C++ call if necessary, though nanobind handles enum bindings
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
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
    arr_c = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
    arr_v = np.ascontiguousarray(np.asarray(inVolume, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
    
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
    arr_v = np.ascontiguousarray(np.asarray(inVolume, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr = np.ascontiguousarray(np.asarray(inReal, dtype=np.float64))
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
    arr_h = np.ascontiguousarray(np.asarray(inHigh, dtype=np.float64))
    arr_l = np.ascontiguousarray(np.asarray(inLow, dtype=np.float64))
    arr_c = np.ascontiguousarray(np.asarray(inClose, dtype=np.float64))
    out_arr = pytafast_ext.ULTOSC(arr_h, arr_l, arr_c, timeperiod1, timeperiod2, timeperiod3)
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inClose.index, name="ULTOSC")
    return out_arr

# Initialize TA-Lib context when the module is loaded
pytafast_ext.initialize()

from pytafast import aio
