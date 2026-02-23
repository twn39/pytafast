import numpy as np

# We import the compiled extension module
from . import pytalib_ext
from .pytalib_ext import MAType

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
    out_arr = pytalib_ext.SMA(arr, timeperiod)
    
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
    out_arr = pytalib_ext.EMA(arr, timeperiod)
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
    out_arr = pytalib_ext.RSI(arr, timeperiod)
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
    macd, macdsignal, macdhist = pytalib_ext.MACD(arr, fastperiod, slowperiod, signalperiod)
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
    upper, middle, lower = pytalib_ext.BBANDS(arr, timeperiod, nbdevup, nbdevdn, int(matype.value) if hasattr(matype, 'value') else int(matype))
    
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
    out_arr = pytalib_ext.ATR(arr_h, arr_l, arr_c, timeperiod)
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
    out_arr = pytalib_ext.ADX(arr_h, arr_l, arr_c, timeperiod)
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
    out_arr = pytalib_ext.CCI(arr_h, arr_l, arr_c, timeperiod)
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
    out_arr = pytalib_ext.OBV(arr_c, arr_v)
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
    out_arr = pytalib_ext.ROC(arr, timeperiod)
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
    
    slowk, slowd = pytalib_ext.STOCH(arr_h, arr_l, arr_c, fastk_period, slowk_period, sk_t, slowd_period, sd_t)
    
    if is_series:
        import pandas as pd
        return (
            pd.Series(slowk, index=inClose.index, name="SlowK"),
            pd.Series(slowd, index=inClose.index, name="SlowD")
        )
    return slowk, slowd

# Initialize TA-Lib context when the module is loaded
pytalib_ext.initialize()

from pytalib import aio
