import numpy as np

# We import the compiled extension module
from . import pytalib_ext

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
    
    # Force a C-contiguous writable array by always copying
    arr = np.array(inReal, dtype=np.float64, copy=True, order='C')
    
    # Call the C++ nanobind wrapper
    out_arr = pytalib_ext.SMA(arr, timeperiod)
    
    if is_series:
        import pandas as pd
        return pd.Series(out_arr, index=inReal.index, name=inReal.name)
    
    return out_arr

# Initialize TA-Lib context when the module is loaded
pytalib_ext.initialize()

