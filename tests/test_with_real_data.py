import pytest
import numpy as np
import pandas as pd
import pytafast
import os

# Data loading fixture
@pytest.fixture(params=[
    "samsung_3m.csv",
    "icbc_2025.csv",
    "nasdaq100_2025_now.csv",
    "sk_hynix_1y.csv",
    "berkshire_1y.csv"
])
def stock_data(request):
    data_path = os.path.join(os.path.dirname(__file__), "data", request.param)
    df = pd.read_csv(data_path)
    
    # Standardize column names to lowercase for easier access
    df.columns = [c.lower() for c in df.columns]
    
    # Map common column names if they differ
    mapping = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume"
    }
    df = df.rename(columns=mapping)
    
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df

def get_talib():
    return pytest.importorskip("talib")

# --- Test Generators/Parametrization ---

def compare_helper(stock_data, func_name, *args, **kwargs):
    talib = get_talib()
    
    # Map inputs based on function requirements (simplified heuristic)
    # Most talib functions take (real) or (high, low, close) or (open, high, low, close)
    
    o_func = getattr(talib, func_name)
    p_func = getattr(pytafast, func_name)
    
    # Heuristic to determine inputs
    # This is a bit complex to automate perfectly for all 150+ functions, 
    # so we'll group them or handle specific signatures.
    
    inputs = []
    # Simplified signature detection or explicit mapping
    if func_name in ["AD", "ADOSC", "MFI"]:
        # Volume functions (High, Low, Close, Volume)
        o_out = o_func(stock_data["high"].values, stock_data["low"].values, 
                       stock_data["close"].values, stock_data["volume"].values.astype(float), **kwargs)
        p_out = p_func(stock_data["high"].values, stock_data["low"].values, 
                       stock_data["close"].values, stock_data["volume"].values.astype(float), **kwargs)
    elif func_name == "OBV":
        # OBV takes (Close, Volume)
        o_out = o_func(stock_data["close"].values, stock_data["volume"].values.astype(float), **kwargs)
        p_out = p_func(stock_data["close"].values, stock_data["volume"].values.astype(float), **kwargs)
    elif func_name.startswith("CDL"):
        # Candlestick (Open, High, Low, Close)
        o_out = o_func(stock_data["open"].values, stock_data["high"].values, 
                       stock_data["low"].values, stock_data["close"].values, **kwargs)
        p_out = p_func(stock_data["open"].values, stock_data["high"].values, 
                       stock_data["low"].values, stock_data["close"].values, **kwargs)
    elif func_name in ["ATR", "ADX", "ADXR", "CCI", "DX", "MINUS_DI", "MINUS_DM", 
                       "PLUS_DI", "PLUS_DM", "WILLR", "AROON", "AROONOSC", "MIDPRICE", "ULTOSC", "STOCH", "STOCHF", "SAR"]:
        # High, Low, Close (or subset)
        if func_name in ["AROON", "AROONOSC", "MIDPRICE", "MINUS_DM", "PLUS_DM", "SAR"]:
            o_out = o_func(stock_data["high"].values, stock_data["low"].values, **kwargs)
            p_out = p_func(stock_data["high"].values, stock_data["low"].values, **kwargs)
        elif func_name in ["STOCH", "STOCHF", "ULTOSC", "ATR", "ADX", "ADXR", "CCI", "DX", "MINUS_DI", "PLUS_DI", "WILLR"]:
             o_out = o_func(stock_data["high"].values, stock_data["low"].values, stock_data["close"].values, **kwargs)
             p_out = p_func(stock_data["high"].values, stock_data["low"].values, stock_data["close"].values, **kwargs)
    elif func_name in ["AVGPRICE", "BOP"]:
        o_out = o_func(stock_data["open"].values, stock_data["high"].values, 
                       stock_data["low"].values, stock_data["close"].values, **kwargs)
        p_out = p_func(stock_data["open"].values, stock_data["high"].values, 
                       stock_data["low"].values, stock_data["close"].values, **kwargs)
    elif func_name in ["MEDPRICE"]:
        o_out = o_func(stock_data["high"].values, stock_data["low"].values, **kwargs)
        p_out = p_func(stock_data["high"].values, stock_data["low"].values, **kwargs)
    elif func_name in ["TYPPRICE", "WCLPRICE"]:
        o_out = o_func(stock_data["high"].values, stock_data["low"].values, stock_data["close"].values, **kwargs)
        p_out = p_func(stock_data["high"].values, stock_data["low"].values, stock_data["close"].values, **kwargs)
    else:
        # Default: Single input (Close)
        o_out = o_func(stock_data["close"].values, **kwargs)
        p_out = p_func(stock_data["close"].values, **kwargs)

    # Comparison
    if isinstance(o_out, tuple):
        for i in range(len(o_out)):
            np.testing.assert_allclose(p_out[i], o_out[i], equal_nan=True, err_msg=f"Mismatch in {func_name} tuple element {i}")
    else:
        # For integer outputs (candlesticks), they might be returned as float by talib wrapper sometimes
        if func_name.startswith("CDL") or func_name in ["HT_TRENDMODE", "MINMAXINDEX"]:
             valid = ~np.isnan(np.array(o_out, dtype=float))
             np.testing.assert_array_equal(np.array(p_out)[valid], np.array(o_out)[valid].astype(int), err_msg=f"Mismatch in {func_name}")
        else:
            np.testing.assert_allclose(p_out, o_out, equal_nan=True, err_msg=f"Mismatch in {func_name}")

# List of functions to test
FUNC_LIST = [
    "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MAMA",
    "RSI", "MACD", "MACDEXT", "MACDFIX", "STOCH", "STOCHF", "STOCHRSI",
    "BBANDS", "SAR", "ATR", "ADX", "ADXR", "CCI", "BOP", "CMO", "DX",
    "MFI", "MINUS_DI", "MINUS_DM", "MOM", "PLUS_DI", "PLUS_DM", "PPO",
    "ROC", "ROCP", "ROCR", "ROCR100", "TRIX", "ULTOSC", "WILLR",
    "AD", "ADOSC", "OBV",
    "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDLINE", "HT_TRENDMODE",
    "MAX", "MIN", "SUM", "STDDEV", "VAR",
] + [
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE", "CDL3OUTSIDE",
    "CDLENGULFING", "CDLHAMMER", "CDLHARAMI", "CDLMARUBOZU", "CDLSHOOTINGSTAR"
]

@pytest.mark.parametrize("func_name", FUNC_LIST)
def test_all_functions_real_data(stock_data, func_name):
    # Skip some functions that need extra parameters or have very specific behavior
    kwargs = {}
    if func_name == "MAMA":
        # MAMA needs fastlimit and slowlimit
        kwargs = {"fastlimit": 0.5, "slowlimit": 0.05}
    
    try:
        compare_helper(stock_data, func_name, **kwargs)
    except AttributeError:
        pytest.skip(f"Function {func_name} not implemented in pytafast yet")
    except Exception as e:
        # Some functions might fail due to data length or other specific reasons
        # but we want to know if there's a real mismatch
        raise e
