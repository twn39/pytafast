import pytest
import numpy as np
import pandas as pd
import pytalib

def test_sma_numpy():
    # 10 elements
    in_real = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    
    # Calculate SMA with period 3
    # Expected: 
    # [NaN, NaN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    out = pytalib.SMA(in_real, timeperiod=3)
    
    assert isinstance(out, np.ndarray)
    assert len(out) == 10
    
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[9] == pytest.approx(9.0)

def test_sma_pandas():
    # 10 elements
    in_real = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=list("abcdefghij"), name="prices")
    
    out = pytalib.SMA(in_real, timeperiod=5)
    
    assert isinstance(out, pd.Series)
    assert len(out) == 10
    assert out.name == "prices"
    assert out.index.tolist() == list("abcdefghij")
    
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[3])
    # SMA5 for element 4 (value 5) is sum(1,2,3,4,5)/5 = 3
    assert out.iloc[4] == pytest.approx(3.0)

def test_empty_array():
    in_real = np.array([])
    out = pytalib.SMA(in_real, timeperiod=3)
    assert len(out) == 0

def test_invalid_input():
    in_real = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(Exception):
        pytalib.SMA(in_real, timeperiod=3)

def test_sma_against_official_talib():
    talib = pytest.importorskip("talib")
    
    # Generate random price data
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    
    # Compare with official talib
    for period in [2, 5, 14, 30]:
        official_out = talib.SMA(in_real, timeperiod=period)
        our_out = pytalib.SMA(in_real, timeperiod=period)
        
        # We use np.testing.assert_allclose to handle NaNs and float precision
        np.testing.assert_allclose(our_out, official_out, equal_nan=True)

def test_ema_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    for period in [5, 14, 30]:
        official_out = talib.EMA(in_real, timeperiod=period)
        our_out = pytalib.EMA(in_real, timeperiod=period)
        np.testing.assert_allclose(our_out, official_out, equal_nan=True)

def test_rsi_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    for period in [5, 14, 30]:
        official_out = talib.RSI(in_real, timeperiod=period)
        our_out = pytalib.RSI(in_real, timeperiod=period)
        np.testing.assert_allclose(our_out, official_out, equal_nan=True)

def test_macd_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    
    # default params: 12, 26, 9
    o_macd, o_signal, o_hist = talib.MACD(in_real, fastperiod=12, slowperiod=26, signalperiod=9)
    p_macd, p_signal, p_hist = pytalib.MACD(in_real, fastperiod=12, slowperiod=26, signalperiod=9)
    
    np.testing.assert_allclose(p_macd, o_macd, equal_nan=True)
    np.testing.assert_allclose(p_signal, o_signal, equal_nan=True)
    np.testing.assert_allclose(p_hist, o_hist, equal_nan=True)

def test_bbands_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    
    for matype_val in [0, 1]:  # SMA, EMA
        o_upper, o_middle, o_lower = talib.BBANDS(in_real, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=matype_val)
        p_upper, p_middle, p_lower = pytalib.BBANDS(in_real, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=matype_val)
        
        np.testing.assert_allclose(p_upper, o_upper, equal_nan=True)
        np.testing.assert_allclose(p_middle, o_middle, equal_nan=True)
def test_atr_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.ATR(in_high, in_low, in_close, timeperiod=period)
        p_out = pytalib.ATR(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_adx_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.ADX(in_high, in_low, in_close, timeperiod=period)
        p_out = pytalib.ADX(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_cci_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.CCI(in_high, in_low, in_close, timeperiod=period)
        p_out = pytalib.CCI(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_obv_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    in_vol = np.random.random(100) * 1000
    o_out = talib.OBV(in_real, in_vol)
    p_out = pytalib.OBV(in_real, in_vol)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_roc_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10]:
        o_out = talib.ROC(in_real, timeperiod=period)
        p_out = pytalib.ROC(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_stoch_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    
    for matype_val in [0, 1]:  # SMA, EMA
        o_slowk, o_slowd = talib.STOCH(in_high, in_low, in_close, fastk_period=5, slowk_period=3, slowk_matype=matype_val, slowd_period=3, slowd_matype=matype_val)
        p_slowk, p_slowd = pytalib.STOCH(in_high, in_low, in_close, fastk_period=5, slowk_period=3, slowk_matype=matype_val, slowd_period=3, slowd_matype=matype_val)
        
        np.testing.assert_allclose(p_slowk, o_slowk, equal_nan=True)
        np.testing.assert_allclose(p_slowd, o_slowd, equal_nan=True)
