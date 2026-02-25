import pytest
import numpy as np
import pandas as pd
import pytafast

def test_sma_numpy():
    # 10 elements
    in_real = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    
    # Calculate SMA with period 3
    # Expected: 
    # [NaN, NaN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    out = pytafast.SMA(in_real, timeperiod=3)
    
    assert isinstance(out, np.ndarray)
    assert len(out) == 10
    
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[9] == pytest.approx(9.0)

def test_sma_pandas():
    # 10 elements
    in_real = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=list("abcdefghij"), name="prices")
    
    out = pytafast.SMA(in_real, timeperiod=5)
    
    assert isinstance(out, pd.Series)
    assert len(out) == 10
    assert out.name == "SMA"
    assert out.index.tolist() == list("abcdefghij")
    
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[3])
    # SMA5 for element 4 (value 5) is sum(1,2,3,4,5)/5 = 3
    assert out.iloc[4] == pytest.approx(3.0)

def test_empty_array():
    in_real = np.array([])
    out = pytafast.SMA(in_real, timeperiod=3)
    assert len(out) == 0

def test_invalid_input():
    in_real = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(Exception):
        pytafast.SMA(in_real, timeperiod=3)

def test_sma_against_official_talib():
    talib = pytest.importorskip("talib")
    
    # Generate random price data
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    
    # Compare with official talib
    for period in [2, 5, 14, 30]:
        official_out = talib.SMA(in_real, timeperiod=period)
        our_out = pytafast.SMA(in_real, timeperiod=period)
        
        # We use np.testing.assert_allclose to handle NaNs and float precision
        np.testing.assert_allclose(our_out, official_out, equal_nan=True)

def test_ema_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    for period in [5, 14, 30]:
        official_out = talib.EMA(in_real, timeperiod=period)
        our_out = pytafast.EMA(in_real, timeperiod=period)
        np.testing.assert_allclose(our_out, official_out, equal_nan=True)

def test_rsi_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    for period in [5, 14, 30]:
        official_out = talib.RSI(in_real, timeperiod=period)
        our_out = pytafast.RSI(in_real, timeperiod=period)
        np.testing.assert_allclose(our_out, official_out, equal_nan=True)

def test_macd_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    
    # default params: 12, 26, 9
    o_macd, o_signal, o_hist = talib.MACD(in_real, fastperiod=12, slowperiod=26, signalperiod=9)
    p_macd, p_signal, p_hist = pytafast.MACD(in_real, fastperiod=12, slowperiod=26, signalperiod=9)
    
    np.testing.assert_allclose(p_macd, o_macd, equal_nan=True)
    np.testing.assert_allclose(p_signal, o_signal, equal_nan=True)
    np.testing.assert_allclose(p_hist, o_hist, equal_nan=True)

def test_bbands_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    
    for matype_val in [0, 1]:  # SMA, EMA
        o_upper, o_middle, o_lower = talib.BBANDS(in_real, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=matype_val)
        p_upper, p_middle, p_lower = pytafast.BBANDS(in_real, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=matype_val)
        
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
        p_out = pytafast.ATR(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_adx_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.ADX(in_high, in_low, in_close, timeperiod=period)
        p_out = pytafast.ADX(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_cci_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.CCI(in_high, in_low, in_close, timeperiod=period)
        p_out = pytafast.CCI(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_obv_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    in_vol = np.random.random(100) * 1000
    o_out = talib.OBV(in_real, in_vol)
    p_out = pytafast.OBV(in_real, in_vol)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_roc_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10]:
        o_out = talib.ROC(in_real, timeperiod=period)
        p_out = pytafast.ROC(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_stoch_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    
    for matype_val in [0, 1]:  # SMA, EMA
        o_slowk, o_slowd = talib.STOCH(in_high, in_low, in_close, fastk_period=5, slowk_period=3, slowk_matype=matype_val, slowd_period=3, slowd_matype=matype_val)
        p_slowk, p_slowd = pytafast.STOCH(in_high, in_low, in_close, fastk_period=5, slowk_period=3, slowk_matype=matype_val, slowd_period=3, slowd_matype=matype_val)
        
        np.testing.assert_allclose(p_slowk, o_slowk, equal_nan=True)
        np.testing.assert_allclose(p_slowd, o_slowd, equal_nan=True)

def test_mom_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10, 14]:
        o_out = talib.MOM(in_real, timeperiod=period)
        p_out = pytafast.MOM(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_stddev_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10, 14]:
        for nbdev in [1.0, 2.0]:
            o_out = talib.STDDEV(in_real, timeperiod=period, nbdev=nbdev)
            p_out = pytafast.STDDEV(in_real, timeperiod=period, nbdev=nbdev)
            np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_willr_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.WILLR(in_high, in_low, in_close, timeperiod=period)
        p_out = pytafast.WILLR(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_natr_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.NATR(in_high, in_low, in_close, timeperiod=period)
        p_out = pytafast.NATR(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_mfi_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    in_vol = np.random.random(100) * 1000
    for period in [5, 14]:
        o_out = talib.MFI(in_high, in_low, in_close, in_vol, timeperiod=period)
        p_out = pytafast.MFI(in_high, in_low, in_close, in_vol, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_cmo_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 14]:
        o_out = talib.CMO(in_real, timeperiod=period)
        p_out = pytafast.CMO(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_dx_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.DX(in_high, in_low, in_close, timeperiod=period)
        p_out = pytafast.DX(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_minus_di_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.MINUS_DI(in_high, in_low, in_close, timeperiod=period)
        p_out = pytafast.MINUS_DI(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_minus_dm_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    for period in [5, 14]:
        o_out = talib.MINUS_DM(in_high, in_low, timeperiod=period)
        p_out = pytafast.MINUS_DM(in_high, in_low, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_plus_di_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.PLUS_DI(in_high, in_low, in_close, timeperiod=period)
        p_out = pytafast.PLUS_DI(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_plus_dm_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    for period in [5, 14]:
        o_out = talib.PLUS_DM(in_high, in_low, timeperiod=period)
        p_out = pytafast.PLUS_DM(in_high, in_low, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_apo_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    
    for matype_val in [0, 1]:
        o_out = talib.APO(in_real, fastperiod=12, slowperiod=26, matype=matype_val)
        p_out = pytafast.APO(in_real, fastperiod=12, slowperiod=26, matype=matype_val)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_aroon_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    for period in [5, 14]:
        o_down, o_up = talib.AROON(in_high, in_low, timeperiod=period)
        p_down, p_up = pytafast.AROON(in_high, in_low, timeperiod=period)
        np.testing.assert_allclose(p_down, o_down, equal_nan=True)
        np.testing.assert_allclose(p_up, o_up, equal_nan=True)

def test_aroonosc_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    for period in [5, 14]:
        o_out = talib.AROONOSC(in_high, in_low, timeperiod=period)
        p_out = pytafast.AROONOSC(in_high, in_low, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_ppo_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    
    for matype_val in [0, 1]:
        o_out = talib.PPO(in_real, fastperiod=12, slowperiod=26, matype=matype_val)
        p_out = pytafast.PPO(in_real, fastperiod=12, slowperiod=26, matype=matype_val)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_trix_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10]:
        o_out = talib.TRIX(in_real, timeperiod=period)
        p_out = pytafast.TRIX(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_ultosc_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    
    o_out = talib.ULTOSC(in_high, in_low, in_close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    p_out = pytafast.ULTOSC(in_high, in_low, in_close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

# --- Batch 1: Overlap Studies ---

def test_dema_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 14, 30]:
        o_out = talib.DEMA(in_real, timeperiod=period)
        p_out = pytafast.DEMA(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_kama_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 14, 30]:
        o_out = talib.KAMA(in_real, timeperiod=period)
        p_out = pytafast.KAMA(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_ma_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for matype_val in [0, 1, 2, 3, 4, 5, 6, 8]:  # SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3
        o_out = talib.MA(in_real, timeperiod=14, matype=matype_val)
        p_out = pytafast.MA(in_real, timeperiod=14, matype=matype_val)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_t3_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10]:
        o_out = talib.T3(in_real, timeperiod=period, vfactor=0.7)
        p_out = pytafast.T3(in_real, timeperiod=period, vfactor=0.7)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_tema_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 14, 30]:
        o_out = talib.TEMA(in_real, timeperiod=period)
        p_out = pytafast.TEMA(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_trima_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 14, 30]:
        o_out = talib.TRIMA(in_real, timeperiod=period)
        p_out = pytafast.TRIMA(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_wma_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 14, 30]:
        o_out = talib.WMA(in_real, timeperiod=period)
        p_out = pytafast.WMA(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

# --- Batch 2: SAR, TRANGE, Price Transforms ---

def test_sar_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    o_out = talib.SAR(in_high, in_low, acceleration=0.02, maximum=0.2)
    p_out = pytafast.SAR(in_high, in_low, acceleration=0.02, maximum=0.2)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_trange_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    o_out = talib.TRANGE(in_high, in_low, in_close)
    p_out = pytafast.TRANGE(in_high, in_low, in_close)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_avgprice_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_open = np.random.random(100) * 100 + 10
    in_high = in_open + np.random.random(100) * 5
    in_low = in_open - np.random.random(100) * 5
    in_close = in_open + np.random.random(100) * 2 - 1
    o_out = talib.AVGPRICE(in_open, in_high, in_low, in_close)
    p_out = pytafast.AVGPRICE(in_open, in_high, in_low, in_close)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_medprice_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    o_out = talib.MEDPRICE(in_high, in_low)
    p_out = pytafast.MEDPRICE(in_high, in_low)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_typprice_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    o_out = talib.TYPPRICE(in_high, in_low, in_close)
    p_out = pytafast.TYPPRICE(in_high, in_low, in_close)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_wclprice_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    o_out = talib.WCLPRICE(in_high, in_low, in_close)
    p_out = pytafast.WCLPRICE(in_high, in_low, in_close)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_midpoint_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 14]:
        o_out = talib.MIDPOINT(in_real, timeperiod=period)
        p_out = pytafast.MIDPOINT(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_midprice_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    for period in [5, 14]:
        o_out = talib.MIDPRICE(in_high, in_low, timeperiod=period)
        p_out = pytafast.MIDPRICE(in_high, in_low, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

# --- Batch 3: Momentum indicators ---

def test_adxr_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(200) * 100 + 10
    in_low = in_high - np.random.random(200) * 5
    in_close = in_low + (in_high - in_low) / 2
    for period in [5, 14]:
        o_out = talib.ADXR(in_high, in_low, in_close, timeperiod=period)
        p_out = pytafast.ADXR(in_high, in_low, in_close, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_bop_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_open = np.random.random(100) * 100 + 10
    in_high = in_open + np.random.random(100) * 5
    in_low = in_open - np.random.random(100) * 5
    in_close = in_open + np.random.random(100) * 2 - 1
    o_out = talib.BOP(in_open, in_high, in_low, in_close)
    p_out = pytafast.BOP(in_open, in_high, in_low, in_close)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_macdext_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    o_macd, o_signal, o_hist = talib.MACDEXT(in_real, fastperiod=12, fastmatype=0,
                                              slowperiod=26, slowmatype=0,
                                              signalperiod=9, signalmatype=0)
    p_macd, p_signal, p_hist = pytafast.MACDEXT(in_real, fastperiod=12, fastmatype=0,
                                                 slowperiod=26, slowmatype=0,
                                                 signalperiod=9, signalmatype=0)
    np.testing.assert_allclose(p_macd, o_macd, equal_nan=True)
    np.testing.assert_allclose(p_signal, o_signal, equal_nan=True)
    np.testing.assert_allclose(p_hist, o_hist, equal_nan=True)

def test_macdfix_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    o_macd, o_signal, o_hist = talib.MACDFIX(in_real, signalperiod=9)
    p_macd, p_signal, p_hist = pytafast.MACDFIX(in_real, signalperiod=9)
    np.testing.assert_allclose(p_macd, o_macd, equal_nan=True)
    np.testing.assert_allclose(p_signal, o_signal, equal_nan=True)
    np.testing.assert_allclose(p_hist, o_hist, equal_nan=True)

def test_stochf_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    o_fastk, o_fastd = talib.STOCHF(in_high, in_low, in_close,
                                      fastk_period=5, fastd_period=3, fastd_matype=0)
    p_fastk, p_fastd = pytafast.STOCHF(in_high, in_low, in_close,
                                         fastk_period=5, fastd_period=3, fastd_matype=0)
    np.testing.assert_allclose(p_fastk, o_fastk, equal_nan=True)
    np.testing.assert_allclose(p_fastd, o_fastd, equal_nan=True)

def test_stochrsi_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(200) * 100 + 10
    o_fastk, o_fastd = talib.STOCHRSI(in_real, timeperiod=14,
                                        fastk_period=5, fastd_period=3, fastd_matype=0)
    p_fastk, p_fastd = pytafast.STOCHRSI(in_real, timeperiod=14,
                                           fastk_period=5, fastd_period=3, fastd_matype=0)
    np.testing.assert_allclose(p_fastk, o_fastk, equal_nan=True)
    np.testing.assert_allclose(p_fastd, o_fastd, equal_nan=True)

def test_rocp_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10]:
        o_out = talib.ROCP(in_real, timeperiod=period)
        p_out = pytafast.ROCP(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_rocr_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10]:
        o_out = talib.ROCR(in_real, timeperiod=period)
        p_out = pytafast.ROCR(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_rocr100_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [5, 10]:
        o_out = talib.ROCR100(in_real, timeperiod=period)
        p_out = pytafast.ROCR100(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

# --- Batch 4: Volume + Statistics indicators ---

def test_ad_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    in_volume = np.random.random(100) * 1000000
    o_out = talib.AD(in_high, in_low, in_close, in_volume)
    p_out = pytafast.AD(in_high, in_low, in_close, in_volume)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_adosc_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_high = np.random.random(100) * 100 + 10
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    in_volume = np.random.random(100) * 1000000
    o_out = talib.ADOSC(in_high, in_low, in_close, in_volume, fastperiod=3, slowperiod=10)
    p_out = pytafast.ADOSC(in_high, in_low, in_close, in_volume, fastperiod=3, slowperiod=10)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_beta_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in0 = np.random.random(100) * 100 + 10
    in1 = np.random.random(100) * 100 + 10
    for period in [5, 10]:
        o_out = talib.BETA(in0, in1, timeperiod=period)
        p_out = pytafast.BETA(in0, in1, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_correl_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in0 = np.random.random(100) * 100 + 10
    in1 = np.random.random(100) * 100 + 10
    for period in [10, 30]:
        o_out = talib.CORREL(in0, in1, timeperiod=period)
        p_out = pytafast.CORREL(in0, in1, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_linearreg_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [7, 14]:
        o_out = talib.LINEARREG(in_real, timeperiod=period)
        p_out = pytafast.LINEARREG(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_linearreg_angle_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    o_out = talib.LINEARREG_ANGLE(in_real, timeperiod=14)
    p_out = pytafast.LINEARREG_ANGLE(in_real, timeperiod=14)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_linearreg_intercept_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    o_out = talib.LINEARREG_INTERCEPT(in_real, timeperiod=14)
    p_out = pytafast.LINEARREG_INTERCEPT(in_real, timeperiod=14)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_linearreg_slope_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    o_out = talib.LINEARREG_SLOPE(in_real, timeperiod=14)
    p_out = pytafast.LINEARREG_SLOPE(in_real, timeperiod=14)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_tsf_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [7, 14]:
        o_out = talib.TSF(in_real, timeperiod=period)
        p_out = pytafast.TSF(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_var_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    o_out = talib.VAR(in_real, timeperiod=5, nbdev=1.0)
    p_out = pytafast.VAR(in_real, timeperiod=5, nbdev=1.0)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_avgdev_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [7, 14]:
        o_out = talib.AVGDEV(in_real, timeperiod=period)
        p_out = pytafast.AVGDEV(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

# --- Batch 5: Math Operators + Math Transforms ---

def test_add_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in0 = np.random.random(100) * 100
    in1 = np.random.random(100) * 100
    o_out = talib.ADD(in0, in1)
    p_out = pytafast.ADD(in0, in1)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_sub_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in0 = np.random.random(100) * 100
    in1 = np.random.random(100) * 100
    o_out = talib.SUB(in0, in1)
    p_out = pytafast.SUB(in0, in1)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_mult_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in0 = np.random.random(100) * 100
    in1 = np.random.random(100) * 100
    o_out = talib.MULT(in0, in1)
    p_out = pytafast.MULT(in0, in1)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_div_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in0 = np.random.random(100) * 100
    in1 = np.random.random(100) * 100 + 1
    o_out = talib.DIV(in0, in1)
    p_out = pytafast.DIV(in0, in1)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

_MATH_TRANSFORMS = [
    "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH", "EXP", "FLOOR",
    "LN", "LOG10", "SIN", "SINH", "SQRT", "TAN", "TANH",
]

@pytest.mark.parametrize("func_name", _MATH_TRANSFORMS)
def test_math_transform_against_official_talib(func_name):
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 0.98 + 0.01
    o_out = getattr(talib, func_name)(in_real)
    p_out = getattr(pytafast, func_name)(in_real)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

# --- Batch 6: Statistics (MIN/MAX/SUM/MINMAX/MINMAXINDEX) + Cycle (HT_*) ---

def test_max_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [10, 30]:
        o_out = talib.MAX(in_real, timeperiod=period)
        p_out = pytafast.MAX(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_min_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [10, 30]:
        o_out = talib.MIN(in_real, timeperiod=period)
        p_out = pytafast.MIN(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_sum_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    for period in [10, 30]:
        o_out = talib.SUM(in_real, timeperiod=period)
        p_out = pytafast.SUM(in_real, timeperiod=period)
        np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_minmax_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    o_min, o_max = talib.MINMAX(in_real, timeperiod=30)
    p_min, p_max = pytafast.MINMAX(in_real, timeperiod=30)
    np.testing.assert_allclose(p_min, o_min, equal_nan=True)
    np.testing.assert_allclose(p_max, o_max, equal_nan=True)

def test_minmaxindex_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.random.random(100) * 100 + 10
    o_minidx, o_maxidx = talib.MINMAXINDEX(in_real, timeperiod=30)
    p_minidx, p_maxidx = pytafast.MINMAXINDEX(in_real, timeperiod=30)
    # Compare only positions after lookback (where TA-Lib outputs valid data)
    lookback = 29  # timeperiod - 1
    np.testing.assert_array_equal(np.array(p_minidx)[lookback:], np.array(o_minidx)[lookback:].astype(int))
    np.testing.assert_array_equal(np.array(p_maxidx)[lookback:], np.array(o_maxidx)[lookback:].astype(int))

def test_ht_dcperiod_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.cumsum(np.random.random(200) - 0.5) + 100
    o_out = talib.HT_DCPERIOD(in_real)
    p_out = pytafast.HT_DCPERIOD(in_real)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_ht_dcphase_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.cumsum(np.random.random(200) - 0.5) + 100
    o_out = talib.HT_DCPHASE(in_real)
    p_out = pytafast.HT_DCPHASE(in_real)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_ht_phasor_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.cumsum(np.random.random(200) - 0.5) + 100
    o_ip, o_q = talib.HT_PHASOR(in_real)
    p_ip, p_q = pytafast.HT_PHASOR(in_real)
    np.testing.assert_allclose(p_ip, o_ip, equal_nan=True)
    np.testing.assert_allclose(p_q, o_q, equal_nan=True)

def test_ht_sine_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.cumsum(np.random.random(200) - 0.5) + 100
    o_sine, o_lead = talib.HT_SINE(in_real)
    p_sine, p_lead = pytafast.HT_SINE(in_real)
    np.testing.assert_allclose(p_sine, o_sine, equal_nan=True)
    np.testing.assert_allclose(p_lead, o_lead, equal_nan=True)

def test_ht_trendline_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.cumsum(np.random.random(200) - 0.5) + 100
    o_out = talib.HT_TRENDLINE(in_real)
    p_out = pytafast.HT_TRENDLINE(in_real)
    np.testing.assert_allclose(p_out, o_out, equal_nan=True)

def test_ht_trendmode_against_official_talib():
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_real = np.cumsum(np.random.random(200) - 0.5) + 100
    o_out = talib.HT_TRENDMODE(in_real)
    p_out = pytafast.HT_TRENDMODE(in_real)
    # Compare valid positions (after lookback)
    valid = ~np.isnan(o_out.astype(float))
    np.testing.assert_array_equal(np.array(p_out)[valid], np.array(o_out)[valid].astype(int))

# --- Batch 7: Candlestick Patterns ---

_CDL_STANDARD_LIST = [
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

_CDL_PEN_LIST = [
    "CDLABANDONEDBABY", "CDLDARKCLOUDCOVER", "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR", "CDLMATHOLD", "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR",
]

@pytest.mark.parametrize("func_name", _CDL_STANDARD_LIST)
def test_cdl_standard_against_official_talib(func_name):
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_open = np.random.random(100) * 50 + 50
    in_high = in_open + np.random.random(100) * 5
    in_low = in_open - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) * np.random.random(100)
    o_out = getattr(talib, func_name)(in_open, in_high, in_low, in_close)
    p_out = getattr(pytafast, func_name)(in_open, in_high, in_low, in_close)
    valid = ~np.isnan(o_out.astype(float))
    np.testing.assert_array_equal(np.array(p_out)[valid], np.array(o_out)[valid].astype(int))

@pytest.mark.parametrize("func_name", _CDL_PEN_LIST)
def test_cdl_penetration_against_official_talib(func_name):
    talib = pytest.importorskip("talib")
    np.random.seed(42)
    in_open = np.random.random(100) * 50 + 50
    in_high = in_open + np.random.random(100) * 5
    in_low = in_open - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) * np.random.random(100)
    o_out = getattr(talib, func_name)(in_open, in_high, in_low, in_close)
    p_out = getattr(pytafast, func_name)(in_open, in_high, in_low, in_close)
    valid = ~np.isnan(o_out.astype(float))
    np.testing.assert_array_equal(np.array(p_out)[valid], np.array(o_out)[valid].astype(int))
