"""
Comprehensive performance benchmark: pytafast vs official ta-lib-python.

Run with:
    uv run pytest tests/test_benchmark.py -v --benchmark-columns=mean,stddev,rounds
    uv run pytest tests/test_benchmark.py --benchmark-json=.benchmarks/results.json

Skip benchmarks (only verify no errors):
    uv run pytest tests/test_benchmark.py --benchmark-disable
"""

import pytest
import numpy as np
import pandas as pd
import pytafast

talib = pytest.importorskip("talib")

# ---------------------------------------------------------------------------
# Test data: 100k random OHLCV, deterministic seed
# ---------------------------------------------------------------------------
np.random.seed(42)
N = 100_000
TIME_PERIOD = 30

_close = np.random.random(N) * 100 + 50
_high = _close + np.random.random(N) * 5
_low = _close - np.random.random(N) * 5
_open = _close + (np.random.random(N) - 0.5) * 3
_volume = np.random.random(N) * 1_000_000 + 100_000

# Pandas versions
_close_s = pd.Series(_close, name="close")
_high_s = pd.Series(_high, name="high")
_low_s = pd.Series(_low, name="low")
_open_s = pd.Series(_open, name="open")
_volume_s = pd.Series(_volume, name="volume")

# A second array for dual-input indicators (BETA, CORREL, ADD, etc.)
_close2 = np.random.random(N) * 100 + 50
_close2_s = pd.Series(_close2, name="close2")

# Math transform input (values in [0, 1] for ACOS/ASIN)
_unit = np.random.random(N)
_unit_s = pd.Series(_unit, name="unit")


# ===== Helper: generate paired benchmarks (pytafast vs talib) ==============

def _make_single_input_benchmarks(indicator_name, timeperiod=TIME_PERIOD):
    """Generate benchmark pair for single-array indicators like SMA, EMA."""

    def test_pytafast_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(pytafast, indicator_name)
        benchmark(func, _close, timeperiod=timeperiod)

    def test_talib_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(talib, indicator_name)
        benchmark(func, _close, timeperiod=timeperiod)

    def test_pytafast_pandas(benchmark):
        benchmark.group = f"{indicator_name}_pandas"
        func = getattr(pytafast, indicator_name)
        benchmark(func, _close_s, timeperiod=timeperiod)

    def test_talib_pandas(benchmark):
        benchmark.group = f"{indicator_name}_pandas"
        func = getattr(talib, indicator_name)
        benchmark(func, _close_s, timeperiod=timeperiod)

    return test_pytafast_numpy, test_talib_numpy, test_pytafast_pandas, test_talib_pandas


# ======================== Overlap Studies ==================================

# --- SMA ---
(test_benchmark_pytafast_sma_numpy,
 test_benchmark_talib_sma_numpy,
 test_benchmark_pytafast_sma_pandas,
 test_benchmark_talib_sma_pandas) = _make_single_input_benchmarks("SMA")

# --- EMA ---
(test_benchmark_pytafast_ema_numpy,
 test_benchmark_talib_ema_numpy,
 test_benchmark_pytafast_ema_pandas,
 test_benchmark_talib_ema_pandas) = _make_single_input_benchmarks("EMA")

# --- WMA ---
(test_benchmark_pytafast_wma_numpy,
 test_benchmark_talib_wma_numpy,
 test_benchmark_pytafast_wma_pandas,
 test_benchmark_talib_wma_pandas) = _make_single_input_benchmarks("WMA")

# --- DEMA ---
(test_benchmark_pytafast_dema_numpy,
 test_benchmark_talib_dema_numpy,
 test_benchmark_pytafast_dema_pandas,
 test_benchmark_talib_dema_pandas) = _make_single_input_benchmarks("DEMA")

# --- KAMA ---
(test_benchmark_pytafast_kama_numpy,
 test_benchmark_talib_kama_numpy,
 test_benchmark_pytafast_kama_pandas,
 test_benchmark_talib_kama_pandas) = _make_single_input_benchmarks("KAMA")

# --- BBANDS ---
def test_benchmark_pytafast_bbands_numpy(benchmark):
    benchmark.group = "BBANDS"
    benchmark(pytafast.BBANDS, _close, timeperiod=TIME_PERIOD)

def test_benchmark_talib_bbands_numpy(benchmark):
    benchmark.group = "BBANDS"
    benchmark(talib.BBANDS, _close, timeperiod=TIME_PERIOD)

def test_benchmark_pytafast_bbands_pandas(benchmark):
    benchmark.group = "BBANDS_pandas"
    benchmark(pytafast.BBANDS, _close_s, timeperiod=TIME_PERIOD)

def test_benchmark_talib_bbands_pandas(benchmark):
    benchmark.group = "BBANDS_pandas"
    benchmark(talib.BBANDS, _close_s, timeperiod=TIME_PERIOD)


# ======================== Momentum =========================================

# --- RSI ---
(test_benchmark_pytafast_rsi_numpy,
 test_benchmark_talib_rsi_numpy,
 test_benchmark_pytafast_rsi_pandas,
 test_benchmark_talib_rsi_pandas) = _make_single_input_benchmarks("RSI", timeperiod=14)

# --- MOM ---
(test_benchmark_pytafast_mom_numpy,
 test_benchmark_talib_mom_numpy,
 test_benchmark_pytafast_mom_pandas,
 test_benchmark_talib_mom_pandas) = _make_single_input_benchmarks("MOM", timeperiod=10)

# --- ROC ---
(test_benchmark_pytafast_roc_numpy,
 test_benchmark_talib_roc_numpy,
 test_benchmark_pytafast_roc_pandas,
 test_benchmark_talib_roc_pandas) = _make_single_input_benchmarks("ROC", timeperiod=10)

# --- CMO ---
(test_benchmark_pytafast_cmo_numpy,
 test_benchmark_talib_cmo_numpy,
 test_benchmark_pytafast_cmo_pandas,
 test_benchmark_talib_cmo_pandas) = _make_single_input_benchmarks("CMO", timeperiod=14)

# --- MACD ---
def test_benchmark_pytafast_macd_numpy(benchmark):
    benchmark.group = "MACD"
    benchmark(pytafast.MACD, _close)

def test_benchmark_talib_macd_numpy(benchmark):
    benchmark.group = "MACD"
    benchmark(talib.MACD, _close)

def test_benchmark_pytafast_macd_pandas(benchmark):
    benchmark.group = "MACD_pandas"
    benchmark(pytafast.MACD, _close_s)

def test_benchmark_talib_macd_pandas(benchmark):
    benchmark.group = "MACD_pandas"
    benchmark(talib.MACD, _close_s)

# --- STOCH ---
def test_benchmark_pytafast_stoch_numpy(benchmark):
    benchmark.group = "STOCH"
    benchmark(pytafast.STOCH, _high, _low, _close)

def test_benchmark_talib_stoch_numpy(benchmark):
    benchmark.group = "STOCH"
    benchmark(talib.STOCH, _high, _low, _close)

# --- ADX ---
def test_benchmark_pytafast_adx_numpy(benchmark):
    benchmark.group = "ADX"
    benchmark(pytafast.ADX, _high, _low, _close, timeperiod=14)

def test_benchmark_talib_adx_numpy(benchmark):
    benchmark.group = "ADX"
    benchmark(talib.ADX, _high, _low, _close, timeperiod=14)

# --- CCI ---
def test_benchmark_pytafast_cci_numpy(benchmark):
    benchmark.group = "CCI"
    benchmark(pytafast.CCI, _high, _low, _close, timeperiod=14)

def test_benchmark_talib_cci_numpy(benchmark):
    benchmark.group = "CCI"
    benchmark(talib.CCI, _high, _low, _close, timeperiod=14)

# --- WILLR ---
def test_benchmark_pytafast_willr_numpy(benchmark):
    benchmark.group = "WILLR"
    benchmark(pytafast.WILLR, _high, _low, _close, timeperiod=14)

def test_benchmark_talib_willr_numpy(benchmark):
    benchmark.group = "WILLR"
    benchmark(talib.WILLR, _high, _low, _close, timeperiod=14)


# ======================== Volatility =======================================

# --- ATR ---
def test_benchmark_pytafast_atr_numpy(benchmark):
    benchmark.group = "ATR"
    benchmark(pytafast.ATR, _high, _low, _close, timeperiod=14)

def test_benchmark_talib_atr_numpy(benchmark):
    benchmark.group = "ATR"
    benchmark(talib.ATR, _high, _low, _close, timeperiod=14)

# --- NATR ---
def test_benchmark_pytafast_natr_numpy(benchmark):
    benchmark.group = "NATR"
    benchmark(pytafast.NATR, _high, _low, _close, timeperiod=14)

def test_benchmark_talib_natr_numpy(benchmark):
    benchmark.group = "NATR"
    benchmark(talib.NATR, _high, _low, _close, timeperiod=14)

# --- TRANGE ---
def test_benchmark_pytafast_trange_numpy(benchmark):
    benchmark.group = "TRANGE"
    benchmark(pytafast.TRANGE, _high, _low, _close)

def test_benchmark_talib_trange_numpy(benchmark):
    benchmark.group = "TRANGE"
    benchmark(talib.TRANGE, _high, _low, _close)


# ======================== Volume ===========================================

# --- OBV ---
def test_benchmark_pytafast_obv_numpy(benchmark):
    benchmark.group = "OBV"
    benchmark(pytafast.OBV, _close, _volume)

def test_benchmark_talib_obv_numpy(benchmark):
    benchmark.group = "OBV"
    benchmark(talib.OBV, _close, _volume)

# --- AD ---
def test_benchmark_pytafast_ad_numpy(benchmark):
    benchmark.group = "AD"
    benchmark(pytafast.AD, _high, _low, _close, _volume)

def test_benchmark_talib_ad_numpy(benchmark):
    benchmark.group = "AD"
    benchmark(talib.AD, _high, _low, _close, _volume)

# --- ADOSC ---
def test_benchmark_pytafast_adosc_numpy(benchmark):
    benchmark.group = "ADOSC"
    benchmark(pytafast.ADOSC, _high, _low, _close, _volume)

def test_benchmark_talib_adosc_numpy(benchmark):
    benchmark.group = "ADOSC"
    benchmark(talib.ADOSC, _high, _low, _close, _volume)


# ======================== Price Transform ==================================

# --- AVGPRICE ---
def test_benchmark_pytafast_avgprice_numpy(benchmark):
    benchmark.group = "AVGPRICE"
    benchmark(pytafast.AVGPRICE, _open, _high, _low, _close)

def test_benchmark_talib_avgprice_numpy(benchmark):
    benchmark.group = "AVGPRICE"
    benchmark(talib.AVGPRICE, _open, _high, _low, _close)

# --- MEDPRICE ---
def test_benchmark_pytafast_medprice_numpy(benchmark):
    benchmark.group = "MEDPRICE"
    benchmark(pytafast.MEDPRICE, _high, _low)

def test_benchmark_talib_medprice_numpy(benchmark):
    benchmark.group = "MEDPRICE"
    benchmark(talib.MEDPRICE, _high, _low)

# --- TYPPRICE ---
def test_benchmark_pytafast_typprice_numpy(benchmark):
    benchmark.group = "TYPPRICE"
    benchmark(pytafast.TYPPRICE, _high, _low, _close)

def test_benchmark_talib_typprice_numpy(benchmark):
    benchmark.group = "TYPPRICE"
    benchmark(talib.TYPPRICE, _high, _low, _close)


# ======================== Statistics =======================================

# --- STDDEV ---
def test_benchmark_pytafast_stddev_numpy(benchmark):
    benchmark.group = "STDDEV"
    benchmark(pytafast.STDDEV, _close, timeperiod=TIME_PERIOD)

def test_benchmark_talib_stddev_numpy(benchmark):
    benchmark.group = "STDDEV"
    benchmark(talib.STDDEV, _close, timeperiod=TIME_PERIOD)

# --- LINEARREG ---
(test_benchmark_pytafast_linearreg_numpy,
 test_benchmark_talib_linearreg_numpy,
 test_benchmark_pytafast_linearreg_pandas,
 test_benchmark_talib_linearreg_pandas) = _make_single_input_benchmarks("LINEARREG", timeperiod=14)

# --- BETA ---
def test_benchmark_pytafast_beta_numpy(benchmark):
    benchmark.group = "BETA"
    benchmark(pytafast.BETA, _close, _close2, timeperiod=5)

def test_benchmark_talib_beta_numpy(benchmark):
    benchmark.group = "BETA"
    benchmark(talib.BETA, _close, _close2, timeperiod=5)

# --- CORREL ---
def test_benchmark_pytafast_correl_numpy(benchmark):
    benchmark.group = "CORREL"
    benchmark(pytafast.CORREL, _close, _close2, timeperiod=TIME_PERIOD)

def test_benchmark_talib_correl_numpy(benchmark):
    benchmark.group = "CORREL"
    benchmark(talib.CORREL, _close, _close2, timeperiod=TIME_PERIOD)


# ======================== Math Operators ===================================

# --- ADD ---
def test_benchmark_pytafast_add_numpy(benchmark):
    benchmark.group = "ADD"
    benchmark(pytafast.ADD, _close, _close2)

def test_benchmark_talib_add_numpy(benchmark):
    benchmark.group = "ADD"
    benchmark(talib.ADD, _close, _close2)

# --- MULT ---
def test_benchmark_pytafast_mult_numpy(benchmark):
    benchmark.group = "MULT"
    benchmark(pytafast.MULT, _close, _close2)

def test_benchmark_talib_mult_numpy(benchmark):
    benchmark.group = "MULT"
    benchmark(talib.MULT, _close, _close2)


# ======================== Math Transforms ==================================

# --- SIN ---
def test_benchmark_pytafast_sin_numpy(benchmark):
    benchmark.group = "SIN"
    benchmark(pytafast.SIN, _close)

def test_benchmark_talib_sin_numpy(benchmark):
    benchmark.group = "SIN"
    benchmark(talib.SIN, _close)

# --- EXP ---
def test_benchmark_pytafast_exp_numpy(benchmark):
    benchmark.group = "EXP"
    benchmark(pytafast.EXP, _unit)  # small values to avoid overflow

def test_benchmark_talib_exp_numpy(benchmark):
    benchmark.group = "EXP"
    benchmark(talib.EXP, _unit)

# --- SQRT ---
def test_benchmark_pytafast_sqrt_numpy(benchmark):
    benchmark.group = "SQRT"
    benchmark(pytafast.SQRT, _close)

def test_benchmark_talib_sqrt_numpy(benchmark):
    benchmark.group = "SQRT"
    benchmark(talib.SQRT, _close)


# ======================== Cycle Indicators =================================

# --- HT_DCPERIOD ---
def test_benchmark_pytafast_ht_dcperiod_numpy(benchmark):
    benchmark.group = "HT_DCPERIOD"
    benchmark(pytafast.HT_DCPERIOD, _close)

def test_benchmark_talib_ht_dcperiod_numpy(benchmark):
    benchmark.group = "HT_DCPERIOD"
    benchmark(talib.HT_DCPERIOD, _close)

# --- HT_TRENDLINE ---
def test_benchmark_pytafast_ht_trendline_numpy(benchmark):
    benchmark.group = "HT_TRENDLINE"
    benchmark(pytafast.HT_TRENDLINE, _close)

def test_benchmark_talib_ht_trendline_numpy(benchmark):
    benchmark.group = "HT_TRENDLINE"
    benchmark(talib.HT_TRENDLINE, _close)


# ======================== Candlestick Patterns =============================

# --- CDLENGULFING ---
def test_benchmark_pytafast_cdlengulfing_numpy(benchmark):
    benchmark.group = "CDLENGULFING"
    benchmark(pytafast.CDLENGULFING, _open, _high, _low, _close)

def test_benchmark_talib_cdlengulfing_numpy(benchmark):
    benchmark.group = "CDLENGULFING"
    benchmark(talib.CDLENGULFING, _open, _high, _low, _close)

# --- CDLDOJI ---
def test_benchmark_pytafast_cdldoji_numpy(benchmark):
    benchmark.group = "CDLDOJI"
    benchmark(pytafast.CDLDOJI, _open, _high, _low, _close)

def test_benchmark_talib_cdldoji_numpy(benchmark):
    benchmark.group = "CDLDOJI"
    benchmark(talib.CDLDOJI, _open, _high, _low, _close)

# --- CDL3WHITESOLDIERS ---
def test_benchmark_pytafast_cdl3whitesoldiers_numpy(benchmark):
    benchmark.group = "CDL3WHITESOLDIERS"
    benchmark(pytafast.CDL3WHITESOLDIERS, _open, _high, _low, _close)

def test_benchmark_talib_cdl3whitesoldiers_numpy(benchmark):
    benchmark.group = "CDL3WHITESOLDIERS"
    benchmark(talib.CDL3WHITESOLDIERS, _open, _high, _low, _close)
