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


# ===== Helpers: generate paired benchmarks (pytafast vs talib) ==============

def _make_single_input_benchmarks(indicator_name, **kwargs):
    """Generate benchmark pair for single-array indicators like SMA, EMA."""

    def test_pytafast_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(pytafast, indicator_name)
        benchmark(func, _close, **kwargs)

    def test_talib_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(talib, indicator_name)
        benchmark(func, _close, **kwargs)

    def test_pytafast_pandas(benchmark):
        benchmark.group = f"{indicator_name}_pandas"
        func = getattr(pytafast, indicator_name)
        benchmark(func, _close_s, **kwargs)

    def test_talib_pandas(benchmark):
        benchmark.group = f"{indicator_name}_pandas"
        func = getattr(talib, indicator_name)
        benchmark(func, _close_s, **kwargs)

    return test_pytafast_numpy, test_talib_numpy, test_pytafast_pandas, test_talib_pandas

def _make_hlc_benchmarks(indicator_name, **kwargs):
    """HLC input indicators."""
    def test_pytafast_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(pytafast, indicator_name)
        benchmark(func, _high, _low, _close, **kwargs)

    def test_talib_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(talib, indicator_name)
        benchmark(func, _high, _low, _close, **kwargs)
    return test_pytafast_numpy, test_talib_numpy

def _make_hl_benchmarks(indicator_name, **kwargs):
    """HL input indicators."""
    def test_pytafast_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(pytafast, indicator_name)
        benchmark(func, _high, _low, **kwargs)

    def test_talib_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(talib, indicator_name)
        benchmark(func, _high, _low, **kwargs)
    return test_pytafast_numpy, test_talib_numpy

def _make_ohlc_benchmarks(indicator_name, **kwargs):
    """OHLC input indicators (mainly Candlesticks)."""
    def test_pytafast_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(pytafast, indicator_name)
        benchmark(func, _open, _high, _low, _close, **kwargs)

    def test_talib_numpy(benchmark):
        benchmark.group = indicator_name
        func = getattr(talib, indicator_name)
        benchmark(func, _open, _high, _low, _close, **kwargs)
    return test_pytafast_numpy, test_talib_numpy


# ======================== Overlap Studies ==================================

# Single input Overlap
for name in ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3"]:
    p_np, t_numpy, p_pd, t_pd = _make_single_input_benchmarks(name)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy
    globals()[f"test_benchmark_pytafast_{name.lower()}_pandas"] = p_pd
    globals()[f"test_benchmark_talib_{name.lower()}_pandas"] = t_pd

# BBANDS
(test_benchmark_pytafast_bbands_numpy,
 test_benchmark_talib_bbands_numpy,
 test_benchmark_pytafast_bbands_pandas,
 test_benchmark_talib_bbands_pandas) = _make_single_input_benchmarks("BBANDS", timeperiod=TIME_PERIOD)

# SAR
(test_benchmark_pytafast_sar_numpy,
 test_benchmark_talib_sar_numpy) = _make_hl_benchmarks("SAR")

# ======================== Momentum =========================================

# Single input Momentum
for name in ["RSI", "MOM", "ROC", "ROCP", "ROCR", "TRIX", "CMO"]:
    p_np, t_numpy, p_pd, t_pd = _make_single_input_benchmarks(name)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy
    globals()[f"test_benchmark_pytafast_{name.lower()}_pandas"] = p_pd
    globals()[f"test_benchmark_talib_{name.lower()}_pandas"] = t_pd

# HLC Momentum
for name in ["ADX", "ADXR", "CCI", "DX", "MINUS_DI", "PLUS_DI", "WILLR"]:
    p_np, t_numpy = _make_hlc_benchmarks(name)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy

# MACD
def test_benchmark_pytafast_macd_numpy(benchmark):
    benchmark.group = "MACD"
    benchmark(pytafast.MACD, _close)
def test_benchmark_talib_macd_numpy(benchmark):
    benchmark.group = "MACD"
    benchmark(talib.MACD, _close)

# STOCH, STOCHF
(test_benchmark_pytafast_stoch_numpy,
 test_benchmark_talib_stoch_numpy) = _make_hlc_benchmarks("STOCH")
(test_benchmark_pytafast_stochf_numpy,
 test_benchmark_talib_stochf_numpy) = _make_hlc_benchmarks("STOCHF")

# MFI (HLCV)
def test_benchmark_pytafast_mfi_numpy(benchmark):
    benchmark.group = "MFI"
    benchmark(pytafast.MFI, _high, _low, _close, _volume)
def test_benchmark_talib_mfi_numpy(benchmark):
    benchmark.group = "MFI"
    benchmark(talib.MFI, _high, _low, _close, _volume)

# ULTOSC
(test_benchmark_pytafast_ultosc_numpy,
 test_benchmark_talib_ultosc_numpy) = _make_hlc_benchmarks("ULTOSC")

# BOP (OHLC)
(test_benchmark_pytafast_bop_numpy,
 test_benchmark_talib_bop_numpy) = _make_ohlc_benchmarks("BOP")

# ======================== Volatility =======================================

(test_benchmark_pytafast_atr_numpy,
 test_benchmark_talib_atr_numpy) = _make_hlc_benchmarks("ATR")
(test_benchmark_pytafast_natr_numpy,
 test_benchmark_talib_natr_numpy) = _make_hlc_benchmarks("NATR")
(test_benchmark_pytafast_trange_numpy,
 test_benchmark_talib_trange_numpy) = _make_hlc_benchmarks("TRANGE")

# ======================== Volume ===========================================

def test_benchmark_pytafast_obv_numpy(benchmark):
    benchmark.group = "OBV"
    benchmark(pytafast.OBV, _close, _volume)
def test_benchmark_talib_obv_numpy(benchmark):
    benchmark.group = "OBV"
    benchmark(talib.OBV, _close, _volume)

def test_benchmark_pytafast_ad_numpy(benchmark):
    benchmark.group = "AD"
    benchmark(pytafast.AD, _high, _low, _close, _volume)
def test_benchmark_talib_ad_numpy(benchmark):
    benchmark.group = "AD"
    benchmark(talib.AD, _high, _low, _close, _volume)

# ======================== Price Transform ==================================

(test_benchmark_pytafast_avgprice_numpy,
 test_benchmark_talib_avgprice_numpy) = _make_ohlc_benchmarks("AVGPRICE")
(test_benchmark_pytafast_medprice_numpy,
 test_benchmark_talib_medprice_numpy) = _make_hl_benchmarks("MEDPRICE") 
(test_benchmark_pytafast_typprice_numpy,
 test_benchmark_talib_typprice_numpy) = _make_hlc_benchmarks("TYPPRICE")

# ======================== Statistics =======================================

for name in ["STDDEV", "VAR", "TSF", "LINEARREG", "LINEARREG_SLOPE", "AVGDEV"]:
    p_np, t_numpy, p_pd, t_pd = _make_single_input_benchmarks(name)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy

# BETA, CORREL
def test_benchmark_pytafast_beta_numpy(benchmark):
    benchmark.group = "BETA"
    benchmark(pytafast.BETA, _close, _close2)
def test_benchmark_talib_beta_numpy(benchmark):
    benchmark.group = "BETA"
    benchmark(talib.BETA, _close, _close2)

# ======================== Math Operators ===================================

def test_benchmark_pytafast_add_numpy(benchmark):
    benchmark.group = "ADD"
    benchmark(pytafast.ADD, _close, _close2)
def test_benchmark_talib_add_numpy(benchmark):
    benchmark.group = "ADD"
    benchmark(talib.ADD, _close, _close2)

# ======================== Math Transforms ==================================

for name in ["SIN", "COS", "SQRT", "LN", "EXP"]:
    p_np, t_numpy, _, _ = _make_single_input_benchmarks(name)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy

# ======================== Cycle Indicators =================================

for name in ["HT_DCPERIOD", "HT_DCPHASE", "HT_TRENDLINE", "HT_TRENDMODE"]:
    p_np, t_numpy, _, _ = _make_single_input_benchmarks(name)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy

def test_benchmark_pytafast_ht_phasor_numpy(benchmark):
    benchmark.group = "HT_PHASOR"
    benchmark(pytafast.HT_PHASOR, _close)
def test_benchmark_talib_ht_phasor_numpy(benchmark):
    benchmark.group = "HT_PHASOR"
    benchmark(talib.HT_PHASOR, _close)

from concurrent.futures import ThreadPoolExecutor

# ======================== Candlestick Patterns =============================

for name in ["CDLENGULFING", "CDLDOJI", "CDLHAMMER", "CDLHARAMI", "CDLMARUBOZU"]:
    p_np, t_numpy = _make_ohlc_benchmarks(name)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy

# ======================== Concurrency Benchmarks ===========================
# Benchmarking multi-threaded performance (GIL release effectiveness)

CONCURRENT_TASKS = 100
MAX_WORKERS = 4

def _run_concurrent(func, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(func, *args, **kwargs) for _ in range(CONCURRENT_TASKS)]
        for f in futures:
            f.result()

def test_benchmark_pytafast_concurrency_sma(benchmark):
    benchmark.group = "Concurrency_SMA"
    benchmark(_run_concurrent, pytafast.SMA, _close, timeperiod=30)

def test_benchmark_talib_concurrency_sma(benchmark):
    benchmark.group = "Concurrency_SMA"
    benchmark(_run_concurrent, talib.SMA, _close, timeperiod=30)

def test_benchmark_pytafast_concurrency_rsi(benchmark):
    benchmark.group = "Concurrency_RSI"
    benchmark(_run_concurrent, pytafast.RSI, _close, timeperiod=14)

def test_benchmark_talib_concurrency_rsi(benchmark):
    benchmark.group = "Concurrency_RSI"
    benchmark(_run_concurrent, talib.RSI, _close, timeperiod=14)

def test_benchmark_pytafast_concurrency_macd(benchmark):
    benchmark.group = "Concurrency_MACD"
    benchmark(_run_concurrent, pytafast.MACD, _close)

def test_benchmark_talib_concurrency_macd(benchmark):
    benchmark.group = "Concurrency_MACD"
    benchmark(_run_concurrent, talib.MACD, _close)
