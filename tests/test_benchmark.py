import pytest
import numpy as np
import pytafast
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

try:
    import talib
except ImportError:
    pytest.skip("talib not available", allow_module_level=True)

# ======================== Benchmarks ===================================

OVERLAP_FUNCS = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MIDPOINT"]

@pytest.mark.parametrize("name", OVERLAP_FUNCS)
def test_benchmark_overlap_pytafast(benchmark, name, benchmark_data):
    func = getattr(pytafast, name)
    benchmark.group = name
    benchmark(func, benchmark_data["close"], timeperiod=30)

@pytest.mark.parametrize("name", OVERLAP_FUNCS)
def test_benchmark_overlap_talib(benchmark, name, benchmark_data):
    func = getattr(talib, name)
    benchmark.group = name
    benchmark(func, benchmark_data["close"], timeperiod=30)


MOMENTUM_FUNCS = ["RSI", "MOM", "ROC", "ROCP", "ROCR", "ROCR100", "TRIX"]

@pytest.mark.parametrize("name", MOMENTUM_FUNCS)
def test_benchmark_momentum_pytafast(benchmark, name, benchmark_data):
    func = getattr(pytafast, name)
    benchmark.group = name
    benchmark(func, benchmark_data["close"], timeperiod=14)

@pytest.mark.parametrize("name", MOMENTUM_FUNCS)
def test_benchmark_momentum_talib(benchmark, name, benchmark_data):
    func = getattr(talib, name)
    benchmark.group = name
    benchmark(func, benchmark_data["close"], timeperiod=14)


def test_benchmark_pytafast_macd_numpy(benchmark, benchmark_data):
    benchmark.group = "MACD"
    benchmark(pytafast.MACD, benchmark_data["close"])

def test_benchmark_talib_macd_numpy(benchmark, benchmark_data):
    benchmark.group = "MACD"
    benchmark(talib.MACD, benchmark_data["close"])

def test_benchmark_pytafast_atr_numpy(benchmark, benchmark_data):
    benchmark.group = "ATR"
    benchmark(pytafast.ATR, benchmark_data["high"], benchmark_data["low"], benchmark_data["close"], timeperiod=14)

def test_benchmark_talib_atr_numpy(benchmark, benchmark_data):
    benchmark.group = "ATR"
    benchmark(talib.ATR, benchmark_data["high"], benchmark_data["low"], benchmark_data["close"], timeperiod=14)

def test_benchmark_pytafast_obv_numpy(benchmark, benchmark_data):
    benchmark.group = "OBV"
    benchmark(pytafast.OBV, benchmark_data["close"], benchmark_data["volume"])

def test_benchmark_talib_obv_numpy(benchmark, benchmark_data):
    benchmark.group = "OBV"
    benchmark(talib.OBV, benchmark_data["close"], benchmark_data["volume"])

STAT_FUNCS = ["LINEARREG", "LINEARREG_SLOPE", "TSF"]

@pytest.mark.parametrize("name", STAT_FUNCS)
def test_benchmark_stat_pytafast(benchmark, name, benchmark_data):
    func = getattr(pytafast, name)
    benchmark.group = name
    benchmark(func, benchmark_data["close"], timeperiod=14)

@pytest.mark.parametrize("name", STAT_FUNCS)
def test_benchmark_stat_talib(benchmark, name, benchmark_data):
    func = getattr(talib, name)
    benchmark.group = name
    benchmark(func, benchmark_data["close"], timeperiod=14)


MATH_FUNCS = ["SIN", "COS", "TAN", "SQRT", "LN", "LOG10"]

@pytest.mark.parametrize("name", MATH_FUNCS)
def test_benchmark_math_pytafast(benchmark, name, benchmark_data):
    func = getattr(pytafast, name)
    benchmark.group = name
    benchmark(func, benchmark_data["close"])

@pytest.mark.parametrize("name", MATH_FUNCS)
def test_benchmark_math_talib(benchmark, name, benchmark_data):
    func = getattr(talib, name)
    benchmark.group = name
    benchmark(func, benchmark_data["close"])

def test_benchmark_pytafast_ht_phasor_numpy(benchmark, benchmark_data):
    benchmark.group = "HT_PHASOR"
    benchmark(pytafast.HT_PHASOR, benchmark_data["close"])

def test_benchmark_talib_ht_phasor_numpy(benchmark, benchmark_data):
    benchmark.group = "HT_PHASOR"
    benchmark(talib.HT_PHASOR, benchmark_data["close"])

CDL_FUNCS = ["CDLENGULFING", "CDLDOJI", "CDLHAMMER", "CDLHARAMI", "CDLMARUBOZU"]

@pytest.mark.parametrize("name", CDL_FUNCS)
def test_benchmark_cdl_pytafast(benchmark, name, benchmark_data):
    func = getattr(pytafast, name)
    benchmark.group = name
    benchmark(func, benchmark_data["open"], benchmark_data["high"], benchmark_data["low"], benchmark_data["close"])

@pytest.mark.parametrize("name", CDL_FUNCS)
def test_benchmark_cdl_talib(benchmark, name, benchmark_data):
    func = getattr(talib, name)
    benchmark.group = name
    benchmark(func, benchmark_data["open"], benchmark_data["high"], benchmark_data["low"], benchmark_data["close"])

# ======================== Concurrency Benchmarks ===========================

CONCURRENT_TASKS = 100
MAX_WORKERS = 4

def _run_concurrent(func, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(func, *args, **kwargs) for _ in range(CONCURRENT_TASKS)]
        for f in futures:
            f.result()

def test_benchmark_pytafast_concurrency_sma(benchmark, benchmark_data):
    benchmark.group = "Concurrency_SMA"
    benchmark(_run_concurrent, pytafast.SMA, benchmark_data["close"], timeperiod=30)

def test_benchmark_talib_concurrency_sma(benchmark, benchmark_data):
    benchmark.group = "Concurrency_SMA"
    benchmark(_run_concurrent, talib.SMA, benchmark_data["close"], timeperiod=30)

def test_benchmark_pytafast_concurrency_rsi(benchmark, benchmark_data):
    benchmark.group = "Concurrency_RSI"
    benchmark(_run_concurrent, pytafast.RSI, benchmark_data["close"], timeperiod=14)

def test_benchmark_talib_concurrency_rsi(benchmark, benchmark_data):
    benchmark.group = "Concurrency_RSI"
    benchmark(_run_concurrent, talib.RSI, benchmark_data["close"], timeperiod=14)

def test_benchmark_pytafast_concurrency_macd(benchmark, benchmark_data):
    benchmark.group = "Concurrency_MACD"
    benchmark(_run_concurrent, pytafast.MACD, benchmark_data["close"])

def test_benchmark_talib_concurrency_macd(benchmark, benchmark_data):
    benchmark.group = "Concurrency_MACD"
    benchmark(_run_concurrent, talib.MACD, benchmark_data["close"])
