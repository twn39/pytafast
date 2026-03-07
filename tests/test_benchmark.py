import numpy as np
import talib
import pytafast
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# ======================== Setup Data ========================================

# Use a realistic data size (e.g., 5000 points)
N = 5000
np.random.seed(42)
_close = np.random.random(N)
_high = _close + np.random.random(N) * 0.05
_low = _close - np.random.random(N) * 0.05
_open = _low + np.random.random(N) * (_high - _low)
_volume = np.random.random(N) * 1000000

# Pandas variants
_close_pd = pd.Series(_close)
_high_pd = pd.Series(_high)
_low_pd = pd.Series(_low)
_open_pd = pd.Series(_open)
_volume_pd = pd.Series(_volume)

# ======================== Factory Helpers ===================================


def _make_single_benchmarks(name, **kwargs):
    """Generates numpy vs talib benchmarks for a single-input indicator."""
    p_func = getattr(pytafast, name)
    t_func = getattr(talib, name)

    def p_np(benchmark):
        benchmark.group = name
        benchmark(p_func, _close, **kwargs)

    def t_numpy(benchmark):
        benchmark.group = name
        benchmark(t_func, _close, **kwargs)

    return p_np, t_numpy


def _make_ohlc_benchmarks(name, **kwargs):
    """Generates benchmarks for OHLC indicators."""
    p_func = getattr(pytafast, name)
    t_func = getattr(talib, name)

    def p_np(benchmark):
        benchmark.group = name
        benchmark(p_func, _open, _high, _low, _close, **kwargs)

    def t_numpy(benchmark):
        benchmark.group = name
        benchmark(t_func, _open, _high, _low, _close, **kwargs)

    return p_np, t_numpy


# ======================== Overlap Studies ===================================

for name in ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MIDPOINT"]:
    p_np, t_numpy = _make_single_benchmarks(name, timeperiod=30)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy

# ======================== Momentum Indicators ===============================

for name in ["RSI", "MOM", "ROC", "ROCP", "ROCR", "ROCR100", "TRIX"]:
    p_np, t_numpy = _make_single_benchmarks(name, timeperiod=14)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy


def test_benchmark_pytafast_macd_numpy(benchmark):
    benchmark.group = "MACD"
    benchmark(pytafast.MACD, _close)


def test_benchmark_talib_macd_numpy(benchmark):
    benchmark.group = "MACD"
    benchmark(talib.MACD, _close)


# ======================== Volatility ========================================


def test_benchmark_pytafast_atr_numpy(benchmark):
    benchmark.group = "ATR"
    benchmark(pytafast.ATR, _high, _low, _close, timeperiod=14)


def test_benchmark_talib_atr_numpy(benchmark):
    benchmark.group = "ATR"
    benchmark(talib.ATR, _high, _low, _close, timeperiod=14)


# ======================== Volume ============================================


def test_benchmark_pytafast_obv_numpy(benchmark):
    benchmark.group = "OBV"
    benchmark(pytafast.OBV, _close, _volume)


def test_benchmark_talib_obv_numpy(benchmark):
    benchmark.group = "OBV"
    benchmark(talib.OBV, _close, _volume)


# ======================== Statistics ========================================

for name in ["LINEARREG", "LINEARREG_SLOPE", "TSF"]:
    p_np, t_numpy = _make_single_benchmarks(name, timeperiod=14)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy

# ======================== Math Transforms ===================================

for name in ["SIN", "COS", "TAN", "SQRT", "LN", "LOG10"]:
    p_np, t_numpy = _make_single_benchmarks(name)
    globals()[f"test_benchmark_pytafast_{name.lower()}_numpy"] = p_np
    globals()[f"test_benchmark_talib_{name.lower()}_numpy"] = t_numpy


def test_benchmark_pytafast_ht_phasor_numpy(benchmark):
    benchmark.group = "HT_PHASOR"
    benchmark(pytafast.HT_PHASOR, _close)


def test_benchmark_talib_ht_phasor_numpy(benchmark):
    benchmark.group = "HT_PHASOR"
    benchmark(talib.HT_PHASOR, _close)


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
        futures = [
            executor.submit(func, *args, **kwargs) for _ in range(CONCURRENT_TASKS)
        ]
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
