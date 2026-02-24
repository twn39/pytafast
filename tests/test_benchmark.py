import pytest
import numpy as np
import pandas as pd
import pytafast
talib = pytest.importorskip("talib")

# Generate a large dataset for meaningful benchmarking
np.random.seed(42)
LARGE_ARRAY = np.random.random(100_000) * 100
LARGE_SERIES = pd.Series(LARGE_ARRAY)

TIME_PERIOD = 30

def test_benchmark_pytafast_sma_numpy(benchmark):
    benchmark(pytafast.SMA, LARGE_ARRAY, timeperiod=TIME_PERIOD)

def test_benchmark_official_talib_sma_numpy(benchmark):
    benchmark(talib.SMA, LARGE_ARRAY, timeperiod=TIME_PERIOD)

def test_benchmark_pytafast_sma_pandas(benchmark):
    benchmark(pytafast.SMA, LARGE_SERIES, timeperiod=TIME_PERIOD)

def test_benchmark_official_talib_sma_pandas(benchmark):
    benchmark(talib.SMA, LARGE_SERIES, timeperiod=TIME_PERIOD)
