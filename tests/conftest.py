import os
import pytest
import numpy as np
import pandas as pd
import subprocess
import shutil


# Check if Rscript is available and has required packages
def check_r_env():
    if shutil.which("Rscript") is None:
        return False
    try:
        # Try to load required libraries
        result = subprocess.run(
            ["Rscript", "-e", "library(TTR); library(quantmod)"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


HAS_R_ENV = check_r_env()

# Use absolute paths relative to this test file to avoid FileNotFoundError in CI
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

if os.path.exists(DATA_DIR):
    # List all CSV files in data directory
    DATA_FILES = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and "r_all_results" not in f
    ]
else:
    DATA_FILES = []


@pytest.fixture
def prices():
    """100 synthetic price bars."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 100))
    high = close + rng.uniform(0.1, 2.0, 100)
    low = close - rng.uniform(0.1, 2.0, 100)
    open_ = close - rng.uniform(-1, 1, 100)
    volume = rng.uniform(1e5, 1e7, 100)
    return open_, high, low, close, volume


@pytest.fixture
def prices_pd(prices):
    """Same as prices but as pandas Series with a date index."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    o, h, low_val, c, v = prices
    return (
        pd.Series(o, index=idx, name="Open"),
        pd.Series(h, index=idx, name="High"),
        pd.Series(low_val, index=idx, name="Low"),
        pd.Series(c, index=idx, name="Close"),
        pd.Series(v, index=idx, name="Volume"),
    )


@pytest.fixture
def random_prices():
    """Simple 100-length random array for general tests."""
    np.random.seed(42)
    in_real = np.random.random(100) * 100
    in_high = in_real + 10 + np.random.random(100) * 5
    in_low = in_high - np.random.random(100) * 5
    in_close = in_low + (in_high - in_low) / 2
    in_open = in_low + np.random.random(100) * (in_high - in_low)
    in_vol = np.random.random(100) * 1000
    return {
        "real": in_real,
        "high": in_high,
        "low": in_low,
        "close": in_close,
        "open": in_open,
        "volume": in_vol,
        "in0": in_real,
        "in1": in_real + np.random.random(100) * 2,
    }


@pytest.fixture
def benchmark_data():
    """5000-length arrays for benchmark tests."""
    N = 5000
    np.random.seed(42)
    _close = np.random.random(N)
    _high = _close + np.random.random(N) * 0.05
    _low = _close - np.random.random(N) * 0.05
    _open = _low + np.random.random(N) * (_high - _low)
    _volume = np.random.random(N) * 1000000
    return {
        "close": _close,
        "high": _high,
        "low": _low,
        "open": _open,
        "volume": _volume,
    }


@pytest.fixture(
    params=[
        "samsung_3m.csv",
        "icbc_2025.csv",
        "nasdaq100_2025_now.csv",
        "sk_hynix_1y.csv",
        "berkshire_1y.csv",
    ]
)
def stock_data_context(request):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", request.param)
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
        "成交量": "volume",
    }
    df = df.rename(columns=mapping)

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return data_path, df


@pytest.fixture
def stock_data(stock_data_context):
    return stock_data_context[1]


@pytest.fixture(scope="function")
def reference_data(request):
    """Run R script for a specific data file to generate reference values."""
    if not HAS_R_ENV:
        pytest.skip("R environment or required packages (TTR, quantmod) not found.")

    data_file = request.param
    if not os.path.exists(data_file):
        pytest.skip(f"Data file not found: {data_file}")

    # Set environment variable for R script
    os.environ["DATA_FILE"] = data_file
    # Run R script from project root
    subprocess.run(
        ["Rscript", os.path.join(BASE_DIR, "scripts", "validation", "compute_all_r.R")],
        check=True,
        cwd=BASE_DIR,
    )

    # The R script now saves to data/r_all_results.csv
    r_results_path = os.path.join(DATA_DIR, "r_all_results.csv")
    r_results = pd.read_csv(r_results_path)

    df = pd.read_csv(data_file)
    df.columns = [c.lower() for c in df.columns]

    mapping = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
    df = df.rename(columns=mapping)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Change column names back to Title case for downstream test compatibility
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "date": "Date",
        }
    )

    return df, r_results
