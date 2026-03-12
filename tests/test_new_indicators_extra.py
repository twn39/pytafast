import pytest
import pytafast
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILES = [
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
    if f.endswith(".csv") and "r_all_results" not in f and "expected" not in f and "benchmark" not in f
] if os.path.exists(DATA_DIR) else []

def assert_aligned(p_val, r_val, rtol=1e-4, atol=1e-4):
    """Compare Python and R results only where both are not NaN."""
    mask = (~np.isnan(p_val)) & (~np.isnan(r_val))
    if not np.any(mask):
        assert False, "No overlapping non-NaN values to compare"
    np.testing.assert_allclose(p_val[mask], r_val[mask], rtol=rtol, atol=atol)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_alma_alignment(reference_data):
    df, r_df = reference_data
    close = df["Close"].values
    out = pytafast.ALMA(close, timeperiod=9, offset=0.85, sigma=6.0)
    assert_aligned(out, r_df['ALMA'].values)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_mama_stability(reference_data):
    df, r_df = reference_data
    close = df["Close"].values
    # MAMA not in TTR, verify stability and metadata
    mama, fama = pytafast.MAMA(close, fastlimit=0.5, slowlimit=0.05)
    assert len(mama) == len(close)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_snr_alignment(reference_data):
    df, r_df = reference_data
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    out = pytafast.SNR(high, low, close, timeperiod=14)
    assert_aligned(out, r_df['SNR'].values)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_vhf_alignment(reference_data):
    df, r_df = reference_data
    close = df["Close"].values
    out = pytafast.VHF(close, timeperiod=28)
    assert_aligned(out, r_df['VHF'].values)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_zlema_alignment(reference_data):
    df, r_df = reference_data
    close = df["Close"].values
    out = pytafast.ZLEMA(close, timeperiod=30)
    assert_aligned(out, r_df['ZLEMA'].values)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_evwma_alignment(reference_data):
    df, r_df = reference_data
    close = df["Close"].values
    vol = df["Volume"].values
    out = pytafast.EVWMA(close, vol, timeperiod=30)
    assert_aligned(out, r_df['EVWMA'].values)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_hma_alignment(reference_data):
    df, r_df = reference_data
    close = df["Close"].values
    out = pytafast.HMA(close, timeperiod=20)
    assert_aligned(out, r_df['HMA'].values)
