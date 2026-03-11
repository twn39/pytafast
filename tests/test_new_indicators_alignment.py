import os
import subprocess
import pytest
import pandas as pd
import numpy as np
import pytafast
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
            text=True
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
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
        if f.endswith(".csv") and "r_all_results" not in f
    ]
else:
    DATA_FILES = []

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
    subprocess.run(["Rscript", os.path.join(BASE_DIR, "scripts/compute_all_r.R")], 
                   check=True, cwd=BASE_DIR)
    
    # The R script now saves to data/r_all_results.csv
    r_results_path = os.path.join(DATA_DIR, "r_all_results.csv")
    r_results = pd.read_csv(r_results_path)
    input_data = pd.read_csv(data_file)
    return input_data, r_results

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_adx_alignment(reference_data):
    df, r_df = reference_data
    H, L, C = df['High'].values, df['Low'].values, df['Close'].values
    
    # Calculate with pytafast
    adx = pytafast.ADX(H, L, C, 14)
    pdi = pytafast.PLUS_DI(H, L, C, 14)
    mdi = pytafast.MINUS_DI(H, L, C, 14)
    
    # Compare with R (TTR ADX uses wilder=TRUE by default)
    # Filter NaNs for DI comparison as initialization varies slightly
    mask = ~np.isnan(adx) & ~np.isnan(r_df['ADX'].values)
    
    np.testing.assert_allclose(adx[mask], r_df['ADX'].values[mask], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pdi[mask], r_df['PLUS_DI'].values[mask], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mdi[mask], r_df['MINUS_DI'].values[mask], rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_ichimoku_alignment(reference_data):
    df, r_df = reference_data
    H, L = df['High'].values, df['Low'].values
    
    # Calculate components
    t_high = pytafast.MAX(H, 9)
    t_low = pytafast.MIN(L, 9)
    tenkan = (t_high + t_low) / 2
    
    k_high = pytafast.MAX(H, 26)
    k_low = pytafast.MIN(L, 26)
    kijun = (k_high + k_low) / 2
    
    ssa = (tenkan + kijun) / 2
    
    b_high = pytafast.MAX(H, 52)
    b_low = pytafast.MIN(L, 52)
    ssb = (b_high + b_low) / 2
    
    # Compare non-nan values
    mask = ~np.isnan(tenkan) & ~np.isnan(r_df['Tenkan'].values)
    np.testing.assert_allclose(tenkan[mask], r_df['Tenkan'].values[mask], rtol=1e-7)
    
    mask_k = ~np.isnan(kijun) & ~np.isnan(r_df['Kijun'].values)
    np.testing.assert_allclose(kijun[mask_k], r_df['Kijun'].values[mask_k], rtol=1e-7)
    
    mask_ssa = ~np.isnan(ssa) & ~np.isnan(r_df['SenkouA'].values)
    np.testing.assert_allclose(ssa[mask_ssa], r_df['SenkouA'].values[mask_ssa], rtol=1e-7)
    
    mask_ssb = ~np.isnan(ssb) & ~np.isnan(r_df['SenkouB'].values)
    np.testing.assert_allclose(ssb[mask_ssb], r_df['SenkouB'].values[mask_ssb], rtol=1e-7)

@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_tdi_alignment(reference_data):
    df, r_df = reference_data
    C = df['Close'].values
    
    # TDI Logic: RSI(13), Price Line = SMA(RSI, 2), Signal Line = SMA(RSI, 7)
    # Market Base Line = SMA(RSI, 34), BBands(RSI, 34, 1.6185)
    rsi = pytafast.RSI(C, 13)
    price_line = pytafast.SMA(rsi, 2)
    signal_line = pytafast.SMA(rsi, 7)
    
    u, m, l = pytafast.BBANDS(rsi, 34, 1.6185, 1.6185)
    
    # Compare only non-NaN values in both
    mask = ~np.isnan(price_line) & ~np.isnan(r_df['TDI_price'].values)
    np.testing.assert_allclose(price_line[mask], r_df['TDI_price'].values[mask], rtol=1e-5, atol=1e-5)
    
    mask_sig = ~np.isnan(signal_line) & ~np.isnan(r_df['TDI_signal'].values)
    np.testing.assert_allclose(signal_line[mask_sig], r_df['TDI_signal'].values[mask_sig], rtol=1e-5, atol=1e-5)
    
    mask_bb = ~np.isnan(m) & ~np.isnan(r_df['TDI_mbl'].values)
    np.testing.assert_allclose(m[mask_bb], r_df['TDI_mbl'].values[mask_bb], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(u[mask_bb], r_df['TDI_ub'].values[mask_bb], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(l[mask_bb], r_df['TDI_lb'].values[mask_bb], rtol=1e-5, atol=1e-5)
