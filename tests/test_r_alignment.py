import pytest
import numpy as np
import pandas as pd
import pytafast
import subprocess
import os
import io

def run_r_ttr(data_path, r_code):
    """Run R code and return the resulting CSV as a pandas DataFrame."""
    # Ensure data_path is absolute for R
    abs_data_path = os.path.abspath(data_path)
    
    full_r_script = f"""
    library(TTR)
    df <- read.csv("{abs_data_path}")
    # Map to expected TTR inputs
    open <- df$Open
    high <- df$High
    low <- df$Low
    close <- df$Close
    volume <- df$Volume
    
    {r_code}
    
    # Print as CSV to stdout
    write.csv(res, row.names=FALSE)
    """
    
    process = subprocess.Popen(
        ["Rscript", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=full_r_script)
    
    if process.returncode != 0:
        raise RuntimeError(f"R execution failed: {stderr}")
        
    return pd.read_csv(io.StringIO(stdout))

def assert_aligned(p_val, r_val, atol=1e-5):
    """Compare Python and R results only where both are not NaN."""
    mask = (~np.isnan(p_val)) & (~np.isnan(r_val))
    if not np.any(mask):
        # If no overlapping non-NaN values, something is wrong
        assert False, "No overlapping non-NaN values to compare"
    np.testing.assert_allclose(p_val[mask], r_val[mask], atol=atol)

@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_aroon_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R calculation
    r_res = run_r_ttr(data_path, "res <- aroon(cbind(high, low), n=14)")
    
    # Python calculation
    p_dn, p_up = pytafast.AROON(df["High"].values, df["Low"].values, 14)
    
    # Compare overlapping regions
    assert_aligned(p_up, r_res["aroonUp"].values)
    assert_aligned(p_dn, r_res["aroonDn"].values)

@pytest.mark.xfail(reason="SMI in pytafast follows TA-Lib/Blau standard, which differs from TTR in NaN handling and smoothing seeds.")
@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_smi_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R calculation: TTR::SMI(HLC, n=13, nFast=2, nSlow=25, nSig=9)
    r_res = run_r_ttr(data_path, "res <- SMI(cbind(high, low, close), n=13, nFast=2, nSlow=25, nSig=9)")
    
    # Python
    p_smi, p_sig = pytafast.SMI(df["High"].values, df["Low"].values, df["Close"].values, 13, 2, 25, 9)
    
    assert_aligned(p_smi, r_res["SMI"].values)
    assert_aligned(p_sig, r_res["signal"].values)

@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_emv_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R: TTR::EMV(HL, vol, n=9)
    r_res = run_r_ttr(data_path, "res <- EMV(cbind(high, low), volume, n=9)")
    
    # Python
    p_emv, p_sig = pytafast.EMV(df["High"].values, df["Low"].values, df["Volume"].values, 9)
    
    assert_aligned(p_emv, r_res["emv"].values)
    assert_aligned(p_sig, r_res["maEMV"].values)

@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_dpo_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R: TTR::DPO(x, n=10)
    r_res = run_r_ttr(data_path, "res <- data.frame(dpo=DPO(close, n=10))")
    
    # Python
    p_dpo = pytafast.DPO(df["Close"].values, 10)
    
    assert_aligned(p_dpo, r_res["dpo"].values)

@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_obv_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R: TTR::OBV(price, vol)
    r_res = run_r_ttr(data_path, "res <- data.frame(obv=OBV(close, volume))")
    
    # Python
    p_obv = pytafast.OBV(df["Close"].values, df["Volume"].values)
    
    assert_aligned(p_obv, r_res["obv"].values)

@pytest.mark.xfail(reason="CMO in pytafast uses TA-Lib logic (similar to RSI), TTR uses a slightly different scale/smoothing.")
@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_cmo_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R: TTR::CMO(x, n=14)
    r_res = run_r_ttr(data_path, "res <- data.frame(cmo=CMO(close, n=14))")
    
    # Python
    p_cmo = pytafast.CMO(df["Close"].values, 14)
    
    assert_aligned(p_cmo, r_res["cmo"].values)

@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_roc_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R: TTR::ROC(x, n=10, type="discrete")
    r_res = run_r_ttr(data_path, 'res <- data.frame(roc=ROC(close, n=10, type="discrete") * 100)')
    
    # Python
    p_roc = pytafast.ROC(df["Close"].values, 10)
    
    assert_aligned(p_roc, r_res["roc"].values)

@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_clv_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R: TTR::CLV(HLC)
    r_res = run_r_ttr(data_path, "res <- data.frame(clv=CLV(cbind(high, low, close)))")
    
    # Python (manually calculated in plotting.py, let's test that logic)
    denom = df["High"].values - df["Low"].values
    p_clv = np.where(denom != 0, ((df["Close"].values - df["Low"].values) - (df["High"].values - df["Close"].values)) / denom, 0.0)
    
    assert_aligned(p_clv, r_res["clv"].values)

@pytest.mark.parametrize("data_file", ["nasdaq100_2025_now.csv"])
def test_chaikin_vol_alignment(data_file):
    data_path = os.path.join("data", data_file)
    df = pd.read_csv(data_path)
    
    # R: TTR::chaikinVolatility(HL, n=10)
    r_res = run_r_ttr(data_path, "res <- data.frame(chv=chaikinVolatility(cbind(high, low), n=10) * 100)")
    
    # Python (plotting.py logic)
    hl = df["High"].values - df["Low"].values
    ema_hl = pytafast.EMA(hl, 10)
    p_chv = np.full_like(ema_hl, np.nan)
    if len(ema_hl) > 10:
        p_chv[10:] = ((ema_hl[10:] / ema_hl[:-10]) - 1.0) * 100
    
    assert_aligned(p_chv, r_res["chv"].values)
