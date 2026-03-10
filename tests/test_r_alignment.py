import pytest
import numpy as np
import pandas as pd
import pytafast
import subprocess
import os
import io


def load_r_res(data_path, indicator_name):
    r_file = os.path.join(os.path.dirname(data_path), "r_expected", f"expected_{indicator_name}_{os.path.basename(data_path)}")
    return pd.read_csv(r_file)

def assert_aligned(p_val, r_val, atol=1e-5):
    """Compare Python and R results only where both are not NaN."""
    mask = (~np.isnan(p_val)) & (~np.isnan(r_val))
    if not np.any(mask):
        # If no overlapping non-NaN values, something is wrong
        assert False, "No overlapping non-NaN values to compare"
    np.testing.assert_allclose(p_val[mask], r_val[mask], atol=atol)


def test_aroon_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R calculation
    r_res = load_r_res(data_path, "aroon")

    # Python calculation
    p_dn, p_up = pytafast.AROON(df["high"].values, df["low"].values, 14)

    # Compare overlapping regions
    assert_aligned(p_up, r_res["aroonUp"].values)
    assert_aligned(p_dn, r_res["aroonDn"].values)


@pytest.mark.xfail(
    reason="SMI in pytafast follows TA-Lib/Blau standard, which differs from TTR in NaN handling and smoothing seeds."
)
def test_smi_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R calculation: TTR::SMI(HLC, n=13, nFast=2, nSlow=25, nSig=9)
    r_res = load_r_res(data_path, "smi")

    # Python
    p_smi, p_sig = pytafast.SMI(
        df["high"].values, df["low"].values, df["close"].values, 13, 2, 25, 9
    )

    assert_aligned(p_smi, r_res["SMI"].values)
    assert_aligned(p_sig, r_res["signal"].values)


def test_emv_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R: TTR::EMV(HL, vol, n=9)
    r_res = load_r_res(data_path, "emv")

    # Python
    p_emv, p_sig = pytafast.EMV(
        df["high"].values, df["low"].values, df["volume"].values, 9
    )

    assert_aligned(p_emv, r_res["emv"].values)
    assert_aligned(p_sig, r_res["maEMV"].values)


def test_dpo_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R: TTR::DPO(x, n=10)
    r_res = load_r_res(data_path, "dpo")

    # Python
    p_dpo = pytafast.DPO(df["close"].values, 10)

    assert_aligned(p_dpo, r_res["dpo"].values)


def test_obv_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R: TTR::OBV(price, vol)
    r_res = load_r_res(data_path, "obv")

    # Python
    p_obv = pytafast.OBV(df["close"].values, df["volume"].values)

    assert_aligned(p_obv, r_res["obv"].values)


@pytest.mark.xfail(
    reason="CMO in pytafast uses TA-Lib logic (similar to RSI), TTR uses a slightly different scale/smoothing."
)
def test_cmo_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R: TTR::CMO(x, n=14)
    r_res = load_r_res(data_path, "cmo")

    # Python
    p_cmo = pytafast.CMO(df["close"].values, 14)

    assert_aligned(p_cmo, r_res["cmo"].values)


def test_roc_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R: TTR::ROC(x, n=10, type="discrete")
    r_res = load_r_res(data_path, "roc")

    # Python
    p_roc = pytafast.ROC(df["close"].values, 10)

    assert_aligned(p_roc, r_res["roc"].values)


def test_clv_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R: TTR::CLV(HLC)
    r_res = load_r_res(data_path, "clv")

    # Python (manually calculated in plotting.py, let's test that logic)
    denom = df["high"].values - df["low"].values
    p_clv = np.where(
        denom != 0,
        (
            (df["close"].values - df["low"].values)
            - (df["high"].values - df["close"].values)
        )
        / denom,
        0.0,
    )

    assert_aligned(p_clv, r_res["clv"].values)


def test_chaikin_vol_alignment(r_stock_data_context):
    data_path, df = r_stock_data_context

    # R: TTR::chaikinVolatility(HL, n=10)
    r_res = load_r_res(data_path, "chaikin_vol")

    # Python (plotting.py logic)
    hl = df["high"].values - df["low"].values
    ema_hl = pytafast.EMA(hl, 10)
    p_chv = np.full_like(ema_hl, np.nan)
    if len(ema_hl) > 10:
        p_chv[10:] = ((ema_hl[10:] / ema_hl[:-10]) - 1.0) * 100

    assert_aligned(p_chv, r_res["chv"].values)
