import pytest
import numpy as np
import pytafast
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILES = (
    [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
        and "r_all_results" not in f
        and "expected" not in f
        and "benchmark" not in f
    ]
    if os.path.exists(DATA_DIR)
    else []
)


@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_adx_alignment(reference_data):
    df, r_df = reference_data
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values

    # Calculate with pytafast
    adx = pytafast.ADX(H, L, C, 14)
    pdi = pytafast.PLUS_DI(H, L, C, 14)
    mdi = pytafast.MINUS_DI(H, L, C, 14)

    # Compare with R (TTR ADX uses wilder=TRUE by default)
    # Filter NaNs for DI comparison as initialization varies slightly
    mask = ~np.isnan(adx) & ~np.isnan(r_df["ADX"].values)

    np.testing.assert_allclose(
        adx[mask], r_df["ADX"].values[mask], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        pdi[mask], r_df["PLUS_DI"].values[mask], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        mdi[mask], r_df["MINUS_DI"].values[mask], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_ichimoku_alignment(reference_data):
    df, r_df = reference_data
    H, L = df["High"].values, df["Low"].values

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
    mask = ~np.isnan(tenkan) & ~np.isnan(r_df["Tenkan"].values)
    np.testing.assert_allclose(tenkan[mask], r_df["Tenkan"].values[mask], rtol=1e-7)

    mask_k = ~np.isnan(kijun) & ~np.isnan(r_df["Kijun"].values)
    np.testing.assert_allclose(kijun[mask_k], r_df["Kijun"].values[mask_k], rtol=1e-7)

    mask_ssa = ~np.isnan(ssa) & ~np.isnan(r_df["SenkouA"].values)
    np.testing.assert_allclose(
        ssa[mask_ssa], r_df["SenkouA"].values[mask_ssa], rtol=1e-7
    )

    mask_ssb = ~np.isnan(ssb) & ~np.isnan(r_df["SenkouB"].values)
    np.testing.assert_allclose(
        ssb[mask_ssb], r_df["SenkouB"].values[mask_ssb], rtol=1e-7
    )


@pytest.mark.parametrize("reference_data", DATA_FILES, indirect=True)
def test_tdi_alignment(reference_data):
    df, r_df = reference_data
    C = df["Close"].values

    # TDI Logic: RSI(13), Price Line = SMA(RSI, 2), Signal Line = SMA(RSI, 7)
    # Market Base Line = SMA(RSI, 34), BBands(RSI, 34, 1.6185)
    rsi = pytafast.RSI(C, 13)
    price_line = pytafast.SMA(rsi, 2)
    signal_line = pytafast.SMA(rsi, 7)

    u, m, l = pytafast.BBANDS(rsi, 34, 1.6185, 1.6185)

    # Compare only non-NaN values in both
    mask = ~np.isnan(price_line) & ~np.isnan(r_df["TDI_price"].values)
    np.testing.assert_allclose(
        price_line[mask], r_df["TDI_price"].values[mask], rtol=1e-5, atol=1e-5
    )

    mask_sig = ~np.isnan(signal_line) & ~np.isnan(r_df["TDI_signal"].values)
    np.testing.assert_allclose(
        signal_line[mask_sig], r_df["TDI_signal"].values[mask_sig], rtol=1e-5, atol=1e-5
    )

    mask_bb = ~np.isnan(m) & ~np.isnan(r_df["TDI_mbl"].values)
    np.testing.assert_allclose(
        m[mask_bb], r_df["TDI_mbl"].values[mask_bb], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        u[mask_bb], r_df["TDI_ub"].values[mask_bb], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        l[mask_bb], r_df["TDI_lb"].values[mask_bb], rtol=1e-5, atol=1e-5
    )
