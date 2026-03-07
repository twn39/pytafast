import pandas as pd
import pytafast


def test_new_indicators():
    df = pd.read_csv("data/berkshire_1y.csv")
    df = df.sort_values("Date")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    print("\n--- Testing DonchianChannel ---")
    upper, mid, lower = pytafast.DonchianChannel(high, low, timeperiod=10)
    print(f"Shapes: {upper.shape}, {mid.shape}, {lower.shape}")
    print("Sample (last 5):")
    print(pd.DataFrame({"Upper": upper, "Mid": mid, "Lower": lower}).tail())

    print("\n--- Testing GMMA ---")
    gmma = pytafast.GMMA(close)
    print(f"GMMA count: {len(gmma)}")
    print(f"First EMA shape: {gmma[0].shape}")

    print("\n--- Testing KST ---")
    kst, signal = pytafast.KST(close)
    print(f"KST shape: {kst.shape}, Signal shape: {signal.shape}")
    kst_df = pd.DataFrame({"KST": kst, "Signal": signal})
    print("KST Sample (middle 5):")
    print(kst_df.iloc[100:105])
    print("KST Valid count:", kst_df["KST"].notna().sum())

    print("\n--- Testing ZigZag ---")
    zz = pytafast.ZIGZAG(high, low, change=5.0, percent=True)
    print(f"ZigZag shape: {zz.shape}")
    print("Sample (last 10):")
    print(zz.tail(10))

    # Check if ZigZag has non-NaN values
    valid_count = zz.notna().sum()
    print(f"Valid ZigZag points: {valid_count}")
    assert valid_count > 0
