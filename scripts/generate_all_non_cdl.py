import pandas as pd
import pytafast

# Load sample data
df = pd.read_csv("data/nasdaq100_2025_now.csv")

# 1. All Overlays (Part 1: Moving Averages)
chart1 = (
    pytafast.Chart(df)
    .add_candlestick()
    .add_sma(20)
    .add_ema(30)
    .add_wma(40)
    .add_dema(20)
    .add_tema(20)
    .add_kama(30)
    .add_hma(20)
    .add_alma(9)
    .add_zlema(30)
    .add_evwma(30)
)
chart1.title = "All Moving Average Overlays"
chart1.save_image("all_overlays_ma.png", w=1400, h=800)

# 2. All Overlays (Part 2: Channels, SAR, ZigZag)
chart2 = (
    pytafast.Chart(df)
    .add_candlestick()
    .add_bbands()
    .add_keltner()
    .add_donchian()
    .add_sar()
    .add_zigzag(change=2.0)
)
chart2.title = "All Channel & Trend Overlays"
chart2.save_image("all_overlays_channels.png", w=1400, h=800)

# 3. All Oscillators (Part 1)
chart3 = pytafast.Chart(df).add_line().add_macd().add_rsi().add_stoch().add_willr()
chart3.title = "Non-CDL Oscillators - Part 1"
chart3.save_image("all_oscillators_1.png", w=1200, h=1200)

# 4. All Oscillators (Part 2)
chart4 = (
    pytafast.Chart(df)
    .add_candlestick()
    .add_volume()
    .add_cci()
    .add_mfi()
    .add_cmf()
    .add_kst()
    .add_atr()
)
chart4.title = "Non-CDL Oscillators & Volume - Part 2"
chart4.save_image("all_oscillators_2.png", w=1200, h=1400)

print("All non-CDL charts generated.")
