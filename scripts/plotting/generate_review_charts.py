import pandas as pd
import pytafast

# Load sample data
df = pd.read_csv("data/nasdaq100_2025_now.csv")

# 1. Trend Focus: Candlestick + MA + Bands
chart1 = (pytafast.Chart(df)
          .add_candlestick(name="NASDAQ 100")
          .add_ema(n=50, color='#5d62b5')
          .add_bbands(n=20)
          .add_volume()
          .add_atr())
chart1.title = "Review 1: Trend & Volatility (Candlestick Focus)"
chart1.save_image("chart_trend.png", w=1200, h=800)

# 2. Momentum Focus: Line Chart + MACD + RSI + MFI
chart2 = (pytafast.Chart(df)
          .add_line(name="Close", color='#1f77b4')
          .add_donchian(n=20)
          .add_macd()
          .add_rsi()
          .add_mfi())
chart2.title = "Review 2: Momentum & Oscillators (Subplot Focus)"
chart2.save_image("chart_momentum.png", w=1200, h=1000)

# 3. Pattern & Flow: Candlestick + Pattern Recognition + CMF
chart3 = (pytafast.Chart(df)
          .add_candlestick()
          .add_patterns()  # Auto-label CDL patterns
          .add_zigzag(change=2.0)
          .add_volume()
          .add_cmf())
chart3.title = "Review 3: Pattern Recognition & Money Flow"
chart3.save_image("chart_special.png", w=1200, h=1000)

print("Three review charts generated: chart_trend.png, chart_momentum.png, chart_special.png")
