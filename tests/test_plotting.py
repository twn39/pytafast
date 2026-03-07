import pandas as pd
import pytafast

# Load sample data
df = pd.read_csv("data/nasdaq100_2025_now.csv")

# Create a complex, quantmod-style chart with one chained command
chart = (
    pytafast.Chart(df)
    .add_candlestick(name="NASDAQ 100")
    .add_sma(n=20, color="orange")
    .add_bbands(n=20, sd=2.0)
    .add_zigzag(change=2.0)
    .add_volume()
    .add_rsi(n=14)
    .add_macd(fast=12, slow=26, signal=9)
)

# Save the interactive chart to HTML for inspection
chart.save_html("sample_chart.html")
# Save as static PNG
chart.save_image("sample_chart.png")
print("Plotting test completed successfully.")
