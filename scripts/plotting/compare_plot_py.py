import pandas as pd
import pytafast
import os

# Load data
df = pd.read_csv("data/nasdaq100_2025_now.csv")

# Create chart
chart = (
    pytafast.Chart(df)
    .set_theme("light")
    .add_candlestick(name="NASDAQ 100")
    .add_ichimoku()
    .add_adx()
    .add_momentum()
)

# Save as static image
chart.save_image("python_chart.png", w=1200, h=1000)
print("Python chart saved to python_chart.png")
