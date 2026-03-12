import pandas as pd
import pytafast

# Load sample data
df = pd.read_csv("data/nasdaq100_2025_now.csv")

def create_base_chart(df):
    return (pytafast.Chart(df)
            .add_candlestick()
            .add_sma(20)
            .add_bbands()
            .add_volume()
            .add_macd()
            .add_rsi())

# 1. Dark Theme
chart_dark = create_base_chart(df).set_theme("dark")
chart_dark.title = "Review: Dark Theme (plotly_dark)"
chart_dark.save_image("theme_dark.png")

# 2. ggplot2 Theme
chart_gg = create_base_chart(df).set_theme("ggplot2")
chart_gg.title = "Review: ggplot2 Style"
chart_gg.save_image("theme_ggplot2.png")

# 3. Seaborn Theme
chart_sb = create_base_chart(df).set_theme("seaborn")
chart_sb.title = "Review: Seaborn Style"
chart_sb.save_image("theme_seaborn.png")

print("Theme review charts generated: theme_dark.png, theme_ggplot2.png, theme_seaborn.png")
