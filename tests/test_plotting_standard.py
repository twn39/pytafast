import os
import pytest
import pandas as pd
import pytafast

def test_plotting_standard():
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping plotting test in CI environment")
    pytest.importorskip("kaleido")
    # Load sample data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "nasdaq100_2025_now.csv")
    df = pd.read_csv(data_path)

    # Create a clean, standard analysis chart
    # Focus on CLARITY: don't overlap too many things on the main chart
    chart = (
        pytafast.Chart(df)
        .add_candlestick(name="NASDAQ 100")  # Standard Candlestick
        .add_sma(n=20, color="rgba(242, 157, 75, 0.8)")  # Single clean MA
        .add_bbands(n=20, sd=2.0)  # Standard Bands with light fill
        .add_volume()  # Subplot 1
        .add_macd()  # Subplot 2
        .add_rsi()  # Subplot 3
    )

    # Set a custom title
    chart.title = "NASDAQ 100 - Standard Technical Analysis"

    # Save both formats
    chart.save_html("standard_analysis.html")
    chart.save_image("standard_analysis.png", w=1200, h=1000)

    print("Standard plotting test completed. Check standard_analysis.html/png")
