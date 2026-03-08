import os
import pytest
import pandas as pd
import pytafast

# Force skip in CI environments to avoid ChromeNotFoundError
if os.getenv("GITHUB_ACTIONS"):
    pytest.skip("Skipping plotting tests in CI environment", allow_module_level=True)

def test_plotting_all():
    pytest.importorskip("kaleido")

    # Load sample data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "nasdaq100_2025_now.csv")
    df = pd.read_csv(data_path)

    # Create a master analysis chart with almost everything
    chart = (
        pytafast.Chart(df)
        .add_candlestick(name="Main Price")
        .add_bbands(n=20)
        .add_donchian(n=10)
        .add_ema(n=50, color="blue")
        .add_alma(n=9)
        .add_zigzag(change=1.5)
        .add_patterns()  # Automatically label all candlestick patterns
        .add_volume()
        .add_macd()
        .add_rsi()
        .add_mfi()
        .add_stoch()
        .add_atr()
    )

    # Set a custom title
    chart.title = "NASDAQ 100 Comprehensive Technical Analysis (pytafast)"

    # Save both formats
    chart.save_html("full_analysis.html")
    chart.save_image("full_analysis.png", w=1400, h=1600)  # Tall image for many subplots

    print("Full-indicator plotting test completed.")
