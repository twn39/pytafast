import os
import pytest
import pandas as pd
import pytafast

# Force skip in CI environments to avoid ChromeNotFoundError
# Using pytestmark for more robust skipping during collection
pytestmark = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping plotting tests in CI environment (requires Chrome)",
)


def test_plotting():
    pytest.importorskip("kaleido")

    # Load sample data
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "nasdaq100_2025_now.csv"
    )
    df = pd.read_csv(data_path)

    # Create a complex, quantmod-style chart with one chained command
    chart = (
        pytafast.Chart(df)
        .add_candlestick(name="NASDAQ 100")
        .add_sma(n=20, color="orange")
        .add_bbands(n=20, sd=2.0)
        .add_zigzag(change=2.0)
        .add_volume()
        .add_rsi(n=14)
        .add_macd(f=12, s=26, sig=9)
    )

    # Save the interactive chart to HTML for inspection
    chart.save_html("sample_chart.html")
    # Save as static PNG
    chart.save_image("sample_chart.png")
    print("Plotting test completed successfully.")
