import os
import pytest
import pandas as pd
import pytafast


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping plotting tests in CI environment",
)
def test_new_indicators_plotting():
    # Load sample data
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "nasdaq100_2025_now.csv"
    )
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Create a chart with the new indicators
    chart = (
        pytafast.Chart(df)
        .add_candlestick(name="NASDAQ 100")
        .add_ichimoku()
        .add_adx()
        .add_momentum()
        .add_tdi()
    )

    # Save the interactive chart to HTML
    output_path = "new_indicators_chart.html"
    chart.save_html(output_path)
    assert os.path.exists(output_path)
    print(f"New indicators plotting test completed. Output: {output_path}")
