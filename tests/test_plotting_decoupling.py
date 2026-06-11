# tests/test_plotting_decoupling.py
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from pytafast import Chart
from pytafast.plotting import register_custom_indicator

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2026-01-01", periods=n, freq="D")
    close = 100 + np.random.randn(n).cumsum()
    open_val = close + np.random.randn(n) * 0.5
    high = np.maximum(open_val, close) + np.random.rand(n)
    low = np.minimum(open_val, close) - np.random.rand(n)
    volume = np.random.randint(1000, 5000, size=n).astype(float)
    
    df = pd.DataFrame({
        "Date": dates,
        "Open": open_val,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })
    return df


def test_core_chart_rendering(sample_data):
    chart = Chart(sample_data)
    # Add standard plots
    chart.add_candlestick()
    chart.add_volume()
    
    fig = chart.render()
    assert fig is not None
    # We should have main price row and volume row
    # Trace 0 is candlestick
    assert len(fig.data) >= 2
    assert any(isinstance(t, go.Candlestick) for t in fig.data)
    assert any(isinstance(t, go.Bar) for t in fig.data)


def test_dynamic_indicator_routing(sample_data):
    chart = Chart(sample_data)
    # add_sma and add_rsi are handled dynamically via __getattr__
    chart.add_sma(20)
    chart.add_rsi(14)
    chart.add_macd()
    
    fig = chart.render()
    assert fig is not None
    
    # Verify traces are added
    # SMA should be in main_traces (overlay)
    assert any(t.name and "SMA(20)" in t.name for t in fig.data if isinstance(t, go.Scatter))
    # RSI should be in subplots
    assert any(t.name and "RSI" in t.name for t in fig.data)
    # MACD should be in subplots (MACD line, Signal line, MACD Hist bar)
    assert any(t.name and "MACD" in t.name for t in fig.data)


def test_generic_add_indicator(sample_data):
    chart = Chart(sample_data)
    
    # Custom indicator function that matches single-input signature
    def my_custom_indicator(inReal, factor=2.0):
        return inReal * factor
        
    chart.add_indicator(my_custom_indicator, factor=1.5, name="CustomLine")
    
    fig = chart.render()
    assert fig is not None
    assert any(t.name and "CustomLine" in t.name for t in fig.data)


def test_custom_indicator_registration(sample_data):
    # Custom plotter strategy
    from pytafast.plotting.plotters import BasePlotter
    
    class SpecialPlotter(BasePlotter):
        def plot(self, chart):
            y = self.indicator_fn(chart.C)
            chart._add_overlay(y, "SpecialOverlay", color="purple", width=3)
            
    def my_special_indicator(inReal):
        return inReal + 5.0
        
    # Register the custom indicator with SpecialPlotter and the indicator function
    register_custom_indicator("SPECIAL", SpecialPlotter, my_special_indicator)
    
    # Plot using dynamic routing
    chart = Chart(sample_data)
    chart.add_special()
    
    fig = chart.render()
    assert fig is not None
    assert any(t.name and "SpecialOverlay" in t.name for t in fig.data)
    
    # Find the special trace and check color and width
    special_trace = next(t for t in fig.data if t.name == "SpecialOverlay")
    assert special_trace.line.color == "purple"
    assert special_trace.line.width == 3


def test_nonexistent_indicator_raises_attribute_error(sample_data):
    chart = Chart(sample_data)
    with pytest.raises(AttributeError):
        chart.add_nonexistent_indicator_12345()
