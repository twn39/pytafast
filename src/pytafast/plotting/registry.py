# src/pytafast/plotting/registry.py
from .plotters import (
    LinePlotter,
    BandPlotter,
    OscillatorPlotter,
    MACDPlotter,
    StochasticPlotter,
    AroonPlotter,
    ADXPlotter,
    TDIPlotter,
    KSTPlotter,
    SMIPlotter,
    IchimokuPlotter,
    ZigZagPlotter,
    SARPlotter,
    EMVPlotter,
    CMFPlotter,
    OBVPlotter,
    DPOPlotter,
    CLVPlotter,
    VolatilityPlotter,
    PatternsPlotter,
    EnvelopePlotter,
)

# Map indicator function name (as an uppercase string) to its Plotter configuration
INDICATOR_REGISTRY = {
    # Overlap Studies (Main Chart Overlays)
    "SMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#f29d4b"}},
    "EMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#5d62b5"}},
    "WMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#33a02c"}},
    "DEMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#e31a1c"}},
    "TEMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#fb9a99"}},
    "KAMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#6a3d9a"}},
    "HMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#b15928"}},
    "ALMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#1f78b4"}},
    "ZLEMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#ff7f00"}},
    "EVWMA": {"plotter": LinePlotter, "default_kwargs": {"color": "#b2df8a"}},
    "ZIGZAG": {"plotter": ZigZagPlotter, "default_kwargs": {}},
    "SAR": {"plotter": SARPlotter, "default_kwargs": {}},
    # Bands
    "BBANDS": {
        "plotter": BandPlotter,
        "default_kwargs": {
            "fillcolor": "rgba(173, 216, 230, 0.15)",
            "linecolor": "rgba(173, 216, 230, 0.4)",
        },
    },
    "KELTNERCHANNELS": {
        "plotter": BandPlotter,
        "default_kwargs": {
            "fillcolor": "rgba(255,165,0,0.1)",
            "linecolor": "rgba(255,165,0,0.3)",
            "mid_color": "orange",
        },
    },
    "KELTNER": {
        "plotter": BandPlotter,
        "function": "keltnerChannels",
        "default_kwargs": {
            "fillcolor": "rgba(255,165,0,0.1)",
            "linecolor": "rgba(255,165,0,0.3)",
            "mid_color": "orange",
        },
    },
    "DONCHIANCHANNEL": {
        "plotter": BandPlotter,
        "default_kwargs": {
            "fillcolor": "rgba(200,200,200,0.05)",
            "linecolor": "rgba(200,200,200,0.3)",
        },
    },
    "DONCHIAN": {
        "plotter": BandPlotter,
        "function": "DonchianChannel",
        "default_kwargs": {
            "fillcolor": "rgba(200,200,200,0.05)",
            "linecolor": "rgba(200,200,200,0.3)",
        },
    },
    "ENVELOPE": {
        "plotter": EnvelopePlotter,
        "default_kwargs": {"color": "rgba(0, 0, 255, 0.1)"},
    },
    "ICHIMOKU": {"plotter": IchimokuPlotter, "default_kwargs": {}},
    # Subplots - Oscillators (with thresholds)
    "RSI": {
        "plotter": OscillatorPlotter,
        "default_kwargs": {"yrange": [0, 100], "hlines": [30, 70], "color": "#9467bd"},
    },
    "CMO": {
        "plotter": OscillatorPlotter,
        "default_kwargs": {"yrange": [-100, 100], "hlines": [-50, 50]},
    },
    "WILLR": {
        "plotter": OscillatorPlotter,
        "default_kwargs": {"yrange": [-100, 0], "hlines": [-20, -80]},
    },
    # Subplots - Single line
    "ATR": {
        "plotter": LinePlotter,
        "default_kwargs": {"subplot": True, "height": 0.15},
    },
    "CCI": {"plotter": LinePlotter, "default_kwargs": {"subplot": True}},
    "MOM": {"plotter": LinePlotter, "default_kwargs": {"subplot": True}},
    "MOMENTUM": {
        "plotter": LinePlotter,
        "function": "MOM",
        "default_kwargs": {"subplot": True, "height": 0.2},
    },
    "MFI": {
        "plotter": LinePlotter,
        "default_kwargs": {"subplot": True, "yrange": [0, 100]},
    },
    "TRIX": {"plotter": LinePlotter, "default_kwargs": {"subplot": True}},
    "ROC": {"plotter": LinePlotter, "default_kwargs": {"subplot": True, "hlines": [0]}},
    # Subplots - Custom/Multi-line
    "MACD": {"plotter": MACDPlotter, "default_kwargs": {}},
    "STOCH": {"plotter": StochasticPlotter, "default_kwargs": {}},
    "AROON": {"plotter": AroonPlotter, "default_kwargs": {}},
    "SMI": {"plotter": SMIPlotter, "default_kwargs": {}},
    "KST": {"plotter": KSTPlotter, "default_kwargs": {}},
    "TDI": {"plotter": TDIPlotter, "default_kwargs": {}},
    "ADX": {"plotter": ADXPlotter, "default_kwargs": {}},
    "EMV": {"plotter": EMVPlotter, "default_kwargs": {}},
    "CMF": {"plotter": CMFPlotter, "default_kwargs": {}},
    "OBV": {"plotter": OBVPlotter, "default_kwargs": {}},
    "DPO": {"plotter": DPOPlotter, "default_kwargs": {}},
    "CLV": {"plotter": CLVPlotter, "default_kwargs": {}},
    "VOLATILITY": {"plotter": VolatilityPlotter, "default_kwargs": {}},
    "PATTERNS": {"plotter": PatternsPlotter, "default_kwargs": {}},
    "ADOSC": {
        "plotter": LinePlotter,
        "default_kwargs": {"subplot": True, "height": 0.2},
    },
    "CHAIKIN_OSC": {
        "plotter": LinePlotter,
        "function": "ADOSC",
        "default_kwargs": {"subplot": True, "height": 0.2},
    },
}


def register_custom_indicator(name, plotter_cls, indicator_fn=None, **default_kwargs):
    """Allows runtime registration of custom indicator plotters on Chart."""
    INDICATOR_REGISTRY[name.upper()] = {
        "plotter": plotter_cls,
        "indicator_fn": indicator_fn,
        "default_kwargs": default_kwargs,
    }
