# src/pytafast/plotting/chart.py
import importlib.util
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .registry import INDICATOR_REGISTRY
from .plotters import LinePlotter

class Chart:
    """
    A quantmod-inspired chaining chart builder for pytafast using Plotly.
    """

    THEMES = {
        "light": "plotly_white",
        "dark": "plotly_dark",
        "ggplot2": "ggplot2",
        "seaborn": "seaborn",
        "simple": "simple_white",
    }

    def __init__(
        self,
        df,
        date_col="Date",
        open_col="Open",
        high_col="High",
        low_col="Low",
        close_col="Close",
        vol_col="Volume",
    ):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            self.df[date_col] = pd.to_datetime(self.df[date_col])

        self.dt = self.df[date_col]
        self.O = self.df[open_col].values
        self.H = self.df[high_col].values
        self.L = self.df[low_col].values
        self.C = self.df[close_col].values
        self.V = self.df[vol_col].values if vol_col in self.df.columns else None

        self.main_traces = []
        self.subplots = []
        self.shapes = []  # For shading, lines, etc.
        self.title = "pytafast Analysis"
        self.theme = "plotly_white"
        self._has_main_price = False

    def set_theme(self, theme_name):
        """Set the chart theme (light, dark, ggplot2, seaborn)."""
        if theme_name in self.THEMES:
            self.theme = self.THEMES[theme_name]
        else:
            self.theme = theme_name  # Allow direct plotly template names
        return self

    # --- Internal Helpers ---
    def _add_overlay(self, y, name, color=None, width=1.5, dash=None):
        trace = go.Scatter(
            x=self.dt,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash),
        )
        self.main_traces.append(trace)
        return self

    def _add_subplot(self, traces, name, height=0.2, yrange=None):
        self.subplots.append(
            {"traces": traces, "height": height, "name": name, "yrange": yrange}
        )
        return self

    # --- Core Plots ---
    def add_candlestick(self, name="Price"):
        is_dark = "dark" in self.theme
        inc = "#26a69a" if not is_dark else "#00ffad"
        dec = "#ef5350" if not is_dark else "#ff5e5e"

        self.main_traces.append(
            go.Candlestick(
                x=self.dt,
                open=self.O,
                high=self.H,
                low=self.L,
                close=self.C,
                name=name,
                increasing_line_color=inc,
                decreasing_line_color=dec,
                increasing_fillcolor=inc,
                decreasing_fillcolor=dec,
            )
        )
        self._has_main_price = True
        return self

    def add_line(self, name="Close Price", color=None):
        if color is None:
            color = (
                "#1f77b4" if "white" in self.theme or "light" in self.theme else "white"
            )
        self._add_overlay(self.C, name, color, width=2)
        self._has_main_price = True
        return self

    def add_volume(self, height=0.15):
        if self.V is None:
            return self
        is_dark = "dark" in self.theme
        inc = "#26a69a" if not is_dark else "#00ffad"
        dec = "#ef5350" if not is_dark else "#ff5e5e"
        colors = np.where(self.C >= self.O, inc, dec)
        return self._add_subplot(
            [
                go.Bar(
                    x=self.dt,
                    y=self.V,
                    name="Volume",
                    marker_color=colors,
                    opacity=0.5,
                )
            ],
            "Volume",
            height,
        )

    # --- Generic Indicator Interface ---
    def add_indicator(self, indicator_fn, *args, plotter=None, **kwargs):
        """
        Generic API to compute and add an indicator to the chart.
        If `plotter` is not specified, it will look up the default plotter for the
        indicator in the registry.
        """
        # 1. Resolve indicator function name
        if indicator_fn is not None:
            if isinstance(indicator_fn, str):
                import pytafast
                name = indicator_fn.upper()
                indicator_fn = getattr(pytafast, name)
            else:
                name = indicator_fn.__name__.upper()
        else:
            name = ""

        # 2. Look up config in registry
        config = INDICATOR_REGISTRY.get(name, {})
        plotter_cls = plotter or config.get("plotter", LinePlotter)
        
        # 3. Merge default kwargs with user kwargs
        merged_kwargs = {**config.get("default_kwargs", {}), **kwargs}

        # 4. Instantiate plotter and plot
        plotter_instance = plotter_cls(indicator_fn, *args, **merged_kwargs)
        plotter_instance.plot(self)
        return self

    def __getattr__(self, name):
        """
        Dynamically route chainable add_<indicator> calls to add_indicator.
        This provides perfect backward compatibility with existing code.
        """
        if name.startswith("add_"):
            indicator_name = name[4:].upper()
            
            import pytafast
            config = INDICATOR_REGISTRY.get(indicator_name, {})
            
            func = config.get("indicator_fn")
            if func is None:
                func_name = config.get("function", indicator_name)
                func = getattr(pytafast, func_name, None)
            
            if indicator_name in INDICATOR_REGISTRY or func is not None:
                plotter_cls = config.get("plotter", LinePlotter)
                def wrapper(*args, **kwargs):
                    return self.add_indicator(func, *args, plotter=plotter_cls, **kwargs)
                return wrapper

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # --- Shapes and Annotation Overlays ---
    def add_shading(self, start, end, color="rgba(128, 128, 128, 0.2)"):
        self.shapes.append(
            {"type": "rect", "x0": start, "x1": end, "color": color, "axis": "x"}
        )
        return self

    def add_hline(self, y, color="gray", dash="dash"):
        self.shapes.append(
            {
                "type": "line",
                "y0": y,
                "y1": y,
                "color": color,
                "dash": dash,
                "axis": "y",
            }
        )
        return self

    def add_vline(self, x, color="gray", dash="dash"):
        self.shapes.append(
            {
                "type": "line",
                "x0": x,
                "x1": x,
                "color": color,
                "dash": dash,
                "axis": "x",
            }
        )
        return self

    def add_last(self, color="red"):
        last_price = self.C[-1]
        self.add_hline(last_price, color=color, dash="dash")
        self.shapes.append(
            {
                "type": "text",
                "x": self.dt.iloc[-1],
                "y": last_price,
                "text": f"{last_price:.2f}",
                "color": color,
            }
        )
        return self

    def add_points(self, x, y, name="Points", color="black", symbol="circle", size=8):
        if not isinstance(x, (list, np.ndarray, pd.Series)):
            x = [x]
        if not isinstance(y, (list, np.ndarray, pd.Series)):
            y = [y]
        self.main_traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(color=color, symbol=symbol, size=size),
            )
        )
        return self

    def add_text(self, x, y, text, color="black", position="top center"):
        if not isinstance(x, (list, np.ndarray, pd.Series)):
            x = [x]
        if not isinstance(y, (list, np.ndarray, pd.Series)):
            y = [y]
        if not isinstance(text, (list, np.ndarray, pd.Series)):
            text = [text]
        self.main_traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=text,
                textposition=position,
                showlegend=False,
                textfont=dict(color=color),
            )
        )
        return self

    # --- Render Engine ---
    def render(self):
        if not self._has_main_price:
            self.add_candlestick()
        n_subplots = len(self.subplots)
        sub_heights = [sp["height"] for sp in self.subplots]

        total_sub_weight = sum(sub_heights)
        main_weight = max(0.4, 1.0 - total_sub_weight - (0.02 * n_subplots))

        total_weight = main_weight + total_sub_weight
        row_heights = [main_weight / total_weight] + [
            h / total_weight for h in sub_heights
        ]

        fig = make_subplots(
            rows=1 + n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)

        for t in self.main_traces:
            fig.add_trace(t, row=1, col=1)
        for i, sp in enumerate(self.subplots):
            row_num = i + 2
            for t in sp["traces"]:
                fig.add_trace(t, row=row_num, col=1)
            fig.update_yaxes(title_text=sp["name"], row=row_num, col=1)
            if sp.get("yrange"):
                fig.update_yaxes(range=sp["yrange"], row=row_num, col=1)

        for s in self.shapes:
            if s.get("type") == "text":
                fig.add_annotation(
                    x=s["x"],
                    y=s["y"],
                    text=s["text"],
                    showarrow=True,
                    arrowhead=1,
                    ax=40,
                    ay=0,
                    font=dict(color=s["color"], size=12),
                    bgcolor="white",
                    bordercolor=s["color"],
                    row=1,
                    col=1,
                )
                continue

            if s["axis"] == "x":
                if s["type"] == "rect":
                    fig.add_vrect(
                        x0=s["x0"],
                        x1=s["x1"],
                        fillcolor=s["color"],
                        opacity=1.0,
                        layer="below",
                        line_width=0,
                        row=1,
                        col=1,
                    )
                else:
                    for r in range(1, 2 + n_subplots):
                        fig.add_vline(
                            x=s["x0"],
                            line=dict(color=s["color"], dash=s["dash"]),
                            row=r,
                            col=1,
                        )
            elif s["axis"] == "y":
                fig.add_hline(
                    y=s["y0"],
                    line=dict(color=s["color"], dash=s["dash"]),
                    row=1,
                    col=1,
                )

        fig.update_layout(
            title=self.title,
            template=self.theme,
            xaxis_rangeslider_visible=False,
            height=600 + (n_subplots * 150),
            margin=dict(l=60, r=40, t=60, b=40),
            hovermode="x unified",
            hoversubplots="axis",
        )
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="dash",
            spikecolor="rgba(128, 128, 128, 0.5)",
        )
        fig.update_yaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="dash",
            spikecolor="rgba(128, 128, 128, 0.5)",
        )
        fig.update_xaxes(rangeslider=dict(visible=True), row=1 + n_subplots, col=1)
        return fig

    def show(self):
        self.render().show()

    def save_html(self, filename="chart.html"):
        self.render().write_html(filename)

    def save_image(self, filename="chart.png", w=1200, h=800):
        if importlib.util.find_spec("kaleido") is None:
            raise ImportError(
                "The 'kaleido' package is required for saving images. "
                "Install it with 'pip install kaleido'."
            )

        import warnings
        f = self.render()
        f.update_layout(width=w, height=h)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
            )
            f.write_image(filename)
        print(f"Chart image saved to {filename}")
