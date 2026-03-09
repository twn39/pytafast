import importlib.util

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pytafast


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
        # Determine colors based on theme
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
        colors = [inc if c >= o else dec for c, o in zip(self.C, self.O)]
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

    # --- Overlap (Main Chart) ---
    def add_sma(self, n=20, color="#f29d4b"):
        return self._add_overlay(pytafast.SMA(self.C, n), f"SMA({n})", color)

    def add_ema(self, n=20, color="#5d62b5"):
        return self._add_overlay(pytafast.EMA(self.C, n), f"EMA({n})", color)

    def add_wma(self, n=20, color="#33a02c"):
        return self._add_overlay(pytafast.WMA(self.C, n), f"WMA({n})", color)

    def add_dema(self, n=20, color="#e31a1c"):
        return self._add_overlay(pytafast.DEMA(self.C, n), f"DEMA({n})", color)

    def add_tema(self, n=20, color="#fb9a99"):
        return self._add_overlay(pytafast.TEMA(self.C, n), f"TEMA({n})", color)

    def add_kama(self, n=20, color="#6a3d9a"):
        return self._add_overlay(pytafast.KAMA(self.C, n), f"KAMA({n})", color)

    def add_hma(self, n=20, color="#b15928"):
        return self._add_overlay(pytafast.HMA(self.C, n), f"HMA({n})", color)

    def add_alma(self, n=9, offset=0.85, sigma=6.0):
        return self._add_overlay(
            pytafast.ALMA(self.C, n, offset, sigma), f"ALMA({n})", "#1f78b4"
        )

    def add_zlema(self, n=30):
        return self._add_overlay(pytafast.ZLEMA(self.C, n), f"ZLEMA({n})", "#ff7f00")

    def add_evwma(self, n=30):
        return self._add_overlay(
            pytafast.EVWMA(self.C, self.V, n), f"EVWMA({n})", "#b2df8a"
        )

    def add_sar(self, accel=0.02, max_step=0.2):
        s = pytafast.SAR(self.H, self.L, accel, max_step)
        color = "black" if "white" in self.theme or "light" in self.theme else "white"
        trace = go.Scatter(
            x=self.dt, y=s, mode="markers", name="SAR", marker=dict(color=color, size=4)
        )
        self.main_traces.append(trace)
        return self

    def add_bbands(self, n=20, sd=2.0, color="rgba(173, 216, 230, 0.15)"):
        u, m, low_val = pytafast.BBANDS(self.C, n, sd, sd)
        self._add_overlay(u, "BB Upper", "rgba(173, 216, 230, 0.4)", 1)
        self._add_overlay(low_val, "BB Lower", "rgba(173, 216, 230, 0.4)", 1)
        self.main_traces[-1].fill = "tonexty"
        self.main_traces[-1].fillcolor = color
        return self._add_overlay(m, "BB Mid", "rgba(128, 128, 128, 0.5)", 1, "dash")

    def add_keltner(self, n=20, mult=2.0):
        u, m, low_val = pytafast.keltnerChannels(self.H, self.L, self.C, n, mult)
        self._add_overlay(u, "Keltner Upper", "rgba(255,165,0,0.3)", 1)
        self._add_overlay(low_val, "Keltner Lower", "rgba(255,165,0,0.3)", 1)
        self.main_traces[-1].fill = "tonexty"
        self.main_traces[-1].fillcolor = "rgba(255,165,0,0.1)"
        return self._add_overlay(m, "Keltner Mid", "orange", 1, "dot")

    def add_donchian(self, n=10):
        u, m, low_val = pytafast.DonchianChannel(self.H, self.L, n)
        self._add_overlay(u, "Donchian High", "rgba(200,200,200,0.3)", 1)
        self._add_overlay(low_val, "Donchian Low", "rgba(200,200,200,0.3)", 1)
        self.main_traces[-1].fill = "tonexty"
        self.main_traces[-1].fillcolor = "rgba(200,200,200,0.05)"
        return self

    def add_zigzag(self, change=5.0, percent=True):
        zz = pytafast.ZIGZAG(self.H, self.L, change, percent)
        return self._add_overlay(zz, "ZigZag", "blue", 2, "dashdot")

    def add_envelope(self, n=20, p=2.5, color="rgba(0, 0, 255, 0.1)"):
        """Add Moving Average Envelope around SMA."""
        ma = pytafast.SMA(self.C, n)
        u = ma * (1 + p / 100)
        low_val = ma * (1 - p / 100)
        self._add_overlay(u, "Env Upper", "rgba(0, 0, 255, 0.3)", 1, "dot")
        self._add_overlay(low_val, "Env Lower", "rgba(0, 0, 255, 0.3)", 1, "dot")
        self.main_traces[-1].fill = "tonexty"
        self.main_traces[-1].fillcolor = color
        return self

    def add_shading(self, start, end, color="rgba(128, 128, 128, 0.2)"):
        """Highlight a vertical time region (start/end can be dates or indices)."""
        self.shapes.append(
            {"type": "rect", "x0": start, "x1": end, "color": color, "axis": "x"}
        )
        return self

    def add_hline(self, y, color="gray", dash="dash"):
        """Add a horizontal line to the main chart."""
        self.shapes.append(
            {"type": "line", "y0": y, "y1": y, "color": color, "dash": dash, "axis": "y"}
        )
        return self

    def add_vline(self, x, color="gray", dash="dash"):
        """Add a vertical line to the main chart."""
        self.shapes.append(
            {"type": "line", "x0": x, "x1": x, "color": color, "dash": dash, "axis": "x"}
        )
        return self

    # --- Momentum (Subplots) ---
    def add_rsi(self, n=14, height=0.2):
        v = pytafast.RSI(self.C, n)
        color = "#9467bd"
        traces = [
            go.Scatter(x=self.dt, y=v, name=f"RSI({n})", line=dict(color=color)),
            go.Scatter(
                x=self.dt,
                y=[70] * len(self.dt),
                showlegend=False,
                line=dict(color="red", dash="dash"),
            ),
            go.Scatter(
                x=self.dt,
                y=[30] * len(self.dt),
                showlegend=False,
                line=dict(color="green", dash="dash"),
            ),
        ]
        return self._add_subplot(traces, "RSI", height, [0, 100])

    def add_macd(self, f=12, s=26, sig=9, height=0.2):
        m, signal, h = pytafast.MACD(self.C, f, s, sig)
        inc = "#26a69a"
        dec = "#ef5350"
        traces = [
            go.Bar(
                x=self.dt,
                y=h,
                name="MACD Hist",
                marker_color=[inc if x >= 0 else dec for x in h],
            ),
            go.Scatter(x=self.dt, y=m, name="MACD"),
            go.Scatter(x=self.dt, y=signal, name="Signal"),
        ]
        return self._add_subplot(traces, "MACD", height)

    def add_stoch(self, k=5, d=3, height=0.2):
        sk, sd = pytafast.STOCH(self.H, self.L, self.C, k, d)
        return self._add_subplot(
            [
                go.Scatter(x=self.dt, y=sk, name="%K"),
                go.Scatter(x=self.dt, y=sd, name="%D"),
            ],
            "Stoch",
            height,
            [0, 100],
        )

    def add_willr(self, n=14, height=0.2):
        return self._add_subplot(
            [
                go.Scatter(
                    x=self.dt, y=pytafast.WILLR(self.H, self.L, self.C, n), name="WILLR"
                )
            ],
            "Williams %R",
            height,
            [-100, 0],
        )

    def add_cci(self, n=14, height=0.2):
        return self._add_subplot(
            [
                go.Scatter(
                    x=self.dt, y=pytafast.CCI(self.H, self.L, self.C, n), name="CCI"
                )
            ],
            "CCI",
            height,
        )

    def add_mfi(self, n=14, height=0.2):
        return self._add_subplot(
            [
                go.Scatter(
                    x=self.dt,
                    y=pytafast.MFI(self.H, self.L, self.C, self.V, n),
                    name="MFI",
                )
            ],
            "MFI",
            height,
            [0, 100],
        )

    def add_cmf(self, n=20, height=0.2):
        return self._add_subplot(
            [
                go.Scatter(
                    x=self.dt,
                    y=pytafast.CMF(self.H, self.L, self.C, self.V, n),
                    name="CMF",
                )
            ],
            "CMF",
            height,
        )

    def add_kst(self, height=0.2):
        k, s = pytafast.KST(self.C)
        return self._add_subplot(
            [
                go.Scatter(x=self.dt, y=k, name="KST"),
                go.Scatter(x=self.dt, y=s, name="Signal"),
            ],
            "KST",
            height,
        )

    def add_obv(self, height=0.15):
        v = pytafast.OBV(self.C, self.V)
        return self._add_subplot(
            [go.Scatter(x=self.dt, y=v, name="OBV", fill="tozeroy")], "OBV", height
        )

    def add_aroon(self, n=14, height=0.2):
        dn, up = pytafast.AROON(self.H, self.L, n)
        return self._add_subplot(
            [
                go.Scatter(x=self.dt, y=up, name="Aroon Up", line=dict(color="green")),
                go.Scatter(x=self.dt, y=dn, name="Aroon Down", line=dict(color="red")),
            ],
            "Aroon",
            height,
            [0, 100],
        )

    def add_smi(self, n=13, nFast=2, nSlow=25, nSig=9, height=0.2):
        smi, signal = pytafast.SMI(self.H, self.L, self.C, n, nFast, nSlow, nSig)
        return self._add_subplot(
            [
                go.Scatter(x=self.dt, y=smi, name="SMI"),
                go.Scatter(x=self.dt, y=signal, name="Signal", line=dict(dash="dash")),
            ],
            "SMI",
            height,
            [-100, 100],
        )

    def add_roc(self, n=10, height=0.2):
        v = pytafast.ROC(self.C, n)
        return self._add_subplot(
            [
                go.Scatter(x=self.dt, y=v, name=f"ROC({n})"),
                go.Scatter(
                    x=self.dt,
                    y=[0] * len(self.dt),
                    showlegend=False,
                    line=dict(color="gray", dash="dot"),
                ),
            ],
            "ROC",
            height,
        )

    def add_cmo(self, n=14, height=0.2):
        v = pytafast.CMO(self.C, n)
        return self._add_subplot(
            [
                go.Scatter(x=self.dt, y=v, name=f"CMO({n})"),
                go.Scatter(
                    x=self.dt,
                    y=[50] * len(self.dt),
                    showlegend=False,
                    line=dict(color="red", dash="dash"),
                ),
                go.Scatter(
                    x=self.dt,
                    y=[-50] * len(self.dt),
                    showlegend=False,
                    line=dict(color="green", dash="dash"),
                ),
            ],
            "CMO",
            height,
            [-100, 100],
        )

    def add_emv(self, n=9, height=0.2):
        v, sv = pytafast.EMV(self.H, self.L, self.V, n)
        return self._add_subplot(
            [
                go.Scatter(x=self.dt, y=v, name="EMV", line=dict(width=1), opacity=0.5),
                go.Scatter(x=self.dt, y=sv, name=f"Signal({n})", line=dict(width=2)),
            ],
            "EMV",
            height,
        )

    def add_trix(self, n=30, height=0.2):
        v = pytafast.TRIX(self.C, n)
        return self._add_subplot(
            [go.Scatter(x=self.dt, y=v, name=f"TRIX({n})")], "TRIX", height
        )

    def add_dpo(self, n=10, height=0.2):
        v = pytafast.DPO(self.C, n)
        return self._add_subplot(
            [
                go.Bar(
                    x=self.dt,
                    y=v,
                    name=f"DPO({n})",
                    marker_color=["green" if x >= 0 else "red" for x in v],
                )
            ],
            "DPO",
            height,
        )

    # --- Volatility (Subplots) ---
    def add_atr(self, n=14, height=0.15):
        return self._add_subplot(
            [
                go.Scatter(
                    x=self.dt, y=pytafast.ATR(self.H, self.L, self.C, n), name="ATR"
                )
            ],
            "ATR",
            height,
        )

    def add_patterns(self):
        """Automatically find and label all recognized candlestick patterns."""
        patterns = [m for m in dir(pytafast) if m.startswith("CDL")]
        for p_name in patterns:
            res = getattr(pytafast, p_name)(self.O, self.H, self.L, self.C)
            # Find indices where pattern is detected (100 or -100)
            hit_idx = np.where(res != 0)[0]
            if len(hit_idx) > 0:
                self.main_traces.append(
                    go.Scatter(
                        x=self.dt.iloc[hit_idx],
                        y=self.H[hit_idx] * 1.02,
                        mode="markers+text",
                        name=p_name,
                        text=[p_name[3:]] * len(hit_idx),
                        textposition="top center",
                        marker=dict(symbol="triangle-down", size=8),
                    )
                )
        return self

    # --- Render Engine ---
    def render(self):
        if not self._has_main_price:
            self.add_candlestick()
        n_subplots = len(self.subplots)
        sub_heights = [sp["height"] for sp in self.subplots]
        
        # Calculate total subplot weight
        total_sub_weight = sum(sub_heights)
        # Ensure main chart is at least 40% of the total height if possible, 
        # otherwise scale everything down
        main_weight = max(0.4, 1.0 - total_sub_weight - (0.02 * n_subplots))
        
        # Normalize weights to sum to ~1.0
        total_weight = main_weight + total_sub_weight
        row_heights = [main_weight / total_weight] + [h / total_weight for h in sub_heights]

        fig = make_subplots(
            rows=1 + n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,  # Tighter spacing
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

        # Apply Shapes
        for s in self.shapes:
            if s["axis"] == "x":
                if s["type"] == "rect":
                    # Shading typically only on main price chart
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
                else:  # vline - apply to all rows
                    for r in range(1, 2 + n_subplots):
                        fig.add_vline(
                            x=s["x0"],
                            line=dict(color=s["color"], dash=s["dash"]),
                            row=r,
                            col=1,
                        )
            elif s["axis"] == "y":
                # Horizontal line only on main chart
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
        )
        fig.update_xaxes(rangeslider=dict(visible=True), row=1 + n_subplots, col=1)
        return fig

    def show(self):
        self.render().show()

    def save_html(self, filename="chart.html"):
        self.render().write_html(filename)

    def save_image(self, filename="chart.png", w=1200, h=800):
        """
        Renders and saves the chart to a static image (PNG, JPG, PDF, SVG).
        Requires 'kaleido' package (pip install kaleido).
        """
        if importlib.util.find_spec("kaleido") is None:
            raise ImportError(
                "The 'kaleido' package is required for saving images. "
                "Install it with 'pip install kaleido'."
            )

        f = self.render()
        f.update_layout(width=w, height=h)
        f.write_image(filename)
        print(f"Chart image saved to {filename}")
