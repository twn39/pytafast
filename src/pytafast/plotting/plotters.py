# src/pytafast/plotting/plotters.py
import inspect
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class BasePlotter:
    def __init__(self, indicator_fn, *args, **kwargs):
        self.indicator_fn = indicator_fn
        self.args = args
        self.kwargs = kwargs

    def _map_inputs(self, chart):
        """Map chart fields (O, H, L, C, V) to the parameters of indicator_fn."""
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        
        inputs = []
        if "inOpen" in params and "inHigh" in params and "inLow" in params and "inClose" in params:
            inputs = [chart.O, chart.H, chart.L, chart.C]
        elif "inHigh" in params and "inLow" in params and "inClose" in params:
            if "inVolume" in params:
                inputs = [chart.H, chart.L, chart.C, chart.V]
            else:
                inputs = [chart.H, chart.L, chart.C]
        elif "inHigh" in params and "inLow" in params:
            inputs = [chart.H, chart.L]
        elif "inReal" in params:
            inputs = [chart.C]
        elif "inReal0" in params and "inReal1" in params:
            inputs = [chart.C, chart.V]
        elif "inReal" in params and "inVolume" in params:
            inputs = [chart.C, chart.V]
        else:
            inputs = [chart.C]
        return inputs

    def _get_indicator_kwargs(self, params):
        """Map legacy parameter names (n, sd, f, s, sig, k, d, accel, max_step) to signature names and filter."""
        param_map = {
            "n": "timeperiod",
            "sd": ("nbdevup", "nbdevdn"),
            "f": "fastperiod",
            "s": "slowperiod",
            "sig": "signalperiod",
            "k": "fastk_period",
            "d": "slowd_period",
            "accel": "acceleration",
            "max_step": "maximum",
        }
        mapped_kwargs = {}
        for k, v in self.kwargs.items():
            if k in param_map:
                target = param_map[k]
                if isinstance(target, tuple):
                    for t in target:
                        mapped_kwargs[t] = v
                else:
                    mapped_kwargs[target] = v
            else:
                mapped_kwargs[k] = v
                
        # Keep only kwargs matching the signature parameter names
        indicator_kwargs = {}
        for k, v in mapped_kwargs.items():
            if k in params:
                indicator_kwargs[k] = v
        return indicator_kwargs


class LinePlotter(BasePlotter):
    """Draws a single indicator line."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        y = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        name = self.kwargs.get("name", f"{self.indicator_fn.__name__}({self.args[0] if self.args else ''})")
        color = self.kwargs.get("color", None)
        width = self.kwargs.get("width", 1.5)
        dash = self.kwargs.get("dash", None)
        subplot = self.kwargs.get("subplot", False)
        height = self.kwargs.get("height", 0.2)
        yrange = self.kwargs.get("yrange", None)
        
        if subplot:
            trace = go.Scatter(x=chart.dt, y=y, mode="lines", name=name, line=dict(color=color, width=width, dash=dash))
            chart._add_subplot([trace], name, height, yrange)
        else:
            chart._add_overlay(y, name, color=color, width=width, dash=dash)


class BandPlotter(BasePlotter):
    """Draws upper, middle, and lower bands with standard tonexty fill."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        upper, middle, lower = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        name = self.kwargs.get("name", self.indicator_fn.__name__)
        fillcolor = self.kwargs.get("fillcolor", "rgba(173, 216, 230, 0.15)")
        linecolor = self.kwargs.get("linecolor", "rgba(173, 216, 230, 0.4)")
        mid_color = self.kwargs.get("mid_color", "rgba(128, 128, 128, 0.5)")
        
        chart._add_overlay(upper, f"{name} Upper", linecolor, 1)
        chart._add_overlay(lower, f"{name} Lower", linecolor, 1)
        chart.main_traces[-1].fill = "tonexty"
        chart.main_traces[-1].fillcolor = fillcolor
        chart._add_overlay(middle, f"{name} Mid", mid_color, 1, "dash")


class OscillatorPlotter(BasePlotter):
    """Draws an oscillator line in a subplot with horizontal lines (e.g. RSI 70/30)."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        y = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        name = self.kwargs.get("name", f"{self.indicator_fn.__name__}")
        color = self.kwargs.get("color", None)
        height = self.kwargs.get("height", 0.2)
        yrange = self.kwargs.get("yrange", None)
        hlines = self.kwargs.get("hlines", [])
        
        trace = go.Scatter(x=chart.dt, y=y, mode="lines", name=name, line=dict(color=color))
        traces = [trace]
        for hl in hlines:
            traces.append(
                go.Scatter(
                    x=chart.dt, y=[hl] * len(chart.dt),
                    showlegend=False,
                    line=dict(color="rgba(128, 128, 128, 0.5)", dash="dash", width=1)
                )
            )
        chart._add_subplot(traces, name, height, yrange)


class MACDPlotter(BasePlotter):
    """Draws MACD, signal line, and histogram bars."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        macd, signal, hist = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.2)
        inc = "#26a69a"
        dec = "#ef5350"
        
        traces = [
            go.Bar(x=chart.dt, y=hist, name="MACD Hist", marker_color=np.where(hist >= 0, inc, dec)),
            go.Scatter(x=chart.dt, y=macd, name="MACD"),
            go.Scatter(x=chart.dt, y=signal, name="Signal")
        ]
        chart._add_subplot(traces, "MACD", height)


class StochasticPlotter(BasePlotter):
    """Draws slowk and slowd stochastic lines in a subplot."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        slowk, slowd = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.2)
        traces = [
            go.Scatter(x=chart.dt, y=slowk, name="%K"),
            go.Scatter(x=chart.dt, y=slowd, name="%D")
        ]
        chart._add_subplot(traces, "Stoch", height, [0, 100])


class AroonPlotter(BasePlotter):
    """Draws Aroon Up and Aroon Down lines."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        aroondown, aroonup = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.2)
        traces = [
            go.Scatter(x=chart.dt, y=aroonup, name="Aroon Up", line=dict(color="green")),
            go.Scatter(x=chart.dt, y=aroondown, name="Aroon Down", line=dict(color="red"))
        ]
        chart._add_subplot(traces, "Aroon", height, [0, 100])


class ADXPlotter(BasePlotter):
    """Draws ADX, +DI, and -DI lines."""
    def plot(self, chart):
        import pytafast
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        n = indicator_kwargs.get("timeperiod", 14)
        
        adx = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        pdi = pytafast.PLUS_DI(*inputs, *self.args, **indicator_kwargs)
        mdi = pytafast.MINUS_DI(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.2)
        traces = [
            go.Scatter(x=chart.dt, y=adx, name=f"ADX({n})", line=dict(color="black")),
            go.Scatter(x=chart.dt, y=pdi, name=f"+DI({n})", line=dict(color="green", dash="dot")),
            go.Scatter(x=chart.dt, y=mdi, name=f"-DI({n})", line=dict(color="red", dash="dot"))
        ]
        chart._add_subplot(traces, "ADX", height)


class TDIPlotter(BasePlotter):
    """Draws Traders Dynamic Index (TDI)."""
    def plot(self, chart):
        import pytafast
        n = self.kwargs.get("n", 13)
        rsi_ma1 = self.kwargs.get("rsi_ma1", 2)
        rsi_ma2 = self.kwargs.get("rsi_ma2", 7)
        bb_n = self.kwargs.get("bb_n", 34)
        bb_sd = self.kwargs.get("bb_sd", 1.6185)
        height = self.kwargs.get("height", 0.25)
        
        rsi = pytafast.RSI(chart.C, n)
        rsi_price_line = pytafast.SMA(rsi, rsi_ma1)
        trade_signal_line = pytafast.SMA(rsi, rsi_ma2)
        
        # Volatility Band around RSI
        u, m, lower = pytafast.BBANDS(rsi, bb_n, bb_sd, bb_sd)
        
        traces = [
            go.Scatter(x=chart.dt, y=rsi_price_line, name="RSI Price Line", line=dict(color="green", width=1.5)),
            go.Scatter(x=chart.dt, y=trade_signal_line, name="Trade Signal", line=dict(color="red", width=1.5)),
            go.Scatter(x=chart.dt, y=m, name="Market Base Line", line=dict(color="orange", width=2)),
            go.Scatter(x=chart.dt, y=u, name="VB Upper", line=dict(color="rgba(0,0,255,0.2)", width=1)),
            go.Scatter(x=chart.dt, y=lower, name="VB Lower", line=dict(color="rgba(0,0,255,0.2)", width=1), fill="tonexty", fillcolor="rgba(0,0,255,0.05)")
        ]
        chart._add_subplot(traces, "TDI", height)


class KSTPlotter(BasePlotter):
    """Draws KST and KST Signal lines."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        k, s = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.2)
        traces = [
            go.Scatter(x=chart.dt, y=k, name="KST"),
            go.Scatter(x=chart.dt, y=s, name="Signal")
        ]
        chart._add_subplot(traces, "KST", height)


class SMIPlotter(BasePlotter):
    """Draws SMI (Stochastic Momentum Index) and Signal lines."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        smi, signal = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.2)
        traces = [
            go.Scatter(x=chart.dt, y=smi, name="SMI"),
            go.Scatter(x=chart.dt, y=signal, name="Signal", line=dict(dash="dash"))
        ]
        chart._add_subplot(traces, "SMI", height, [-100, 100])


class IchimokuPlotter(BasePlotter):
    """Draws Ichimoku Kinko Hyo (Cloud) on the main chart with shifted future index."""
    def plot(self, chart):
        import pytafast
        n1 = self.kwargs.get("n1", 9)
        n2 = self.kwargs.get("n2", 26)
        n3 = self.kwargs.get("n3", 52)
        
        t_high = pytafast.MAX(chart.H, n1)
        t_low = pytafast.MIN(chart.L, n1)
        tenkan = (t_high + t_low) / 2.0
        
        k_high = pytafast.MAX(chart.H, n2)
        k_low = pytafast.MIN(chart.L, n2)
        kijun = (k_high + k_low) / 2.0
        
        ssa = (tenkan + kijun) / 2.0
        
        b_high = pytafast.MAX(chart.H, n3)
        b_low = pytafast.MIN(chart.L, n3)
        ssb = (b_high + b_low) / 2.0
        
        dt_shifted = pd.to_datetime(chart.dt)
        delta = dt_shifted.diff().median()
        future_dt = [dt_shifted.iloc[-1] + (i * delta) for i in range(1, n2 + 1)]
        all_dt = np.concatenate([chart.dt.values, np.array(future_dt)])
        
        ssa_padded = np.full(len(all_dt), np.nan)
        ssb_padded = np.full(len(all_dt), np.nan)
        ssa_padded[n2 : n2 + len(ssa)] = ssa
        ssb_padded[n2 : n2 + len(ssb)] = ssb
        
        chart._add_overlay(tenkan, "Tenkan", "blue", 1)
        chart._add_overlay(kijun, "Kijun", "red", 1)
        
        chart.main_traces.append(
            go.Scatter(x=all_dt, y=ssa_padded, name="Senkou Span A", line=dict(color="rgba(0, 255, 0, 0.3)", width=1))
        )
        chart.main_traces.append(
            go.Scatter(x=all_dt, y=ssb_padded, name="Senkou Span B", line=dict(color="rgba(255, 0, 0, 0.3)", width=1), fill="tonexty", fillcolor="rgba(144, 238, 144, 0.1)")
        )


class ZigZagPlotter(BasePlotter):
    """Draws ZigZag line."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        zz = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        chart._add_overlay(zz, "ZigZag", "blue", 2, "dashdot")


class SARPlotter(BasePlotter):
    """Draws Parabolic SAR markers."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        s = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        color = "black" if "white" in chart.theme or "light" in chart.theme else "white"
        trace = go.Scatter(x=chart.dt, y=s, mode="markers", name="SAR", marker=dict(color=color, size=4))
        chart.main_traces.append(trace)


class EMVPlotter(BasePlotter):
    """Draws Ease of Movement (EMV) and Signal lines."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        v, sv = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        n = indicator_kwargs.get("timeperiod", 9)
        
        height = self.kwargs.get("height", 0.2)
        traces = [
            go.Scatter(x=chart.dt, y=v, name="EMV", line=dict(width=1), opacity=0.5),
            go.Scatter(x=chart.dt, y=sv, name=f"Signal({n})", line=dict(width=2))
        ]
        chart._add_subplot(traces, "EMV", height)


class CMFPlotter(BasePlotter):
    """Draws Chaikin Money Flow line."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        v = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.2)
        trace = go.Scatter(x=chart.dt, y=v, name="CMF")
        chart._add_subplot([trace], "CMF", height)


class OBVPlotter(BasePlotter):
    """Draws On-Balance Volume line with area fill."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        
        v = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.15)
        trace = go.Scatter(x=chart.dt, y=v, name="OBV", fill="tozeroy")
        chart._add_subplot([trace], "OBV", height)


class DPOPlotter(BasePlotter):
    """Draws Detrended Price Oscillator as colored positive/negative bars."""
    def plot(self, chart):
        inputs = self._map_inputs(chart)
        sig = inspect.signature(self.indicator_fn)
        params = list(sig.parameters.keys())
        indicator_kwargs = self._get_indicator_kwargs(params)
        n = indicator_kwargs.get("timeperiod", 10)
        
        v = self.indicator_fn(*inputs, *self.args, **indicator_kwargs)
        
        height = self.kwargs.get("height", 0.2)
        trace = go.Bar(x=chart.dt, y=v, name=f"DPO({n})", marker_color=np.where(v >= 0, "green", "red"))
        chart._add_subplot([trace], "DPO", height)


class CLVPlotter(BasePlotter):
    """Draws Close Location Value (CLV) line."""
    def plot(self, chart):
        denom = chart.H - chart.L
        v = np.where(denom != 0, ((chart.C - chart.L) - (chart.H - chart.C)) / denom, 0.0)
        
        height = self.kwargs.get("height", 0.15)
        trace = go.Scatter(x=chart.dt, y=v, name="CLV")
        chart._add_subplot([trace], "CLV", height)


class VolatilityPlotter(BasePlotter):
    """Draws Chaikin Volatility line."""
    def plot(self, chart):
        import pytafast
        n = self.kwargs.get("n", 10)
        hl = chart.H - chart.L
        ema_hl = pytafast.EMA(hl, n)
        v = np.full_like(ema_hl, np.nan)
        if len(ema_hl) > n:
            v[n:] = (ema_hl[n:] / ema_hl[:-n]) - 1.0
            
        height = self.kwargs.get("height", 0.2)
        trace = go.Scatter(x=chart.dt, y=v * 100, name=f"ChaikinVol({n})")
        chart._add_subplot([trace], "Chaikin Vol %", height)


class PatternsPlotter(BasePlotter):
    """Finds and labels all candlestick patterns automatically on the main chart."""
    def plot(self, chart):
        import pytafast
        patterns = [m for m in dir(pytafast) if m.startswith("CDL")]
        first_pattern = True
        
        for p_name in patterns:
            res = getattr(pytafast, p_name)(chart.O, chart.H, chart.L, chart.C)
            hit_idx = np.where(res != 0)[0]
            if len(hit_idx) > 0:
                chart.main_traces.append(
                    go.Scatter(
                        x=chart.dt.iloc[hit_idx],
                        y=chart.H[hit_idx] * 1.02,
                        mode="markers",
                        name="Candlestick Patterns" if first_pattern else p_name,
                        legendgroup="candlestick_patterns",
                        showlegend=first_pattern,
                        text=[p_name[3:]] * len(hit_idx),
                        hovertext=[f"{p_name[3:]} ({'Bullish' if res[idx] > 0 else 'Bearish'})" for idx in hit_idx],
                        hoverinfo="text",
                        marker=dict(symbol="triangle-down", size=8),
                    )
                )
                first_pattern = False


class EnvelopePlotter(BasePlotter):
    """Draws a Moving Average Envelope around SMA."""
    def plot(self, chart):
        import pytafast
        n = self.kwargs.get("n", 20)
        p = self.kwargs.get("p", 2.5)
        color = self.kwargs.get("color", "rgba(0, 0, 255, 0.1)")
        
        ma = pytafast.SMA(chart.C, n)
        u = ma * (1 + p / 100.0)
        lower = ma * (1 - p / 100.0)
        
        chart._add_overlay(u, "Env Upper", "rgba(0, 0, 255, 0.3)", 1, "dot")
        chart._add_overlay(lower, f"Env Lower", "rgba(0, 0, 255, 0.3)", 1, "dot")
        chart.main_traces[-1].fill = "tonexty"
        chart.main_traces[-1].fillcolor = color
