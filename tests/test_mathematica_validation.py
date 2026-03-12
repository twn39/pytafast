import os
import pytest
import numpy as np
import pandas as pd
try:
    from wolframclient.evaluation import WolframLanguageSession
    from wolframclient.language import wl, wlexpr
    HAS_WOLFRAM = True
except ImportError:
    HAS_WOLFRAM = False

import pytafast

# Skip if in CI or Wolfram is not available
def _should_skip():
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        return True
    if not HAS_WOLFRAM:
        return True
    import shutil
    return not (os.path.exists("/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel") or 
                shutil.which("WolframKernel") or 
                shutil.which("wolframscript"))

pytestmark = pytest.mark.skipif(_should_skip(), reason="Wolfram Engine not available or running in CI")

@pytest.fixture(scope="session")
def wl_session():
    session = WolframLanguageSession(kernel="/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel", startup_timeout=60)
    session.start()
    yield session
    session.terminate()

@pytest.fixture(scope="session")
def sample_data(wl_session):
    print("\nFetching live data from Wolfram FinancialData...")
    wl_session.evaluate(wlexpr("""
        Global`FetchTicker[ticker_] := Module[{raw},
            raw = FinancialData[ticker, "OHLCV", {Today - Quantity[2, "Years"], Today}];
            Map[{
                DateString[#[[1]], "ISODate"], 
                QuantityMagnitude[#[[2, 1]]], 
                QuantityMagnitude[#[[2, 2]]], 
                QuantityMagnitude[#[[2, 3]]], 
                QuantityMagnitude[#[[2, 4]]], 
                QuantityMagnitude[#[[2, 5]]]
            } &, raw["Path"]]
        ];
    """))
    raw_list = wl_session.evaluate(wl.Global.FetchTicker("AAPL"))
    df = pd.DataFrame(raw_list, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(np.float64)
    return df

def compare_results(py_res, wl_res, rtol=1e-4, atol=1e-8, name="Indicator", skip_head=0):
    if wl_res is None:
        pytest.fail(f"{name}: Wolfram returned None.")
    wl_res = np.array(wl_res, dtype=np.float64)
    min_len = min(len(py_res), len(wl_res))
    py_tail = py_res[-min_len:]
    wl_tail = wl_res[-min_len:]
    if skip_head > 0:
        py_tail = py_tail[skip_head:]
        wl_tail = wl_tail[skip_head:]
    mask = ~np.isnan(py_tail) & ~np.isnan(wl_tail)
    if not np.any(mask):
        pytest.fail(f"{name}: No overlapping valid data.")
    np.testing.assert_allclose(py_tail[mask], wl_tail[mask], rtol=rtol, atol=atol, err_msg=f"Mismatch in {name}")

# --- Algorithmic Alignment Tests using Native Mathematica Functions ---
# Note: Mathematica uses slightly different initialization logic for exponential
# functions (like EMA, Wilder's Smoothing). We skip the burn-in period to test convergence.

def _get_wl_indicator(wl_session, sample_data, indicator_name, period, inputs="Close"):
    # Send data to WL. If inputs is 'OHLCV', we group it for TimeSeries.
    data_list = sample_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].values.tolist()
    
    eval_expr = wlexpr('''
    Function[{data, indName, period, inputs}, 
        Module[{tsGroup, tsTarget, ind},
            tsGroup = TimeSeries[Map[{#[[1]], Drop[#, 1]} &, data]];
            If[inputs == "OHLCV",
                If[period === Null,
                    ind = FinancialIndicator[indName][tsGroup],
                    ind = FinancialIndicator[indName, period][tsGroup]
                ],
                (* Default to Close price which is component 4 of {O,H,L,C,V} *)
                tsTarget = TimeSeriesMap[#[[4]] &, tsGroup];
                If[period === Null,
                    ind = FinancialIndicator[indName][tsTarget],
                    ind = FinancialIndicator[indName, period][tsTarget]
                ]
            ];
            ind["Values"]
        ]
    ]
    ''')
    res = wl_session.evaluate(eval_expr(data_list, indicator_name, period, inputs))
    return res

def test_dema_alignment(wl_session, sample_data):
    period = 30
    py_res = pytafast.DEMA(sample_data['Close'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "DoubleExponentialMovingAverage", period, "Close")
    compare_results(py_res, wl_res, name="DEMA", skip_head=period*8, rtol=1e-4)

def test_tema_alignment(wl_session, sample_data):
    period = 30
    py_res = pytafast.TEMA(sample_data['Close'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "TripleExponentialMovingAverage", period, "Close")
    compare_results(py_res, wl_res, name="TEMA", skip_head=period*8, rtol=1e-4)

def test_wma_alignment(wl_session, sample_data):
    period = 30
    # WMA doesn't have exponential decay, so it should match exactly without large skip
    py_res = pytafast.WMA(sample_data['Close'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "WeightedMovingAverage", period, "Close")
    compare_results(py_res, wl_res, name="WMA", skip_head=period)

def test_tr_alignment(wl_session, sample_data):
    py_res = pytafast.TRANGE(sample_data['High'].values, sample_data['Low'].values, sample_data['Close'].values)
    wl_res = _get_wl_indicator(wl_session, sample_data, "TrueRange", period=None, inputs="OHLCV")
    compare_results(py_res, wl_res, name="TR", skip_head=2, rtol=1e-4)

def test_atr_alignment(wl_session, sample_data):
    period = 14
    py_res = pytafast.ATR(sample_data['High'].values, sample_data['Low'].values, sample_data['Close'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "AverageTrueRange", period, "OHLCV")
    compare_results(py_res, wl_res, name="ATR", skip_head=period*10, rtol=1e-4)

def test_obv_alignment(wl_session, sample_data):
    py_res = pytafast.OBV(sample_data['Close'].values, sample_data['Volume'].values)
    wl_res = _get_wl_indicator(wl_session, sample_data, "OnBalanceVolume", period=None, inputs="OHLCV")
    compare_results(py_res, wl_res, name="OBV", skip_head=2, rtol=1e-4)

def test_medprice_alignment(wl_session, sample_data):
    py_res = pytafast.MEDPRICE(sample_data['High'].values, sample_data['Low'].values)
    wl_res = _get_wl_indicator(wl_session, sample_data, "MedianPrice", period=None, inputs="OHLCV")
    compare_results(py_res, wl_res, name="MEDPRICE", skip_head=2, rtol=1e-4)

def test_typprice_alignment(wl_session, sample_data):
    py_res = pytafast.TYPPRICE(sample_data['High'].values, sample_data['Low'].values, sample_data['Close'].values)
    wl_res = _get_wl_indicator(wl_session, sample_data, "TypicalPrice", period=None, inputs="OHLCV")
    compare_results(py_res, wl_res, name="TYPPRICE", skip_head=2, rtol=1e-4)

def test_wclprice_alignment(wl_session, sample_data):
    py_res = pytafast.WCLPRICE(sample_data['High'].values, sample_data['Low'].values, sample_data['Close'].values)
    wl_res = _get_wl_indicator(wl_session, sample_data, "WeightedClose", period=None, inputs="OHLCV")
    compare_results(py_res, wl_res, name="WCLPRICE", skip_head=2, rtol=1e-4)

def test_willr_alignment(wl_session, sample_data):
    period = 14
    py_res = pytafast.WILLR(sample_data['High'].values, sample_data['Low'].values, sample_data['Close'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "WilliamsPercentR", period, "OHLCV")
    compare_results(py_res, wl_res, name="WILLR", skip_head=period, rtol=1e-4)

def test_mfi_alignment(wl_session, sample_data):
    period = 14
    py_res = pytafast.MFI(sample_data['High'].values, sample_data['Low'].values, sample_data['Close'].values, sample_data['Volume'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "MoneyFlowIndex", period, "OHLCV")
    compare_results(py_res, wl_res, name="MFI", skip_head=period, rtol=1e-4)

def test_adx_alignment(wl_session, sample_data):
    period = 14
    py_res = pytafast.ADX(sample_data['High'].values, sample_data['Low'].values, sample_data['Close'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "AverageDirectionalMovementIndex", period, "OHLCV")
    compare_results(py_res, wl_res, name="ADX", skip_head=period*10, rtol=1e-4)

def test_ema_alignment(wl_session, sample_data):
    period = 14
    py_res = pytafast.EMA(sample_data['Close'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "ExponentialMovingAverage", period, "Close")
    compare_results(py_res, wl_res, name="EMA", skip_head=period*5, rtol=1e-4)

def test_rsi_alignment(wl_session, sample_data):
    period = 14
    py_res = pytafast.RSI(sample_data['Close'].values, timeperiod=period)
    wl_res = _get_wl_indicator(wl_session, sample_data, "RelativeStrengthIndex", period, "Close")
    # RSI also relies on exponential smoothing (Wilder's), so we need burn-in
    compare_results(py_res, wl_res, name="RSI", skip_head=period*10, rtol=1e-4)
