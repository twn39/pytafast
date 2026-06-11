import pandas as pd
import numpy as np
import pytafast
from pytafast import MAType

import os

# Load data
data_file = os.getenv("DATA_FILE", "data/berkshire_1y.csv")
df = pd.read_csv(data_file)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

O = df["Open"].values
H = df["High"].values
L = df["Low"].values
C = df["Close"].values
V = df["Volume"].values.astype(float)

results = {"Date": df["Date"].values}


def safe_compute(name, func, *args, **kwargs):
    try:
        out = func(*args, **kwargs)
        if isinstance(out, tuple):
            for i, val in enumerate(out):
                results[f"{name}_{i}"] = val
        else:
            results[name] = out
    except Exception as e:
        print(f"Error computing {name}: {e}")


# --- Overlap ---
safe_compute("SMA", pytafast.SMA, C, 30)
safe_compute("EMA", pytafast.EMA, C, 30)
safe_compute("WMA", pytafast.WMA, C, 30)
safe_compute("DEMA", pytafast.DEMA, C, 30)
safe_compute("TEMA", pytafast.TEMA, C, 30)
safe_compute("TRIMA", pytafast.TRIMA, C, 30)
safe_compute("KAMA", pytafast.KAMA, C, 30)
safe_compute("MIDPOINT", pytafast.MIDPOINT, C, 14)
safe_compute("MIDPRICE", pytafast.MIDPRICE, H, L, 14)
safe_compute("SAR", pytafast.SAR, H, L)
safe_compute("T3", pytafast.T3, C, 5)
safe_compute("BBANDS", pytafast.BBANDS, C, 5, 2, 2, MAType.SMA)
safe_compute("MAMA", pytafast.MAMA, C)

# --- Momentum ---
safe_compute("RSI", pytafast.RSI, C, 14)
safe_compute("MACD", pytafast.MACD, C, fastperiod=12, slowperiod=26, signalperiod=9)
safe_compute("MOM", pytafast.MOM, C, 10)
safe_compute("ROC", pytafast.ROC, C, 10)
safe_compute("ROCP", pytafast.ROCP, C, 10)
safe_compute("ROCR", pytafast.ROCR, C, 10)
safe_compute("ROCR100", pytafast.ROCR100, C, 10)
safe_compute("TRIX", pytafast.TRIX, C, 30)
safe_compute("ADX", pytafast.ADX, H, L, C, 14)
safe_compute("ADXR", pytafast.ADXR, H, L, C, 14)
safe_compute("DX", pytafast.DX, H, L, C, 14)
safe_compute("PLUS_DI", pytafast.PLUS_DI, H, L, C, 14)
safe_compute("MINUS_DI", pytafast.MINUS_DI, H, L, C, 14)
safe_compute("PLUS_DM", pytafast.PLUS_DM, H, L, 14)
safe_compute("MINUS_DM", pytafast.MINUS_DM, H, L, 14)
safe_compute("CCI", pytafast.CCI, H, L, C, 14)
safe_compute("MFI", pytafast.MFI, H, L, C, V, 14)
safe_compute("WILLR", pytafast.WILLR, H, L, C, 14)
safe_compute("ULTOSC", pytafast.ULTOSC, H, L, C)
safe_compute("BOP", pytafast.BOP, O, H, L, C)
safe_compute("CMO", pytafast.CMO, C, 14)
safe_compute("APO", pytafast.APO, C)
safe_compute("PPO", pytafast.PPO, C)
safe_compute("AROON", pytafast.AROON, H, L, 14)
safe_compute("AROONOSC", pytafast.AROONOSC, H, L, 14)
safe_compute("STOCH", pytafast.STOCH, H, L, C)
safe_compute("STOCHF", pytafast.STOCHF, H, L, C)
safe_compute("STOCHRSI", pytafast.STOCHRSI, C)

# --- Volatility ---
safe_compute("ATR", pytafast.ATR, H, L, C, 14)
safe_compute("NATR", pytafast.NATR, H, L, C, 14)
safe_compute("TRANGE", pytafast.TRANGE, H, L, C)
safe_compute("STDDEV", pytafast.STDDEV, C, 5, 1.0)

# --- Volume ---
safe_compute("OBV", pytafast.OBV, C, V)
safe_compute("AD", pytafast.AD, H, L, C, V)
safe_compute("ADOSC", pytafast.ADOSC, H, L, C, V)


# Fix for manual ones
def fix_manual():
    # OBV
    results["OBV"] = pytafast.OBV(C, V)
    # AD
    results["AD"] = pytafast.AD(H, L, C, V)
    # ADOSC
    results["ADOSC"] = pytafast.ADOSC(H, L, C, V)


fix_manual()

# --- Price ---
safe_compute("AVGPRICE", pytafast.AVGPRICE, O, H, L, C)
safe_compute("MEDPRICE", pytafast.MEDPRICE, H, L)
safe_compute("TYPPRICE", pytafast.TYPPRICE, H, L, C)
safe_compute("WCLPRICE", pytafast.WCLPRICE, H, L, C)

# --- Stats ---
safe_compute("BETA", pytafast.BETA, H, L, 5)
safe_compute("CORREL", pytafast.CORREL, H, L, 30)
safe_compute("LINEARREG", pytafast.LINEARREG, C, 14)
safe_compute("LINEARREG_ANGLE", pytafast.LINEARREG_ANGLE, C, 14)
safe_compute("LINEARREG_INTERCEPT", pytafast.LINEARREG_INTERCEPT, C, 14)
safe_compute("LINEARREG_SLOPE", pytafast.LINEARREG_SLOPE, C, 14)
safe_compute("TSF", pytafast.TSF, C, 14)
safe_compute("VAR", pytafast.VAR, C, 5)
safe_compute("AVGDEV", pytafast.AVGDEV, C, 14)
safe_compute("MAX", pytafast.MAX, C, 30)
safe_compute("MIN", pytafast.MIN, C, 30)
safe_compute("SUM", pytafast.SUM, C, 30)
# MINMAX returns tuple
safe_compute("MINMAX", pytafast.MINMAX, C, 30)

# --- Math ---
safe_compute("ADD", pytafast.ADD, H, L)
safe_compute("SUB", pytafast.SUB, H, L)
safe_compute("MULT", pytafast.MULT, H, L)
safe_compute("DIV", pytafast.DIV, H, L)
safe_compute("SQRT", pytafast.SQRT, C)
safe_compute("LN", pytafast.LN, C)
safe_compute("LOG10", pytafast.LOG10, C)
safe_compute("SIN", pytafast.SIN, C)
safe_compute("COS", pytafast.COS, C)
safe_compute("TAN", pytafast.TAN, C)

# --- Cycle ---
safe_compute("HT_DCPERIOD", pytafast.HT_DCPERIOD, C)
safe_compute("HT_DCPHASE", pytafast.HT_DCPHASE, C)
safe_compute("HT_TRENDLINE", pytafast.HT_TRENDLINE, C)
safe_compute("HT_TRENDMODE", pytafast.HT_TRENDMODE, C)
safe_compute("HT_PHASOR", pytafast.HT_PHASOR, C)
safe_compute("HT_SINE", pytafast.HT_SINE, C)

# --- Candlesticks ---
cdl_list = [
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
    "CDLABANDONEDBABY",
    "CDLDARKCLOUDCOVER",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLMATHOLD",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
]

# --- R-consistent Indicators ---
# MAs
results["ZLEMA"] = pytafast.ZLEMA(C)
results["HMA"] = pytafast.HMA(C)
results["ALMA"] = pytafast.ALMA(C)
results["EVWMA"] = pytafast.EVWMA(C, V)

# Channels
u, m, l = pytafast.keltnerChannels(H, L, C)
results["Keltner_up"] = u
results["Keltner_mid"] = m
results["Keltner_dn"] = l

# Oscillators
results["CMF"] = pytafast.CMF(H, L, C, V)
results["DPO"] = pytafast.DPO(C)
emv_v, emv_ma = pytafast.EMV(H, L, V)
results["EMV_emv"] = emv_v
results["EMV_ma"] = emv_ma
smi_v, smi_sig = pytafast.SMI(H, L, C)
results["SMI_smi"] = smi_v
results["SMI_signal"] = smi_sig

# Special
results["VHF"] = pytafast.VHF(C)
results["SNR"] = pytafast.SNR(H, L, C)

# --- New Plotting Indicators (Validation) ---
# Ichimoku
n1, n2, n3 = 9, 26, 52
results["Tenkan"] = (pytafast.MAX(H, n1) + pytafast.MIN(L, n1)) / 2
results["Kijun"] = (pytafast.MAX(H, n2) + pytafast.MIN(L, n2)) / 2
results["SenkouA"] = (results["Tenkan"] + results["Kijun"]) / 2
results["SenkouB"] = (pytafast.MAX(H, n3) + pytafast.MIN(L, n3)) / 2

# TDI
tdi_rsi = pytafast.RSI(C, 13)
results["TDI_price"] = pytafast.SMA(tdi_rsi, 2)
results["TDI_signal"] = pytafast.SMA(tdi_rsi, 7)
u, m, l = pytafast.BBANDS(tdi_rsi, 34, 1.6185, 1.6185)
results["TDI_mbl"] = m
results["TDI_ub"] = u
results["TDI_lb"] = l

# Legacy ones already there: Donchian, ZigZag, GMMA, KST
# DonchianChannel
u, m, l = pytafast.DonchianChannel(H, L, timeperiod=10)
results["Donchian_high"] = u
results["Donchian_mid"] = m
results["Donchian_low"] = l
results["ZigZag"] = pytafast.ZIGZAG(H, L, change=5.0, percent=True)
gmma_res = pytafast.GMMA(C)
for i, g in enumerate(gmma_res):
    results[f"GMMA_{i}"] = g
kst_val, kst_sig = pytafast.KST(C)
results["KST_kst"] = kst_val
results["KST_signal"] = kst_sig

pd.DataFrame(results).to_csv("py_all_results.csv", index=False)
print("Python results exported.")
