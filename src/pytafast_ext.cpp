// pytafast_ext - Main module definition
// Function implementations are in separate files:
//   overlap.cpp, momentum.cpp, volatility.cpp, price_transform.cpp, volume.cpp
#include "common.h"

// Forward declarations from overlap.cpp
DoubleArrayOUT sma(DoubleArrayIN, int);
DoubleArrayOUT ema(DoubleArrayIN, int);
nb::tuple bbands(DoubleArrayIN, int, double, double, int);
DoubleArrayOUT dema(DoubleArrayIN, int);
DoubleArrayOUT kama(DoubleArrayIN, int);
DoubleArrayOUT ma(DoubleArrayIN, int, int);
DoubleArrayOUT t3(DoubleArrayIN, int, double);
DoubleArrayOUT tema(DoubleArrayIN, int);
DoubleArrayOUT trima(DoubleArrayIN, int);
DoubleArrayOUT wma(DoubleArrayIN, int);
DoubleArrayOUT sar(DoubleArrayIN, DoubleArrayIN, double, double);
DoubleArrayOUT midpoint(DoubleArrayIN, int);

// Forward declarations from momentum.cpp
DoubleArrayOUT rsi(DoubleArrayIN, int);
nb::tuple macd(DoubleArrayIN, int, int, int);
nb::tuple macdext(DoubleArrayIN, int, int, int, int, int, int);
nb::tuple macdfix(DoubleArrayIN, int);
DoubleArrayOUT roc(DoubleArrayIN, int);
DoubleArrayOUT rocp(DoubleArrayIN, int);
DoubleArrayOUT rocr(DoubleArrayIN, int);
DoubleArrayOUT rocr100(DoubleArrayIN, int);
nb::tuple stoch(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int, int, int, int,
                int);
nb::tuple stochf(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int, int, int);
nb::tuple stochrsi(DoubleArrayIN, int, int, int, int);
DoubleArrayOUT mom(DoubleArrayIN, int);
DoubleArrayOUT cmo(DoubleArrayIN, int);
DoubleArrayOUT apo(DoubleArrayIN, int, int, int);
DoubleArrayOUT ppo(DoubleArrayIN, int, int, int);
DoubleArrayOUT trix(DoubleArrayIN, int);
nb::tuple aroon(DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT aroonosc(DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT adx(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT adxr(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT dx(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT minus_di(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT minus_dm(DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT plus_di(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT plus_dm(DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT willr(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT mfi(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, DoubleArrayIN,
                   int);
DoubleArrayOUT cci(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT ultosc(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int, int,
                      int);
DoubleArrayOUT bop(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, DoubleArrayIN);

// Forward declarations from volatility.cpp
DoubleArrayOUT atr(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT natr(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT trange(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT stddev(DoubleArrayIN, int, double);

// Forward declarations from volume.cpp
DoubleArrayOUT obv(DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT ad(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT adosc(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, DoubleArrayIN,
                     int, int);

// Forward declarations from statistic.cpp
DoubleArrayOUT beta(DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT correl(DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT linearreg(DoubleArrayIN, int);
DoubleArrayOUT linearreg_angle(DoubleArrayIN, int);
DoubleArrayOUT linearreg_intercept(DoubleArrayIN, int);
DoubleArrayOUT linearreg_slope(DoubleArrayIN, int);
DoubleArrayOUT tsf(DoubleArrayIN, int);
DoubleArrayOUT var(DoubleArrayIN, int, double);
DoubleArrayOUT avgdev(DoubleArrayIN, int);
DoubleArrayOUT ta_max(DoubleArrayIN, int);
DoubleArrayOUT ta_min(DoubleArrayIN, int);
DoubleArrayOUT ta_sum(DoubleArrayIN, int);
nb::tuple minmax(DoubleArrayIN, int);
nb::tuple minmaxindex(DoubleArrayIN, int);

// Forward declarations from cycle.cpp
DoubleArrayOUT ht_dcperiod(DoubleArrayIN);
DoubleArrayOUT ht_dcphase(DoubleArrayIN);
nb::tuple ht_phasor(DoubleArrayIN);
nb::tuple ht_sine(DoubleArrayIN);
DoubleArrayOUT ht_trendline(DoubleArrayIN);
nb::ndarray<int, nb::numpy, nb::ndim<1>> ht_trendmode(DoubleArrayIN);

DoubleArrayOUT avgprice(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN,
                        DoubleArrayIN);
DoubleArrayOUT medprice(DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT typprice(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT wclprice(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT midprice(DoubleArrayIN, DoubleArrayIN, int);

// Forward declarations from math_operator.cpp
DoubleArrayOUT add(DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT sub(DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT mult(DoubleArrayIN, DoubleArrayIN);
DoubleArrayOUT ta_div(DoubleArrayIN, DoubleArrayIN);

// Forward declarations from math_transform.cpp
DoubleArrayOUT ta_acos(DoubleArrayIN);
DoubleArrayOUT ta_asin(DoubleArrayIN);
DoubleArrayOUT ta_atan(DoubleArrayIN);
DoubleArrayOUT ta_ceil(DoubleArrayIN);
DoubleArrayOUT ta_cos(DoubleArrayIN);
DoubleArrayOUT ta_cosh(DoubleArrayIN);
DoubleArrayOUT ta_exp(DoubleArrayIN);
DoubleArrayOUT ta_floor(DoubleArrayIN);
DoubleArrayOUT ta_ln(DoubleArrayIN);
DoubleArrayOUT ta_log10(DoubleArrayIN);
DoubleArrayOUT ta_sin(DoubleArrayIN);
DoubleArrayOUT ta_sinh(DoubleArrayIN);
DoubleArrayOUT ta_sqrt(DoubleArrayIN);
DoubleArrayOUT ta_tan(DoubleArrayIN);
DoubleArrayOUT ta_tanh(DoubleArrayIN);

// Forward declarations from candlestick.cpp
using IntArrayOUT = nb::ndarray<int, nb::numpy, nb::ndim<1>>;
#define CDL_FWD(NAME)                                                          \
  IntArrayOUT NAME(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, DoubleArrayIN)
#define CDL_FWD_PEN(NAME)                                                      \
  IntArrayOUT NAME(DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, DoubleArrayIN, \
                   double)
CDL_FWD(cdl2crows);
CDL_FWD(cdl3blackcrows);
CDL_FWD(cdl3inside);
CDL_FWD(cdl3linestrike);
CDL_FWD(cdl3outside);
CDL_FWD(cdl3starsinsouth);
CDL_FWD(cdl3whitesoldiers);
CDL_FWD(cdladvanceblock);
CDL_FWD(cdlbelthold);
CDL_FWD(cdlbreakaway);
CDL_FWD(cdlclosingmarubozu);
CDL_FWD(cdlconcealbabyswall);
CDL_FWD(cdlcounterattack);
CDL_FWD(cdldoji);
CDL_FWD(cdldojistar);
CDL_FWD(cdldragonflydoji);
CDL_FWD(cdlengulfing);
CDL_FWD(cdlgapsidesidewhite);
CDL_FWD(cdlgravestonedoji);
CDL_FWD(cdlhammer);
CDL_FWD(cdlhangingman);
CDL_FWD(cdlharami);
CDL_FWD(cdlharamicross);
CDL_FWD(cdlhighwave);
CDL_FWD(cdlhikkake);
CDL_FWD(cdlhikkakemod);
CDL_FWD(cdlhomingpigeon);
CDL_FWD(cdlidentical3crows);
CDL_FWD(cdlinneck);
CDL_FWD(cdlinvertedhammer);
CDL_FWD(cdlkicking);
CDL_FWD(cdlkickingbylength);
CDL_FWD(cdlladderbottom);
CDL_FWD(cdllongleggeddoji);
CDL_FWD(cdllongline);
CDL_FWD(cdlmarubozu);
CDL_FWD(cdlmatchinglow);
CDL_FWD(cdlonneck);
CDL_FWD(cdlpiercing);
CDL_FWD(cdlrickshawman);
CDL_FWD(cdlrisefall3methods);
CDL_FWD(cdlseparatinglines);
CDL_FWD(cdlshootingstar);
CDL_FWD(cdlshortline);
CDL_FWD(cdlspinningtop);
CDL_FWD(cdlstalledpattern);
CDL_FWD(cdlsticksandwich);
CDL_FWD(cdltakuri);
CDL_FWD(cdltasukigap);
CDL_FWD(cdlthrusting);
CDL_FWD(cdltristar);
CDL_FWD(cdlunique3river);
CDL_FWD(cdlupsidegap2crows);
CDL_FWD(cdlxsidegap3methods);
CDL_FWD_PEN(cdlabandonedbaby);
CDL_FWD_PEN(cdldarkcloudcover);
CDL_FWD_PEN(cdleveningdojistar);
CDL_FWD_PEN(cdleveningstar);
CDL_FWD_PEN(cdlmathold);
CDL_FWD_PEN(cdlmorningdojistar);
CDL_FWD_PEN(cdlmorningstar);
#undef CDL_FWD
#undef CDL_FWD_PEN

// Helper to initialize and shutdown TA-lib
void initialize() {
  TA_RetCode retcode = TA_Initialize();
  check_ta_retcode(retcode, "TA_Initialize");
}

void shutdown() {
  TA_RetCode retcode = TA_Shutdown();
  check_ta_retcode(retcode, "TA_Shutdown");
}

NB_MODULE(pytafast_ext, m) {
  m.doc() = "TA-Lib wrapper using nanobind";

  nb::enum_<TA_MAType>(m, "MAType")
      .value("SMA", TA_MAType_SMA)
      .value("EMA", TA_MAType_EMA)
      .value("WMA", TA_MAType_WMA)
      .value("DEMA", TA_MAType_DEMA)
      .value("TEMA", TA_MAType_TEMA)
      .value("TRIMA", TA_MAType_TRIMA)
      .value("KAMA", TA_MAType_KAMA)
      .value("MAMA", TA_MAType_MAMA)
      .value("T3", TA_MAType_T3);

  // --- Overlap Studies ---
  m.def("SMA", &sma, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("EMA", &ema, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("BBANDS", &bbands, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 5, nb::arg("optInNbDevUp") = 2.0,
        nb::arg("optInNbDevDn") = 2.0, nb::arg("optInMAType") = 0);
  m.def("DEMA", &dema, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("KAMA", &kama, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("MA", &ma, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30, nb::arg("optInMAType") = 0);
  m.def("T3", &t3, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 5, nb::arg("optInVFactor") = 0.7);
  m.def("TEMA", &tema, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("TRIMA", &trima, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("WMA", &wma, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("SAR", &sar, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("optInAcceleration") = 0.02,
        nb::arg("optInMaximum") = 0.2);
  m.def("MIDPOINT", &midpoint, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);

  // --- Momentum ---
  m.def("RSI", &rsi, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("MACD", &macd, nb::arg("inReal").noconvert(),
        nb::arg("optInFastPeriod") = 12, nb::arg("optInSlowPeriod") = 26,
        nb::arg("optInSignalPeriod") = 9);
  m.def("MACDEXT", &macdext, nb::arg("inReal").noconvert(),
        nb::arg("optInFastPeriod") = 12, nb::arg("optInFastMAType") = 0,
        nb::arg("optInSlowPeriod") = 26, nb::arg("optInSlowMAType") = 0,
        nb::arg("optInSignalPeriod") = 9, nb::arg("optInSignalMAType") = 0);
  m.def("MACDFIX", &macdfix, nb::arg("inReal").noconvert(),
        nb::arg("optInSignalPeriod") = 9);
  m.def("ROC", &roc, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 10);
  m.def("ROCP", &rocp, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 10);
  m.def("ROCR", &rocr, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 10);
  m.def("ROCR100", &rocr100, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 10);
  m.def("STOCH", &stoch, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInFastK_Period") = 5, nb::arg("optInSlowK_Period") = 3,
        nb::arg("optInSlowK_MAType") = 0, nb::arg("optInSlowD_Period") = 3,
        nb::arg("optInSlowD_MAType") = 0);
  m.def("STOCHF", &stochf, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInFastK_Period") = 5, nb::arg("optInFastD_Period") = 3,
        nb::arg("optInFastD_MAType") = 0);
  m.def("STOCHRSI", &stochrsi, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14, nb::arg("optInFastK_Period") = 5,
        nb::arg("optInFastD_Period") = 3, nb::arg("optInFastD_MAType") = 0);
  m.def("MOM", &mom, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 10);
  m.def("CMO", &cmo, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("APO", &apo, nb::arg("inReal").noconvert(),
        nb::arg("optInFastPeriod") = 12, nb::arg("optInSlowPeriod") = 26,
        nb::arg("optInMAType") = 0);
  m.def("PPO", &ppo, nb::arg("inReal").noconvert(),
        nb::arg("optInFastPeriod") = 12, nb::arg("optInSlowPeriod") = 26,
        nb::arg("optInMAType") = 0);
  m.def("TRIX", &trix, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("AROON", &aroon, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("optInTimePeriod") = 14);
  m.def("AROONOSC", &aroonosc, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("optInTimePeriod") = 14);
  m.def("ADX", &adx, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("ADXR", &adxr, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("DX", &dx, nb::arg("inHigh").noconvert(), nb::arg("inLow").noconvert(),
        nb::arg("inClose").noconvert(), nb::arg("optInTimePeriod") = 14);
  m.def("MINUS_DI", &minus_di, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("MINUS_DM", &minus_dm, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("optInTimePeriod") = 14);
  m.def("PLUS_DI", &plus_di, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("PLUS_DM", &plus_dm, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("optInTimePeriod") = 14);
  m.def("WILLR", &willr, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("MFI", &mfi, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("inVolume").noconvert(), nb::arg("optInTimePeriod") = 14);
  m.def("CCI", &cci, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("ULTOSC", &ultosc, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod1") = 7, nb::arg("optInTimePeriod2") = 14,
        nb::arg("optInTimePeriod3") = 28);
  m.def("BOP", &bop, nb::arg("inOpen").noconvert(),
        nb::arg("inHigh").noconvert(), nb::arg("inLow").noconvert(),
        nb::arg("inClose").noconvert());

  // --- Volatility ---
  m.def("ATR", &atr, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("NATR", &natr, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("TRANGE", &trange, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert());
  m.def("STDDEV", &stddev, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 5, nb::arg("optInNbDev") = 1.0);

  // --- Volume ---
  m.def("OBV", &obv, nb::arg("inReal").noconvert(),
        nb::arg("inVolume").noconvert());
  m.def("AD", &ad, nb::arg("inHigh").noconvert(), nb::arg("inLow").noconvert(),
        nb::arg("inClose").noconvert(), nb::arg("inVolume").noconvert());
  m.def("ADOSC", &adosc, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("inVolume").noconvert(), nb::arg("optInFastPeriod") = 3,
        nb::arg("optInSlowPeriod") = 10);

  // --- Statistics ---
  m.def("BETA", &beta, nb::arg("inReal0").noconvert(),
        nb::arg("inReal1").noconvert(), nb::arg("optInTimePeriod") = 5);
  m.def("CORREL", &correl, nb::arg("inReal0").noconvert(),
        nb::arg("inReal1").noconvert(), nb::arg("optInTimePeriod") = 30);
  m.def("LINEARREG", &linearreg, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("LINEARREG_ANGLE", &linearreg_angle, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("LINEARREG_INTERCEPT", &linearreg_intercept,
        nb::arg("inReal").noconvert(), nb::arg("optInTimePeriod") = 14);
  m.def("LINEARREG_SLOPE", &linearreg_slope, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("TSF", &tsf, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("VAR", &var, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 5, nb::arg("optInNbDev") = 1.0);
  m.def("AVGDEV", &avgdev, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);

  // --- Price Transform ---
  m.def("AVGPRICE", &avgprice, nb::arg("inOpen").noconvert(),
        nb::arg("inHigh").noconvert(), nb::arg("inLow").noconvert(),
        nb::arg("inClose").noconvert());
  m.def("MEDPRICE", &medprice, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert());
  m.def("TYPPRICE", &typprice, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert());
  m.def("WCLPRICE", &wclprice, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert());
  m.def("MIDPRICE", &midprice, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("optInTimePeriod") = 14);

  // --- Math Operators ---
  m.def("ADD", &add, nb::arg("inReal0").noconvert(),
        nb::arg("inReal1").noconvert());
  m.def("SUB", &sub, nb::arg("inReal0").noconvert(),
        nb::arg("inReal1").noconvert());
  m.def("MULT", &mult, nb::arg("inReal0").noconvert(),
        nb::arg("inReal1").noconvert());
  m.def("DIV", &ta_div, nb::arg("inReal0").noconvert(),
        nb::arg("inReal1").noconvert());

  // --- Math Transforms ---
  m.def("ACOS", &ta_acos, nb::arg("inReal").noconvert());
  m.def("ASIN", &ta_asin, nb::arg("inReal").noconvert());
  m.def("ATAN", &ta_atan, nb::arg("inReal").noconvert());
  m.def("CEIL", &ta_ceil, nb::arg("inReal").noconvert());
  m.def("COS", &ta_cos, nb::arg("inReal").noconvert());
  m.def("COSH", &ta_cosh, nb::arg("inReal").noconvert());
  m.def("EXP", &ta_exp, nb::arg("inReal").noconvert());
  m.def("FLOOR", &ta_floor, nb::arg("inReal").noconvert());
  m.def("LN", &ta_ln, nb::arg("inReal").noconvert());
  m.def("LOG10", &ta_log10, nb::arg("inReal").noconvert());
  m.def("SIN", &ta_sin, nb::arg("inReal").noconvert());
  m.def("SINH", &ta_sinh, nb::arg("inReal").noconvert());
  m.def("SQRT", &ta_sqrt, nb::arg("inReal").noconvert());
  m.def("TAN", &ta_tan, nb::arg("inReal").noconvert());
  m.def("TANH", &ta_tanh, nb::arg("inReal").noconvert());

  // --- Statistics (MIN/MAX/SUM/MINMAX/MINMAXINDEX) ---
  m.def("MAX", &ta_max, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("MIN", &ta_min, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("SUM", &ta_sum, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("MINMAX", &minmax, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("MINMAXINDEX", &minmaxindex, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);

  // --- Cycle ---
  m.def("HT_DCPERIOD", &ht_dcperiod, nb::arg("inReal").noconvert());
  m.def("HT_DCPHASE", &ht_dcphase, nb::arg("inReal").noconvert());
  m.def("HT_PHASOR", &ht_phasor, nb::arg("inReal").noconvert());
  m.def("HT_SINE", &ht_sine, nb::arg("inReal").noconvert());
  m.def("HT_TRENDLINE", &ht_trendline, nb::arg("inReal").noconvert());
  m.def("HT_TRENDMODE", &ht_trendmode, nb::arg("inReal").noconvert());

  // --- Candlestick Patterns (standard OHLC) ---
#define CDL_BIND(NAME, FUNC)                                                   \
  m.def(#NAME, &FUNC, nb::arg("inOpen").noconvert(),                           \
        nb::arg("inHigh").noconvert(), nb::arg("inLow").noconvert(),           \
        nb::arg("inClose").noconvert())
  CDL_BIND(CDL2CROWS, cdl2crows);
  CDL_BIND(CDL3BLACKCROWS, cdl3blackcrows);
  CDL_BIND(CDL3INSIDE, cdl3inside);
  CDL_BIND(CDL3LINESTRIKE, cdl3linestrike);
  CDL_BIND(CDL3OUTSIDE, cdl3outside);
  CDL_BIND(CDL3STARSINSOUTH, cdl3starsinsouth);
  CDL_BIND(CDL3WHITESOLDIERS, cdl3whitesoldiers);
  CDL_BIND(CDLADVANCEBLOCK, cdladvanceblock);
  CDL_BIND(CDLBELTHOLD, cdlbelthold);
  CDL_BIND(CDLBREAKAWAY, cdlbreakaway);
  CDL_BIND(CDLCLOSINGMARUBOZU, cdlclosingmarubozu);
  CDL_BIND(CDLCONCEALBABYSWALL, cdlconcealbabyswall);
  CDL_BIND(CDLCOUNTERATTACK, cdlcounterattack);
  CDL_BIND(CDLDOJI, cdldoji);
  CDL_BIND(CDLDOJISTAR, cdldojistar);
  CDL_BIND(CDLDRAGONFLYDOJI, cdldragonflydoji);
  CDL_BIND(CDLENGULFING, cdlengulfing);
  CDL_BIND(CDLGAPSIDESIDEWHITE, cdlgapsidesidewhite);
  CDL_BIND(CDLGRAVESTONEDOJI, cdlgravestonedoji);
  CDL_BIND(CDLHAMMER, cdlhammer);
  CDL_BIND(CDLHANGINGMAN, cdlhangingman);
  CDL_BIND(CDLHARAMI, cdlharami);
  CDL_BIND(CDLHARAMICROSS, cdlharamicross);
  CDL_BIND(CDLHIGHWAVE, cdlhighwave);
  CDL_BIND(CDLHIKKAKE, cdlhikkake);
  CDL_BIND(CDLHIKKAKEMOD, cdlhikkakemod);
  CDL_BIND(CDLHOMINGPIGEON, cdlhomingpigeon);
  CDL_BIND(CDLIDENTICAL3CROWS, cdlidentical3crows);
  CDL_BIND(CDLINNECK, cdlinneck);
  CDL_BIND(CDLINVERTEDHAMMER, cdlinvertedhammer);
  CDL_BIND(CDLKICKING, cdlkicking);
  CDL_BIND(CDLKICKINGBYLENGTH, cdlkickingbylength);
  CDL_BIND(CDLLADDERBOTTOM, cdlladderbottom);
  CDL_BIND(CDLLONGLEGGEDDOJI, cdllongleggeddoji);
  CDL_BIND(CDLLONGLINE, cdllongline);
  CDL_BIND(CDLMARUBOZU, cdlmarubozu);
  CDL_BIND(CDLMATCHINGLOW, cdlmatchinglow);
  CDL_BIND(CDLONNECK, cdlonneck);
  CDL_BIND(CDLPIERCING, cdlpiercing);
  CDL_BIND(CDLRICKSHAWMAN, cdlrickshawman);
  CDL_BIND(CDLRISEFALL3METHODS, cdlrisefall3methods);
  CDL_BIND(CDLSEPARATINGLINES, cdlseparatinglines);
  CDL_BIND(CDLSHOOTINGSTAR, cdlshootingstar);
  CDL_BIND(CDLSHORTLINE, cdlshortline);
  CDL_BIND(CDLSPINNINGTOP, cdlspinningtop);
  CDL_BIND(CDLSTALLEDPATTERN, cdlstalledpattern);
  CDL_BIND(CDLSTICKSANDWICH, cdlsticksandwich);
  CDL_BIND(CDLTAKURI, cdltakuri);
  CDL_BIND(CDLTASUKIGAP, cdltasukigap);
  CDL_BIND(CDLTHRUSTING, cdlthrusting);
  CDL_BIND(CDLTRISTAR, cdltristar);
  CDL_BIND(CDLUNIQUE3RIVER, cdlunique3river);
  CDL_BIND(CDLUPSIDEGAP2CROWS, cdlupsidegap2crows);
  CDL_BIND(CDLXSIDEGAP3METHODS, cdlxsidegap3methods);
#undef CDL_BIND

  // --- Candlestick Patterns (with penetration) ---
#define CDL_BIND_PEN(NAME, FUNC, DEF)                                          \
  m.def(#NAME, &FUNC, nb::arg("inOpen").noconvert(),                           \
        nb::arg("inHigh").noconvert(), nb::arg("inLow").noconvert(),           \
        nb::arg("inClose").noconvert(), nb::arg("penetration") = DEF)
  CDL_BIND_PEN(CDLABANDONEDBABY, cdlabandonedbaby, 0.3);
  CDL_BIND_PEN(CDLDARKCLOUDCOVER, cdldarkcloudcover, 0.5);
  CDL_BIND_PEN(CDLEVENINGDOJISTAR, cdleveningdojistar, 0.3);
  CDL_BIND_PEN(CDLEVENINGSTAR, cdleveningstar, 0.3);
  CDL_BIND_PEN(CDLMATHOLD, cdlmathold, 0.5);
  CDL_BIND_PEN(CDLMORNINGDOJISTAR, cdlmorningdojistar, 0.3);
  CDL_BIND_PEN(CDLMORNINGSTAR, cdlmorningstar, 0.3);
#undef CDL_BIND_PEN

  m.def("initialize", &initialize);
  m.def("shutdown", &shutdown);
}
