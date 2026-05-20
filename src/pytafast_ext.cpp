// pytafast_ext - Main module definition
// Function implementations are in separate files:
//   overlap.cpp, momentum.cpp, volatility.cpp, price_transform.cpp, volume.cpp
#include "common.h"
#include <mutex>

// Forward declarations from overlap.cpp
DoubleArrayOUT sma(DoubleArrayIN, int);
DoubleArrayOUT ema(DoubleArrayIN, int);
nb::tuple bbands(DoubleArrayIN, int, double, double, int);
DoubleArrayOUT dema(DoubleArrayIN, int);
DoubleArrayOUT kama(DoubleArrayIN, int);
DoubleArrayOUT ma(DoubleArrayIN, int, int);
DoubleArrayOUT t3(DoubleArrayIN, int, double);
nb::tuple mama(DoubleArrayIN, double, double);
DoubleArrayOUT tema(DoubleArrayIN, int);
DoubleArrayOUT trima(DoubleArrayIN, int);
DoubleArrayOUT wma(DoubleArrayIN, int);
DoubleArrayOUT sar(DoubleArrayIN, DoubleArrayIN, double, double);
DoubleArrayOUT midpoint(DoubleArrayIN, int);
DoubleArrayOUT zigzag(DoubleArrayIN, DoubleArrayIN, double, bool);
DoubleArrayOUT alma(DoubleArrayIN, int, double, double);
DoubleArrayOUT evwma(DoubleArrayIN, DoubleArrayIN, int);
DoubleArrayOUT zlema(DoubleArrayIN, int);

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
static std::once_flag ta_init_flag;

void initialize() {
  std::call_once(ta_init_flag, []() {
    TA_RetCode retcode = TA_Initialize();
    check_ta_retcode(retcode, "TA_Initialize");
  });
}

void shutdown() {
  TA_RetCode retcode = TA_Shutdown();
  check_ta_retcode(retcode, "TA_Shutdown");
}

NB_MODULE(pytafast_ext, m) {
  using namespace nb::literals;
  m.doc() = "TA-Lib wrapper using nanobind";

  nb::exception<ta_error>(m, "TALibError");

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
  m.def("SMA", &sma, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("EMA", &ema, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("BBANDS", &bbands, "inReal"_a, "optInTimePeriod"_a = 5,
        "optInNbDevUp"_a = 2.0, "optInNbDevDn"_a = 2.0,
        "optInMAType"_a = 0);
  m.def("DEMA", &dema, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("KAMA", &kama, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("MA", &ma, "inReal"_a, "optInTimePeriod"_a = 30,
        "optInMAType"_a = 0);
  m.def("T3", &t3, "inReal"_a, "optInTimePeriod"_a = 5,
        "optInVFactor"_a = 0.7);
  m.def("MAMA", &mama, "inReal"_a, "optInFastLimit"_a = 0.5,
        "optInSlowLimit"_a = 0.05);
  m.def("TEMA", &tema, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("TRIMA", &trima, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("WMA", &wma, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("SAR", &sar, "inHigh"_a, "inLow"_a,
        "optInAcceleration"_a = 0.02, "optInMaximum"_a = 0.2);
  m.def("MIDPOINT", &midpoint, "inReal"_a,
        "optInTimePeriod"_a = 14);
  m.def("ZIGZAG", &zigzag, "inHigh"_a, "inLow"_a,
        "change"_a = 10.0, "percent"_a = true);
  m.def("ALMA", &alma, "inReal"_a, "optInTimePeriod"_a = 9,
        "optInOffset"_a = 0.85, "optInSigma"_a = 6.0);
  m.def("EVWMA", &evwma, "inReal"_a, "inVolume"_a,
        "optInTimePeriod"_a = 30);
  m.def("ZLEMA", &zlema, "inReal"_a, "optInTimePeriod"_a = 30);

  // --- Momentum ---
  m.def("RSI", &rsi, "inReal"_a, "optInTimePeriod"_a = 14);
  m.def("MACD", &macd, "inReal"_a, "optInFastPeriod"_a = 12,
        "optInSlowPeriod"_a = 26, "optInSignalPeriod"_a = 9);
  m.def("MACDEXT", &macdext, "inReal"_a, "optInFastPeriod"_a = 12,
        "optInFastMAType"_a = 0, "optInSlowPeriod"_a = 26,
        "optInSlowMAType"_a = 0, "optInSignalPeriod"_a = 9,
        "optInSignalMAType"_a = 0);
  m.def("MACDFIX", &macdfix, "inReal"_a,
        "optInSignalPeriod"_a = 9);
  m.def("ROC", &roc, "inReal"_a, "optInTimePeriod"_a = 10);
  m.def("ROCP", &rocp, "inReal"_a, "optInTimePeriod"_a = 10);
  m.def("ROCR", &rocr, "inReal"_a, "optInTimePeriod"_a = 10);
  m.def("ROCR100", &rocr100, "inReal"_a,
        "optInTimePeriod"_a = 10);
  m.def("STOCH", &stoch, "inHigh"_a, "inLow"_a,
        "inClose"_a, "optInFastK_Period"_a = 5,
        "optInSlowK_Period"_a = 3, "optInSlowK_MAType"_a = 0,
        "optInSlowD_Period"_a = 3, "optInSlowD_MAType"_a = 0);
  m.def("STOCHF", &stochf, "inHigh"_a, "inLow"_a,
        "inClose"_a, "optInFastK_Period"_a = 5,
        "optInFastD_Period"_a = 3, "optInFastD_MAType"_a = 0);
  m.def("STOCHRSI", &stochrsi, "inReal"_a,
        "optInTimePeriod"_a = 14, "optInFastK_Period"_a = 5,
        "optInFastD_Period"_a = 3, "optInFastD_MAType"_a = 0);
  m.def("MOM", &mom, "inReal"_a, "optInTimePeriod"_a = 10);
  m.def("CMO", &cmo, "inReal"_a, "optInTimePeriod"_a = 14);
  m.def("APO", &apo, "inReal"_a, "optInFastPeriod"_a = 12,
        "optInSlowPeriod"_a = 26, "optInMAType"_a = 0);
  m.def("PPO", &ppo, "inReal"_a, "optInFastPeriod"_a = 12,
        "optInSlowPeriod"_a = 26, "optInMAType"_a = 0);
  m.def("TRIX", &trix, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("AROON", &aroon, "inHigh"_a, "inLow"_a,
        "optInTimePeriod"_a = 14);
  m.def("AROONOSC", &aroonosc, "inHigh"_a, "inLow"_a,
        "optInTimePeriod"_a = 14);
  m.def("ADX", &adx, "inHigh"_a, "inLow"_a, "inClose"_a,
        "optInTimePeriod"_a = 14);
  m.def("ADXR", &adxr, "inHigh"_a, "inLow"_a, "inClose"_a,
        "optInTimePeriod"_a = 14);
  m.def("DX", &dx, "inHigh"_a, "inLow"_a, "inClose"_a,
        "optInTimePeriod"_a = 14);
  m.def("MINUS_DI", &minus_di, "inHigh"_a, "inLow"_a,
        "inClose"_a, "optInTimePeriod"_a = 14);
  m.def("MINUS_DM", &minus_dm, "inHigh"_a, "inLow"_a,
        "optInTimePeriod"_a = 14);
  m.def("PLUS_DI", &plus_di, "inHigh"_a, "inLow"_a,
        "inClose"_a, "optInTimePeriod"_a = 14);
  m.def("PLUS_DM", &plus_dm, "inHigh"_a, "inLow"_a,
        "optInTimePeriod"_a = 14);
  m.def("WILLR", &willr, "inHigh"_a, "inLow"_a,
        "inClose"_a, "optInTimePeriod"_a = 14);
  m.def("MFI", &mfi, "inHigh"_a, "inLow"_a, "inClose"_a,
        "inVolume"_a, "optInTimePeriod"_a = 14);
  m.def("CCI", &cci, "inHigh"_a, "inLow"_a, "inClose"_a,
        "optInTimePeriod"_a = 14);
  m.def("ULTOSC", &ultosc, "inHigh"_a, "inLow"_a,
        "inClose"_a, "optInTimePeriod1"_a = 7,
        "optInTimePeriod2"_a = 14, "optInTimePeriod3"_a = 28);
  m.def("BOP", &bop, "inOpen"_a, "inHigh"_a, "inLow"_a,
        "inClose"_a);

  // --- Volatility ---
  m.def("ATR", &atr, "inHigh"_a, "inLow"_a, "inClose"_a,
        "optInTimePeriod"_a = 14);
  m.def("NATR", &natr, "inHigh"_a, "inLow"_a, "inClose"_a,
        "optInTimePeriod"_a = 14);
  m.def("TRANGE", &trange, "inHigh"_a, "inLow"_a,
        "inClose"_a);
  m.def("STDDEV", &stddev, "inReal"_a, "optInTimePeriod"_a = 5,
        "optInNbDev"_a = 1.0);

  // --- Volume ---
  m.def("OBV", &obv, "inReal"_a, "inVolume"_a);
  m.def("AD", &ad, "inHigh"_a, "inLow"_a, "inClose"_a,
        "inVolume"_a);
  m.def("ADOSC", &adosc, "inHigh"_a, "inLow"_a,
        "inClose"_a, "inVolume"_a, "optInFastPeriod"_a = 3,
        "optInSlowPeriod"_a = 10);

  // --- Statistics ---
  m.def("BETA", &beta, "inReal0"_a, "inReal1"_a,
        "optInTimePeriod"_a = 5);
  m.def("CORREL", &correl, "inReal0"_a, "inReal1"_a,
        "optInTimePeriod"_a = 30);
  m.def("LINEARREG", &linearreg, "inReal"_a,
        "optInTimePeriod"_a = 14);
  m.def("LINEARREG_ANGLE", &linearreg_angle, "inReal"_a,
        "optInTimePeriod"_a = 14);
  m.def("LINEARREG_INTERCEPT", &linearreg_intercept, "inReal"_a,
        "optInTimePeriod"_a = 14);
  m.def("LINEARREG_SLOPE", &linearreg_slope, "inReal"_a,
        "optInTimePeriod"_a = 14);
  m.def("TSF", &tsf, "inReal"_a, "optInTimePeriod"_a = 14);
  m.def("VAR", &var, "inReal"_a, "optInTimePeriod"_a = 5,
        "optInNbDev"_a = 1.0);
  m.def("AVGDEV", &avgdev, "inReal"_a, "optInTimePeriod"_a = 14);

  // --- Price Transform ---
  m.def("AVGPRICE", &avgprice, "inOpen"_a, "inHigh"_a,
        "inLow"_a, "inClose"_a);
  m.def("MEDPRICE", &medprice, "inHigh"_a, "inLow"_a);
  m.def("TYPPRICE", &typprice, "inHigh"_a, "inLow"_a,
        "inClose"_a);
  m.def("WCLPRICE", &wclprice, "inHigh"_a, "inLow"_a,
        "inClose"_a);
  m.def("MIDPRICE", &midprice, "inHigh"_a, "inLow"_a,
        "optInTimePeriod"_a = 14);

  // --- Math Operators ---
  m.def("ADD", &add, "inReal0"_a, "inReal1"_a);
  m.def("SUB", &sub, "inReal0"_a, "inReal1"_a);
  m.def("MULT", &mult, "inReal0"_a, "inReal1"_a);
  m.def("DIV", &ta_div, "inReal0"_a, "inReal1"_a);

  // --- Math Transforms ---
  m.def("ACOS", &ta_acos, "inReal"_a);
  m.def("ASIN", &ta_asin, "inReal"_a);
  m.def("ATAN", &ta_atan, "inReal"_a);
  m.def("CEIL", &ta_ceil, "inReal"_a);
  m.def("COS", &ta_cos, "inReal"_a);
  m.def("COSH", &ta_cosh, "inReal"_a);
  m.def("EXP", &ta_exp, "inReal"_a);
  m.def("FLOOR", &ta_floor, "inReal"_a);
  m.def("LN", &ta_ln, "inReal"_a);
  m.def("LOG10", &ta_log10, "inReal"_a);
  m.def("SIN", &ta_sin, "inReal"_a);
  m.def("SINH", &ta_sinh, "inReal"_a);
  m.def("SQRT", &ta_sqrt, "inReal"_a);
  m.def("TAN", &ta_tan, "inReal"_a);
  m.def("TANH", &ta_tanh, "inReal"_a);

  // --- Statistics (MIN/MAX/SUM/MINMAX/MINMAXINDEX) ---
  m.def("MAX", &ta_max, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("MIN", &ta_min, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("SUM", &ta_sum, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("MINMAX", &minmax, "inReal"_a, "optInTimePeriod"_a = 30);
  m.def("MINMAXINDEX", &minmaxindex, "inReal"_a,
        "optInTimePeriod"_a = 30);

  // --- Cycle ---
  m.def("HT_DCPERIOD", &ht_dcperiod, "inReal"_a);
  m.def("HT_DCPHASE", &ht_dcphase, "inReal"_a);
  m.def("HT_PHASOR", &ht_phasor, "inReal"_a);
  m.def("HT_SINE", &ht_sine, "inReal"_a);
  m.def("HT_TRENDLINE", &ht_trendline, "inReal"_a);
  m.def("HT_TRENDMODE", &ht_trendmode, "inReal"_a);

  // --- Candlestick Patterns (standard OHLC) ---
#define CDL_BIND(NAME, FUNC)                                                   \
  m.def(#NAME, &FUNC, "inOpen"_a, "inHigh"_a, "inLow"_a,  \
        "inClose"_a)
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
  m.def(#NAME, &FUNC, "inOpen"_a, "inHigh"_a, "inLow"_a,  \
        "inClose"_a, "penetration"_a = DEF)
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
