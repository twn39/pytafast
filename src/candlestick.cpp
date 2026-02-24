// Candlestick Pattern Recognition Functions
// All take OHLC input, output integer array (100, -100, or 0)
#include "common.h"

using IntArrayOUT = nb::ndarray<int, nb::numpy, nb::ndim<1>>;

// Helper to allocate int output with 0-fill for lookback region
static std::pair<int *, nb::capsule> alloc_int_output(size_t size,
                                                      int lookback) {
  int *data = new int[size];
  for (size_t i = 0; i < (size_t)lookback && i < size; ++i)
    data[i] = 0;
  nb::capsule owner(data, [](void *p) noexcept { delete[] (int *)p; });
  return {data, std::move(owner)};
}

// Macro for standard CDL functions (OHLC → int, no extra params)
#define CDL_FUNC(NAME, TA_FUNC)                                                \
  IntArrayOUT NAME(DoubleArrayIN inOpen, DoubleArrayIN inHigh,                 \
                   DoubleArrayIN inLow, DoubleArrayIN inClose) {               \
    if (inOpen.size() == 0)                                                    \
      return IntArrayOUT(nullptr, {0}, nb::handle());                          \
    size_t size = inOpen.shape(0);                                             \
    int lookback = TA_FUNC##_Lookback();                                       \
    auto [outData, owner] = alloc_int_output(size, lookback);                  \
    int outBegIdx = 0, outNBElement = 0;                                       \
    TA_RetCode retCode;                                                        \
    {                                                                          \
      nb::gil_scoped_release release;                                          \
      retCode = TA_FUNC(0, size - 1, inOpen.data(), inHigh.data(),             \
                        inLow.data(), inClose.data(), &outBegIdx,              \
                        &outNBElement, outData + lookback);                    \
    }                                                                          \
    check_ta_retcode(retCode, #TA_FUNC);                                       \
    return IntArrayOUT(outData, {size}, owner);                                \
  }

// Macro for CDL functions with penetration parameter
#define CDL_FUNC_PEN(NAME, TA_FUNC, DEFAULT_PEN)                               \
  IntArrayOUT NAME(DoubleArrayIN inOpen, DoubleArrayIN inHigh,                 \
                   DoubleArrayIN inLow, DoubleArrayIN inClose,                 \
                   double optInPenetration = DEFAULT_PEN) {                    \
    if (inOpen.size() == 0)                                                    \
      return IntArrayOUT(nullptr, {0}, nb::handle());                          \
    size_t size = inOpen.shape(0);                                             \
    int lookback = TA_FUNC##_Lookback(optInPenetration);                       \
    auto [outData, owner] = alloc_int_output(size, lookback);                  \
    int outBegIdx = 0, outNBElement = 0;                                       \
    TA_RetCode retCode;                                                        \
    {                                                                          \
      nb::gil_scoped_release release;                                          \
      retCode = TA_FUNC(0, size - 1, inOpen.data(), inHigh.data(),             \
                        inLow.data(), inClose.data(), optInPenetration,        \
                        &outBegIdx, &outNBElement, outData + lookback);        \
    }                                                                          \
    check_ta_retcode(retCode, #TA_FUNC);                                       \
    return IntArrayOUT(outData, {size}, owner);                                \
  }

// Standard CDL functions (no extra params)
CDL_FUNC(cdl2crows, TA_CDL2CROWS)
CDL_FUNC(cdl3blackcrows, TA_CDL3BLACKCROWS)
CDL_FUNC(cdl3inside, TA_CDL3INSIDE)
CDL_FUNC(cdl3linestrike, TA_CDL3LINESTRIKE)
CDL_FUNC(cdl3outside, TA_CDL3OUTSIDE)
CDL_FUNC(cdl3starsinsouth, TA_CDL3STARSINSOUTH)
CDL_FUNC(cdl3whitesoldiers, TA_CDL3WHITESOLDIERS)
CDL_FUNC(cdladvanceblock, TA_CDLADVANCEBLOCK)
CDL_FUNC(cdlbelthold, TA_CDLBELTHOLD)
CDL_FUNC(cdlbreakaway, TA_CDLBREAKAWAY)
CDL_FUNC(cdlclosingmarubozu, TA_CDLCLOSINGMARUBOZU)
CDL_FUNC(cdlconcealbabyswall, TA_CDLCONCEALBABYSWALL)
CDL_FUNC(cdlcounterattack, TA_CDLCOUNTERATTACK)
CDL_FUNC(cdldoji, TA_CDLDOJI)
CDL_FUNC(cdldojistar, TA_CDLDOJISTAR)
CDL_FUNC(cdldragonflydoji, TA_CDLDRAGONFLYDOJI)
CDL_FUNC(cdlengulfing, TA_CDLENGULFING)
CDL_FUNC(cdlgapsidesidewhite, TA_CDLGAPSIDESIDEWHITE)
CDL_FUNC(cdlgravestonedoji, TA_CDLGRAVESTONEDOJI)
CDL_FUNC(cdlhammer, TA_CDLHAMMER)
CDL_FUNC(cdlhangingman, TA_CDLHANGINGMAN)
CDL_FUNC(cdlharami, TA_CDLHARAMI)
CDL_FUNC(cdlharamicross, TA_CDLHARAMICROSS)
CDL_FUNC(cdlhighwave, TA_CDLHIGHWAVE)
CDL_FUNC(cdlhikkake, TA_CDLHIKKAKE)
CDL_FUNC(cdlhikkakemod, TA_CDLHIKKAKEMOD)
CDL_FUNC(cdlhomingpigeon, TA_CDLHOMINGPIGEON)
CDL_FUNC(cdlidentical3crows, TA_CDLIDENTICAL3CROWS)
CDL_FUNC(cdlinneck, TA_CDLINNECK)
CDL_FUNC(cdlinvertedhammer, TA_CDLINVERTEDHAMMER)
CDL_FUNC(cdlkicking, TA_CDLKICKING)
CDL_FUNC(cdlkickingbylength, TA_CDLKICKINGBYLENGTH)
CDL_FUNC(cdlladderbottom, TA_CDLLADDERBOTTOM)
CDL_FUNC(cdllongleggeddoji, TA_CDLLONGLEGGEDDOJI)
CDL_FUNC(cdllongline, TA_CDLLONGLINE)
CDL_FUNC(cdlmarubozu, TA_CDLMARUBOZU)
CDL_FUNC(cdlmatchinglow, TA_CDLMATCHINGLOW)
CDL_FUNC(cdlonneck, TA_CDLONNECK)
CDL_FUNC(cdlpiercing, TA_CDLPIERCING)
CDL_FUNC(cdlrickshawman, TA_CDLRICKSHAWMAN)
CDL_FUNC(cdlrisefall3methods, TA_CDLRISEFALL3METHODS)
CDL_FUNC(cdlseparatinglines, TA_CDLSEPARATINGLINES)
CDL_FUNC(cdlshootingstar, TA_CDLSHOOTINGSTAR)
CDL_FUNC(cdlshortline, TA_CDLSHORTLINE)
CDL_FUNC(cdlspinningtop, TA_CDLSPINNINGTOP)
CDL_FUNC(cdlstalledpattern, TA_CDLSTALLEDPATTERN)
CDL_FUNC(cdlsticksandwich, TA_CDLSTICKSANDWICH)
CDL_FUNC(cdltakuri, TA_CDLTAKURI)
CDL_FUNC(cdltasukigap, TA_CDLTASUKIGAP)
CDL_FUNC(cdlthrusting, TA_CDLTHRUSTING)
CDL_FUNC(cdltristar, TA_CDLTRISTAR)
CDL_FUNC(cdlunique3river, TA_CDLUNIQUE3RIVER)
CDL_FUNC(cdlupsidegap2crows, TA_CDLUPSIDEGAP2CROWS)
CDL_FUNC(cdlxsidegap3methods, TA_CDLXSIDEGAP3METHODS)

// CDL functions with penetration parameter
CDL_FUNC_PEN(cdlabandonedbaby, TA_CDLABANDONEDBABY, 0.3)
CDL_FUNC_PEN(cdldarkcloudcover, TA_CDLDARKCLOUDCOVER, 0.5)
CDL_FUNC_PEN(cdleveningdojistar, TA_CDLEVENINGDOJISTAR, 0.3)
CDL_FUNC_PEN(cdleveningstar, TA_CDLEVENINGSTAR, 0.3)
CDL_FUNC_PEN(cdlmathold, TA_CDLMATHOLD, 0.5)
CDL_FUNC_PEN(cdlmorningdojistar, TA_CDLMORNINGDOJISTAR, 0.3)
CDL_FUNC_PEN(cdlmorningstar, TA_CDLMORNINGSTAR, 0.3)

#undef CDL_FUNC
#undef CDL_FUNC_PEN
