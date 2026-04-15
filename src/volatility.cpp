// Volatility: ATR, NATR, TRANGE, STDDEV
#include "common.h"

// ---------------------------------------------------------
// AVERAGE TRUE RANGE (ATR)
// ---------------------------------------------------------
DoubleArrayOUT atr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::invalid_argument("Input lengths must match");

  return apply_ta_func(inHigh.shape(0), TA_ATR_Lookback(optInTimePeriod), "TA_ATR",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_ATR(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(), inLow.data(),
                     inClose.data(), optInTimePeriod, outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// NORMALIZED AVERAGE TRUE RANGE (NATR)
// ---------------------------------------------------------
DoubleArrayOUT natr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                    DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::invalid_argument("Input lengths must match");

  return apply_ta_func(inHigh.shape(0), TA_NATR_Lookback(optInTimePeriod), "TA_NATR",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_NATR(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                      inLow.data(), inClose.data(), optInTimePeriod, outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// TRUE RANGE (TRANGE)
// ---------------------------------------------------------
DoubleArrayOUT trange(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                      DoubleArrayIN inClose) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::invalid_argument("Input lengths must match");
  return apply_ta_func(inHigh.shape(0), TA_TRANGE_Lookback(), "TA_TRANGE",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_TRANGE(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                          inLow.data(), inClose.data(), outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// STANDARD DEVIATION (STDDEV)
// ---------------------------------------------------------
DoubleArrayOUT stddev(DoubleArrayIN inReal, int optInTimePeriod = 5,
                      double optInNbDev = 1.0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }

  return apply_ta_func(inReal.shape(0), TA_STDDEV_Lookback(optInTimePeriod, optInNbDev), "TA_STDDEV",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_STDDEV(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                        optInTimePeriod, optInNbDev, outBegIdx, outNBElement, outData);
    });
}
