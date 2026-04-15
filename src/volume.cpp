#include "common.h"

// ---------------------------------------------------------
// ON BALANCE VOLUME (OBV)
// ---------------------------------------------------------
DoubleArrayOUT obv(DoubleArrayIN inReal, DoubleArrayIN inVolume) {
  if (inReal.size() == 0 || inVolume.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  if (inReal.shape(0) != inVolume.shape(0))
    throw std::invalid_argument("Input lengths must match");

  return apply_ta_func(inReal.shape(0), TA_OBV_Lookback(), "TA_OBV",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_OBV(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), inVolume.data(), outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// CHAIKIN A/D LINE (AD)
// ---------------------------------------------------------
DoubleArrayOUT ad(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                  DoubleArrayIN inClose, DoubleArrayIN inVolume) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0 ||
      inVolume.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("AD", inHigh.shape(0),
                {inLow.shape(0), inClose.shape(0), inVolume.shape(0)});
  return apply_ta_func(inHigh.shape(0), TA_AD_Lookback(), "TA_AD",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_AD(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                      inLow.data(), inClose.data(), inVolume.data(), outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// CHAIKIN A/D OSCILLATOR (ADOSC)
// ---------------------------------------------------------
DoubleArrayOUT adosc(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                     DoubleArrayIN inClose, DoubleArrayIN inVolume,
                     int optInFastPeriod = 3, int optInSlowPeriod = 10) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0 ||
      inVolume.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("ADOSC", inHigh.shape(0),
                {inLow.shape(0), inClose.shape(0), inVolume.shape(0)});
  return apply_ta_func(inHigh.shape(0), TA_ADOSC_Lookback(optInFastPeriod, optInSlowPeriod), "TA_ADOSC",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_ADOSC(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                         inLow.data(), inClose.data(), inVolume.data(),
                         optInFastPeriod, optInSlowPeriod, outBegIdx, outNBElement, outData);
    });
}
