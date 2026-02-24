// Volume Indicators: OBV
#include "common.h"

// ---------------------------------------------------------
// ON BALANCE VOLUME (OBV)
// ---------------------------------------------------------
DoubleArrayOUT obv(DoubleArrayIN inReal, DoubleArrayIN inVolume) {
  if (inReal.size() == 0 || inVolume.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inReal.shape(0) != inVolume.shape(0))
    throw std::runtime_error("Input lengths must match");

  size_t size = inReal.shape(0);
  int lookback = TA_OBV_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_OBV(0, size - 1, inReal.data(), inVolume.data(), &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_OBV");

  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// CHAIKIN A/D LINE (AD)
// ---------------------------------------------------------
DoubleArrayOUT ad(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                  DoubleArrayIN inClose, DoubleArrayIN inVolume) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0 ||
      inVolume.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inHigh.shape(0);
  int lookback = TA_AD_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_AD(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
              inVolume.data(), &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_AD");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// CHAIKIN A/D OSCILLATOR (ADOSC)
// ---------------------------------------------------------
DoubleArrayOUT adosc(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                     DoubleArrayIN inClose, DoubleArrayIN inVolume,
                     int optInFastPeriod = 3, int optInSlowPeriod = 10) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0 ||
      inVolume.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inHigh.shape(0);
  int lookback = TA_ADOSC_Lookback(optInFastPeriod, optInSlowPeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_ADOSC(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                       inVolume.data(), optInFastPeriod, optInSlowPeriod,
                       &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ADOSC");
  return DoubleArrayOUT(outData, {size}, owner);
}
