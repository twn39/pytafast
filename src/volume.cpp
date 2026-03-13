#include "common.h"

// ---------------------------------------------------------
// ON BALANCE VOLUME (OBV)
// ---------------------------------------------------------
DoubleArrayOUT obv(DoubleArrayIN inReal, DoubleArrayIN inVolume) {
  if (inReal.size() == 0 || inVolume.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  if (inReal.shape(0) != inVolume.shape(0))
    throw std::runtime_error("Input lengths must match");

  size_t size = inReal.shape(0);
  int lookback = TA_OBV_Lookback();
  if (lookback < 0) {
    throw std::invalid_argument("TA_OBV: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    // outData is gsl::not_null<double*>, use .get() for arithmetic
    retCode =
        TA_OBV(0, gsl::narrow<int>(size - 1), inReal.data(), inVolume.data(),
               &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_OBV");

  return DoubleArrayOUT(outData.get(), {size}, owner);
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
  size_t size = inHigh.shape(0);
  int lookback = TA_AD_Lookback();
  if (lookback < 0) {
    throw std::invalid_argument("TA_AD: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_AD(0, gsl::narrow<int>(size - 1), inHigh.data(),
                      inLow.data(), inClose.data(), inVolume.data(), &outBegIdx,
                      &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_AD");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
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
  size_t size = inHigh.shape(0);
  int lookback = TA_ADOSC_Lookback(optInFastPeriod, optInSlowPeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_ADOSC: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_ADOSC(0, gsl::narrow<int>(size - 1), inHigh.data(),
                         inLow.data(), inClose.data(), inVolume.data(),
                         optInFastPeriod, optInSlowPeriod, &outBegIdx,
                         &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_ADOSC");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}
