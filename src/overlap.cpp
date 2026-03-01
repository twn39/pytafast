#include "common.h"

// ---------------------------------------------------------
// SIMPLE MOVING AVERAGE (SMA)
// ---------------------------------------------------------
DoubleArrayOUT sma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_SMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_SMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_SMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// EXPONENTIAL MOVING AVERAGE (EMA)
// ---------------------------------------------------------
DoubleArrayOUT ema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_EMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_EMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_EMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// BOLLINGER BANDS (BBANDS)
// ---------------------------------------------------------
nb::tuple bbands(DoubleArrayIN inReal, int optInTimePeriod = 5,
                 double optInNbDevUp = 2.0, double optInNbDevDn = 2.0,
                 int optInMAType = 0) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback =
      TA_BBANDS_Lookback(optInTimePeriod, optInNbDevUp, optInNbDevDn,
                         static_cast<TA_MAType>(optInMAType));
  auto [outUpper, ownerU] = alloc_output(size, lookback);
  auto [outMiddle, ownerM] = alloc_output(size, lookback);
  auto [outLower, ownerL] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_BBANDS(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
                        optInNbDevUp, optInNbDevDn,
                        static_cast<TA_MAType>(optInMAType), &outBegIdx,
                        &outNBElement, outUpper.get() + lookback,
                        outMiddle.get() + lookback, outLower.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_BBANDS");
  return nb::make_tuple(DoubleArrayOUT(outUpper.get(), {size}, ownerU),
                        DoubleArrayOUT(outMiddle.get(), {size}, ownerM),
                        DoubleArrayOUT(outLower.get(), {size}, ownerL));
}

// ---------------------------------------------------------
// DOUBLE EXPONENTIAL MOVING AVERAGE (DEMA)
// ---------------------------------------------------------
DoubleArrayOUT dema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_DEMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_DEMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_DEMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA)
// ---------------------------------------------------------
DoubleArrayOUT kama(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_KAMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_KAMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_KAMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// MESA ADAPTIVE MOVING AVERAGE (MAMA)
// ---------------------------------------------------------
nb::tuple mama(DoubleArrayIN inReal, double optInFastLimit = 0.5,
               double optInSlowLimit = 0.05) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MAMA_Lookback(optInFastLimit, optInSlowLimit);
  auto [outMAMA, ownerM] = alloc_output(size, lookback);
  auto [outFAMA, ownerF] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MAMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInFastLimit,
                      optInSlowLimit, &outBegIdx, &outNBElement,
                      outMAMA.get() + lookback, outFAMA.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MAMA");
  return nb::make_tuple(DoubleArrayOUT(outMAMA.get(), {size}, ownerM),
                        DoubleArrayOUT(outFAMA.get(), {size}, ownerF));
}

// ---------------------------------------------------------
// MOVING AVERAGE (MA)
// ---------------------------------------------------------
DoubleArrayOUT ma(DoubleArrayIN inReal, int optInTimePeriod = 30,
                  int optInMAType = 0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback =
      TA_MA_Lookback(optInTimePeriod, static_cast<TA_MAType>(optInMAType));
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
                    static_cast<TA_MAType>(optInMAType), &outBegIdx,
                    &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// MIDPOINT OVER PERIOD (MIDPOINT)
// ---------------------------------------------------------
DoubleArrayOUT midpoint(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MIDPOINT_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MIDPOINT(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                          &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MIDPOINT");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// PARABOLIC SAR (SAR)
// ---------------------------------------------------------
DoubleArrayOUT sar(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   double optInAcceleration = 0.02, double optInMaximum = 0.2) {
  if (inHigh.size() == 0 || inLow.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inHigh.shape(0);
  int lookback = TA_SAR_Lookback(optInAcceleration, optInMaximum);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_SAR(0, gsl::narrow<int>(size - 1), inHigh.data(), inLow.data(), optInAcceleration,
                     optInMaximum, &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_SAR");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// TRIPLE EXPONENTIAL MOVING AVERAGE (T3)
// ---------------------------------------------------------
DoubleArrayOUT t3(DoubleArrayIN inReal, int optInTimePeriod = 5,
                  double optInVFactor = 0.7) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_T3_Lookback(optInTimePeriod, optInVFactor);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_T3(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, optInVFactor,
                    &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_T3");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// TRIPLE EXPONENTIAL MOVING AVERAGE (TEMA)
// ---------------------------------------------------------
DoubleArrayOUT tema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_TEMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_TEMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_TEMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// TRIANGULAR MOVING AVERAGE (TRIMA)
// ---------------------------------------------------------
DoubleArrayOUT trima(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_TRIMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_TRIMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                       &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_TRIMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// WEIGHTED MOVING AVERAGE (WMA)
// ---------------------------------------------------------
DoubleArrayOUT wma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_WMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_WMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_WMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}
