// Overlap Studies: SMA, EMA, BBANDS, DEMA, KAMA, MA, T3, TEMA, TRIMA, WMA,
// SAR, MIDPOINT
#include "common.h"

// ---------------------------------------------------------
// SIMPLE MOVING AVERAGE
// ---------------------------------------------------------
DoubleArrayOUT sma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }

  size_t size = inReal.shape(0);
  int lookback = TA_SMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_SMA(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }

  check_ta_retcode(retCode, "TA_SMA");

  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// EXPONENTIAL MOVING AVERAGE
// ---------------------------------------------------------
DoubleArrayOUT ema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }

  size_t size = inReal.shape(0);
  int lookback = TA_EMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_EMA(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_EMA");

  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// BOLLINGER BANDS
// ---------------------------------------------------------
nb::tuple bbands(DoubleArrayIN inReal, int optInTimePeriod = 5,
                 double optInNbDevUp = 2.0, double optInNbDevDn = 2.0,
                 int optInMAType = 0) {
  if (inReal.size() == 0) {
    return nb::make_tuple(DoubleArrayOUT(nullptr, {0}, nb::handle()),
                          DoubleArrayOUT(nullptr, {0}, nb::handle()),
                          DoubleArrayOUT(nullptr, {0}, nb::handle()));
  }

  size_t size = inReal.shape(0);
  int lookback = TA_BBANDS_Lookback(optInTimePeriod, optInNbDevUp, optInNbDevDn,
                                    (TA_MAType)optInMAType);

  auto [outUpper, owner1] = alloc_output(size, lookback);
  auto [outMiddle, owner2] = alloc_output(size, lookback);
  auto [outLower, owner3] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_BBANDS(0, size - 1, inReal.data(), optInTimePeriod,
                        optInNbDevUp, optInNbDevDn, (TA_MAType)optInMAType,
                        &outBegIdx, &outNBElement, outUpper + lookback,
                        outMiddle + lookback, outLower + lookback);
  }
  check_ta_retcode(retCode, "TA_BBANDS");

  return nb::make_tuple(DoubleArrayOUT(outUpper, {size}, owner1),
                        DoubleArrayOUT(outMiddle, {size}, owner2),
                        DoubleArrayOUT(outLower, {size}, owner3));
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
    retCode = TA_DEMA(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_DEMA");
  return DoubleArrayOUT(outData, {size}, owner);
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
    retCode = TA_KAMA(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_KAMA");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MOVING AVERAGE (MA) - generic
// ---------------------------------------------------------
DoubleArrayOUT ma(DoubleArrayIN inReal, int optInTimePeriod = 30,
                  int optInMAType = 0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MA_Lookback(optInTimePeriod, (TA_MAType)optInMAType);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MA(0, size - 1, inReal.data(), optInTimePeriod,
                    (TA_MAType)optInMAType, &outBegIdx, &outNBElement,
                    outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MA");
  return DoubleArrayOUT(outData, {size}, owner);
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
    retCode = TA_T3(0, size - 1, inReal.data(), optInTimePeriod, optInVFactor,
                    &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_T3");
  return DoubleArrayOUT(outData, {size}, owner);
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
    retCode = TA_TEMA(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_TEMA");
  return DoubleArrayOUT(outData, {size}, owner);
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
    retCode = TA_TRIMA(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                       &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_TRIMA");
  return DoubleArrayOUT(outData, {size}, owner);
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
    retCode = TA_WMA(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_WMA");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// PARABOLIC SAR
// ---------------------------------------------------------
DoubleArrayOUT sar(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   double optInAcceleration = 0.02, double optInMaximum = 0.2) {
  if (inHigh.size() == 0 || inLow.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  if (inHigh.shape(0) != inLow.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_SAR_Lookback(optInAcceleration, optInMaximum);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_SAR(0, size - 1, inHigh.data(), inLow.data(), optInAcceleration,
               optInMaximum, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_SAR");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MIDPOINT
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
    retCode = TA_MIDPOINT(0, size - 1, inReal.data(), optInTimePeriod,
                          &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MIDPOINT");
  return DoubleArrayOUT(outData, {size}, owner);
}
