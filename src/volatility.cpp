// Volatility: ATR, NATR, TRANGE, STDDEV
#include "common.h"

// ---------------------------------------------------------
// AVERAGE TRUE RANGE (ATR)
// ---------------------------------------------------------
DoubleArrayOUT atr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");

  size_t size = inHigh.shape(0);
  int lookback = TA_ATR_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ATR(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
               optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ATR");

  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// NORMALIZED AVERAGE TRUE RANGE (NATR)
// ---------------------------------------------------------
DoubleArrayOUT natr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                    DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");

  size_t size = inHigh.shape(0);
  int lookback = TA_NATR_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_NATR(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_NATR");

  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// TRUE RANGE (TRANGE)
// ---------------------------------------------------------
DoubleArrayOUT trange(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                      DoubleArrayIN inClose) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_TRANGE_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_TRANGE(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                  &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_TRANGE");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// STANDARD DEVIATION (STDDEV)
// ---------------------------------------------------------
DoubleArrayOUT stddev(DoubleArrayIN inReal, int optInTimePeriod = 5,
                      double optInNbDev = 1.0) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());

  size_t size = inReal.shape(0);
  int lookback = TA_STDDEV_Lookback(optInTimePeriod, optInNbDev);
  auto [outData, owner] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_STDDEV(0, size - 1, inReal.data(), optInTimePeriod, optInNbDev,
                        &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_STDDEV");

  return DoubleArrayOUT(outData, {size}, owner);
}
