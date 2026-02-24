// Price Transform: AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, MIDPRICE
#include "common.h"

// ---------------------------------------------------------
// AVERAGE PRICE
// ---------------------------------------------------------
DoubleArrayOUT avgprice(DoubleArrayIN inOpen, DoubleArrayIN inHigh,
                        DoubleArrayIN inLow, DoubleArrayIN inClose) {
  if (inOpen.size() == 0 || inHigh.size() == 0 || inLow.size() == 0 ||
      inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inOpen.shape(0);
  int lookback = TA_AVGPRICE_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_AVGPRICE(0, size - 1, inOpen.data(), inHigh.data(),
                          inLow.data(), inClose.data(), &outBegIdx,
                          &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_AVGPRICE");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MEDIAN PRICE
// ---------------------------------------------------------
DoubleArrayOUT medprice(DoubleArrayIN inHigh, DoubleArrayIN inLow) {
  if (inHigh.size() == 0 || inLow.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_MEDPRICE_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MEDPRICE(0, size - 1, inHigh.data(), inLow.data(), &outBegIdx,
                          &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MEDPRICE");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// TYPICAL PRICE
// ---------------------------------------------------------
DoubleArrayOUT typprice(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        DoubleArrayIN inClose) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_TYPPRICE_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_TYPPRICE(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                    &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_TYPPRICE");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// WEIGHTED CLOSE PRICE
// ---------------------------------------------------------
DoubleArrayOUT wclprice(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        DoubleArrayIN inClose) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_WCLPRICE_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_WCLPRICE(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                    &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_WCLPRICE");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MIDPRICE
// ---------------------------------------------------------
DoubleArrayOUT midprice(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_MIDPRICE_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_MIDPRICE(0, size - 1, inHigh.data(), inLow.data(), optInTimePeriod,
                    &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MIDPRICE");
  return DoubleArrayOUT(outData, {size}, owner);
}
