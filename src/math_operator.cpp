// Math Operators: ADD, SUB, MULT, DIV
#include "common.h"

// ---------------------------------------------------------
// VECTOR ARITHMETIC ADD
// ---------------------------------------------------------
DoubleArrayOUT add(DoubleArrayIN inReal0, DoubleArrayIN inReal1) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal0.shape(0);
  int lookback = TA_ADD_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_ADD(0, size - 1, inReal0.data(), inReal1.data(), &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ADD");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// VECTOR ARITHMETIC SUB
// ---------------------------------------------------------
DoubleArrayOUT sub(DoubleArrayIN inReal0, DoubleArrayIN inReal1) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal0.shape(0);
  int lookback = TA_SUB_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_SUB(0, size - 1, inReal0.data(), inReal1.data(), &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_SUB");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// VECTOR ARITHMETIC MULT
// ---------------------------------------------------------
DoubleArrayOUT mult(DoubleArrayIN inReal0, DoubleArrayIN inReal1) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal0.shape(0);
  int lookback = TA_MULT_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MULT(0, size - 1, inReal0.data(), inReal1.data(), &outBegIdx,
                      &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MULT");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// VECTOR ARITHMETIC DIV
// ---------------------------------------------------------
DoubleArrayOUT ta_div(DoubleArrayIN inReal0, DoubleArrayIN inReal1) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal0.shape(0);
  int lookback = TA_DIV_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_DIV(0, size - 1, inReal0.data(), inReal1.data(), &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_DIV");
  return DoubleArrayOUT(outData, {size}, owner);
}
