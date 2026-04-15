// Math Operators: ADD, SUB, MULT, DIV
#include "common.h"

// ---------------------------------------------------------
// VECTOR ARITHMETIC ADD
// ---------------------------------------------------------
DoubleArrayOUT add(DoubleArrayIN inReal0, DoubleArrayIN inReal1) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal0.shape(0), TA_ADD_Lookback(), "TA_ADD",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_ADD(0, gsl::narrow<int>(inReal0.shape(0) - 1), inReal0.data(), inReal1.data(),
               outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// VECTOR ARITHMETIC SUB
// ---------------------------------------------------------
DoubleArrayOUT sub(DoubleArrayIN inReal0, DoubleArrayIN inReal1) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal0.shape(0), TA_SUB_Lookback(), "TA_SUB",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_SUB(0, gsl::narrow<int>(inReal0.shape(0) - 1), inReal0.data(), inReal1.data(),
               outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// VECTOR ARITHMETIC MULT
// ---------------------------------------------------------
DoubleArrayOUT mult(DoubleArrayIN inReal0, DoubleArrayIN inReal1) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal0.shape(0), TA_MULT_Lookback(), "TA_MULT",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_MULT(0, gsl::narrow<int>(inReal0.shape(0) - 1), inReal0.data(), inReal1.data(),
                outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// VECTOR ARITHMETIC DIV
// ---------------------------------------------------------
DoubleArrayOUT ta_div(DoubleArrayIN inReal0, DoubleArrayIN inReal1) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal0.shape(0), TA_DIV_Lookback(), "TA_DIV",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_DIV(0, gsl::narrow<int>(inReal0.shape(0) - 1), inReal0.data(), inReal1.data(),
               outBegIdx, outNBElement, outData);
    });
}
