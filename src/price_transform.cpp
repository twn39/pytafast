// Price Transform: AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, MIDPRICE
#include "common.h"

// ---------------------------------------------------------
// AVERAGE PRICE
// ---------------------------------------------------------
DoubleArrayOUT avgprice(DoubleArrayIN inOpen, DoubleArrayIN inHigh,
                        DoubleArrayIN inLow, DoubleArrayIN inClose) {
  if (inOpen.size() == 0 || inHigh.size() == 0 || inLow.size() == 0 ||
      inClose.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("AVGPRICE", inOpen.shape(0),
                {inHigh.shape(0), inLow.shape(0), inClose.shape(0)});
  return apply_ta_func(inOpen.shape(0), TA_AVGPRICE_Lookback(), "TA_AVGPRICE",
                       [&](int *outBegIdx, int *outNBElement, double *outData) {
                         return TA_AVGPRICE(
                             0, gsl::narrow<int>(inOpen.shape(0) - 1),
                             inOpen.data(), inHigh.data(), inLow.data(),
                             inClose.data(), outBegIdx, outNBElement, outData);
                       });
}

// ---------------------------------------------------------
// MEDIAN PRICE
// ---------------------------------------------------------
DoubleArrayOUT medprice(DoubleArrayIN inHigh, DoubleArrayIN inLow) {
  if (inHigh.size() == 0 || inLow.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("MEDPRICE", inHigh.shape(0), {inLow.shape(0)});
  return apply_ta_func(inHigh.shape(0), TA_MEDPRICE_Lookback(), "TA_MEDPRICE",
                       [&](int *outBegIdx, int *outNBElement, double *outData) {
                         return TA_MEDPRICE(
                             0, gsl::narrow<int>(inHigh.shape(0) - 1),
                             inHigh.data(), inLow.data(), outBegIdx,
                             outNBElement, outData);
                       });
}

// ---------------------------------------------------------
// TYPICAL PRICE
// ---------------------------------------------------------
DoubleArrayOUT typprice(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        DoubleArrayIN inClose) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("TYPPRICE", inHigh.shape(0),
                {inLow.shape(0), inClose.shape(0)});
  return apply_ta_func(inHigh.shape(0), TA_TYPPRICE_Lookback(), "TA_TYPPRICE",
                       [&](int *outBegIdx, int *outNBElement, double *outData) {
                         return TA_TYPPRICE(
                             0, gsl::narrow<int>(inHigh.shape(0) - 1),
                             inHigh.data(), inLow.data(), inClose.data(),
                             outBegIdx, outNBElement, outData);
                       });
}

// ---------------------------------------------------------
// WEIGHTED CLOSE PRICE
// ---------------------------------------------------------
DoubleArrayOUT wclprice(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        DoubleArrayIN inClose) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("WCLPRICE", inHigh.shape(0),
                {inLow.shape(0), inClose.shape(0)});
  return apply_ta_func(inHigh.shape(0), TA_WCLPRICE_Lookback(), "TA_WCLPRICE",
                       [&](int *outBegIdx, int *outNBElement, double *outData) {
                         return TA_WCLPRICE(
                             0, gsl::narrow<int>(inHigh.shape(0) - 1),
                             inHigh.data(), inLow.data(), inClose.data(),
                             outBegIdx, outNBElement, outData);
                       });
}

// ---------------------------------------------------------
// MIDPRICE
// ---------------------------------------------------------
DoubleArrayOUT midprice(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("MIDPRICE", inHigh.shape(0), {inLow.shape(0)});
  return apply_ta_func(
      inHigh.shape(0), TA_MIDPRICE_Lookback(optInTimePeriod), "TA_MIDPRICE",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_MIDPRICE(0, gsl::narrow<int>(inHigh.shape(0) - 1),
                           inHigh.data(), inLow.data(), optInTimePeriod,
                           outBegIdx, outNBElement, outData);
      });
}
