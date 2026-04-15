// Cycle Indicators: HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE,
// HT_TRENDLINE, HT_TRENDMODE
#include "common.h"

// ---------------------------------------------------------
// HILBERT TRANSFORM - DOMINANT CYCLE PERIOD (HT_DCPERIOD)
// ---------------------------------------------------------
DoubleArrayOUT ht_dcperiod(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_HT_DCPERIOD_Lookback(), "TA_HT_DCPERIOD",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_HT_DCPERIOD(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - DOMINANT CYCLE PHASE (HT_DCPHASE)
// ---------------------------------------------------------
DoubleArrayOUT ht_dcphase(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_HT_DCPHASE_Lookback(), "TA_HT_DCPHASE",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_HT_DCPHASE(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - PHASOR COMPONENTS (HT_PHASOR)
// ---------------------------------------------------------
nb::tuple ht_phasor(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  return apply_ta_func_2out(inReal.shape(0), TA_HT_PHASOR_Lookback(), "TA_HT_PHASOR",
    [&](int* outBegIdx, int* outNBElement, double* out1, double* out2) {
      return TA_HT_PHASOR(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), outBegIdx, outNBElement, out1, out2);
    });
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - SINE WAVE (HT_SINE)
// ---------------------------------------------------------
nb::tuple ht_sine(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  return apply_ta_func_2out(inReal.shape(0), TA_HT_SINE_Lookback(), "TA_HT_SINE",
    [&](int* outBegIdx, int* outNBElement, double* out1, double* out2) {
      return TA_HT_SINE(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                           outBegIdx, outNBElement, out1, out2);
    });
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - INSTANTANEOUS TRENDLINE (HT_TRENDLINE)
// ---------------------------------------------------------
DoubleArrayOUT ht_trendline(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_HT_TRENDLINE_Lookback(), "TA_HT_TRENDLINE",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_HT_TRENDLINE(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                        outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - TREND VS CYCLE MODE (HT_TRENDMODE)
// Returns integer array (0=cycle, 1=trend)
// ---------------------------------------------------------
nb::ndarray<int, nb::numpy, nb::ndim<1>> ht_trendmode(DoubleArrayIN inReal) {
  using IntArrayOUT = nb::ndarray<int, nb::numpy, nb::ndim<1>>;
  if (inReal.size() == 0) return IntArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_HT_TRENDMODE_Lookback();
  if (lookback < 0) {
    throw std::invalid_argument(
        "TA_HT_TRENDMODE: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_int_output(size, lookback);

  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_HT_TRENDMODE(0, gsl::narrow<int>(size - 1), inReal.data(),
                        &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_HT_TRENDMODE");
  return IntArrayOUT(outData.get(), {size}, owner);
}
