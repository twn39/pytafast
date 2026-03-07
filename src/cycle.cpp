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
  size_t size = inReal.shape(0);
  int lookback = TA_HT_DCPERIOD_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_HT_DCPERIOD(0, gsl::narrow<int>(size - 1), inReal.data(), &outBegIdx,
                       &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_HT_DCPERIOD");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - DOMINANT CYCLE PHASE (HT_DCPHASE)
// ---------------------------------------------------------
DoubleArrayOUT ht_dcphase(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_HT_DCPHASE_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_HT_DCPHASE(0, gsl::narrow<int>(size - 1), inReal.data(), &outBegIdx,
                      &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_HT_DCPHASE");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - PHASOR COMPONENTS (HT_PHASOR)
// ---------------------------------------------------------
nb::tuple ht_phasor(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_HT_PHASOR_Lookback();
  auto [outInPhase, ownerIP] = alloc_output(size, lookback);
  auto [outQuadrature, ownerQ] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_HT_PHASOR(
        0, gsl::narrow<int>(size - 1), inReal.data(), &outBegIdx, &outNBElement,
        outInPhase.get() + lookback, outQuadrature.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_HT_PHASOR");
  return nb::make_tuple(DoubleArrayOUT(outInPhase.get(), {size}, ownerIP),
                        DoubleArrayOUT(outQuadrature.get(), {size}, ownerQ));
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - SINE WAVE (HT_SINE)
// ---------------------------------------------------------
nb::tuple ht_sine(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_HT_SINE_Lookback();
  auto [outSine, ownerS] = alloc_output(size, lookback);
  auto [outLeadSine, ownerL] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_HT_SINE(0, gsl::narrow<int>(size - 1), inReal.data(),
                         &outBegIdx, &outNBElement, outSine.get() + lookback,
                         outLeadSine.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_HT_SINE");
  return nb::make_tuple(DoubleArrayOUT(outSine.get(), {size}, ownerS),
                        DoubleArrayOUT(outLeadSine.get(), {size}, ownerL));
}

// ---------------------------------------------------------
// HILBERT TRANSFORM - INSTANTANEOUS TRENDLINE (HT_TRENDLINE)
// ---------------------------------------------------------
DoubleArrayOUT ht_trendline(DoubleArrayIN inReal) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_HT_TRENDLINE_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_HT_TRENDLINE(0, gsl::narrow<int>(size - 1), inReal.data(),
                        &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_HT_TRENDLINE");
  return DoubleArrayOUT(outData.get(), {size}, owner);
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
