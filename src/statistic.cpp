// Statistic Functions: BETA, CORREL, LINEARREG, LINEARREG_ANGLE,
// LINEARREG_INTERCEPT, LINEARREG_SLOPE, TSF, VAR, AVGDEV
#include "common.h"

// ---------------------------------------------------------
// BETA
// ---------------------------------------------------------
DoubleArrayOUT beta(DoubleArrayIN inReal0, DoubleArrayIN inReal1,
                    int optInTimePeriod = 5) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal0.shape(0);
  int lookback = TA_BETA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_BETA(0, size - 1, inReal0.data(), inReal1.data(), optInTimePeriod,
                &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_BETA");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// PEARSON'S CORRELATION COEFFICIENT (CORREL)
// ---------------------------------------------------------
DoubleArrayOUT correl(DoubleArrayIN inReal0, DoubleArrayIN inReal1,
                      int optInTimePeriod = 30) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal0.shape(0);
  int lookback = TA_CORREL_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_CORREL(0, size - 1, inReal0.data(), inReal1.data(), optInTimePeriod,
                  &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_CORREL");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// LINEAR REGRESSION (LINEARREG)
// ---------------------------------------------------------
DoubleArrayOUT linearreg(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_LINEARREG_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_LINEARREG(0, size - 1, inReal.data(), optInTimePeriod,
                           &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_LINEARREG");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// LINEAR REGRESSION ANGLE (LINEARREG_ANGLE)
// ---------------------------------------------------------
DoubleArrayOUT linearreg_angle(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_LINEARREG_ANGLE_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_LINEARREG_ANGLE(0, size - 1, inReal.data(), optInTimePeriod,
                                 &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_LINEARREG_ANGLE");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// LINEAR REGRESSION INTERCEPT (LINEARREG_INTERCEPT)
// ---------------------------------------------------------
DoubleArrayOUT linearreg_intercept(DoubleArrayIN inReal,
                                   int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_LINEARREG_INTERCEPT_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_LINEARREG_INTERCEPT(0, size - 1, inReal.data(), optInTimePeriod,
                               &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_LINEARREG_INTERCEPT");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// LINEAR REGRESSION SLOPE (LINEARREG_SLOPE)
// ---------------------------------------------------------
DoubleArrayOUT linearreg_slope(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_LINEARREG_SLOPE_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_LINEARREG_SLOPE(0, size - 1, inReal.data(), optInTimePeriod,
                                 &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_LINEARREG_SLOPE");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// TIME SERIES FORECAST (TSF)
// ---------------------------------------------------------
DoubleArrayOUT tsf(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_TSF_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_TSF(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_TSF");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// VARIANCE (VAR)
// ---------------------------------------------------------
DoubleArrayOUT var(DoubleArrayIN inReal, int optInTimePeriod = 5,
                   double optInNbDev = 1.0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_VAR_Lookback(optInTimePeriod, optInNbDev);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_VAR(0, size - 1, inReal.data(), optInTimePeriod, optInNbDev,
                     &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_VAR");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// AVERAGE DEVIATION (AVGDEV)
// ---------------------------------------------------------
DoubleArrayOUT avgdev(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_AVGDEV_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_AVGDEV(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                        &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_AVGDEV");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// HIGHEST VALUE (MAX)
// ---------------------------------------------------------
DoubleArrayOUT ta_max(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MAX_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MAX(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MAX");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// LOWEST VALUE (MIN)
// ---------------------------------------------------------
DoubleArrayOUT ta_min(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MIN_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MIN(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MIN");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// SUMMATION (SUM)
// ---------------------------------------------------------
DoubleArrayOUT ta_sum(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_SUM_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_SUM(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_SUM");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MINMAX - Lowest and Highest values over period
// ---------------------------------------------------------
nb::tuple minmax(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MINMAX_Lookback(optInTimePeriod);
  auto [outMin, ownerMin] = alloc_output(size, lookback);
  auto [outMax, ownerMax] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MINMAX(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                        &outNBElement, outMin + lookback, outMax + lookback);
  }
  check_ta_retcode(retCode, "TA_MINMAX");
  return nb::make_tuple(DoubleArrayOUT(outMin, {size}, ownerMin),
                        DoubleArrayOUT(outMax, {size}, ownerMax));
}

// ---------------------------------------------------------
// MINMAXINDEX - Indexes of lowest and highest values
// ---------------------------------------------------------
nb::tuple minmaxindex(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    using IntArrayOUT = nb::ndarray<int, nb::numpy, nb::ndim<1>>;
    auto emptyMin = IntArrayOUT(nullptr, {0}, nb::handle());
    auto emptyMax = IntArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(emptyMin, emptyMax);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MINMAXINDEX_Lookback(optInTimePeriod);

  // Allocate int arrays for index output with NaN-like fill
  int *outMinIdx = new int[size];
  int *outMaxIdx = new int[size];
  for (size_t i = 0; i < (size_t)lookback && i < size; ++i) {
    outMinIdx[i] = -1;
    outMaxIdx[i] = -1;
  }

  nb::capsule ownerMin(outMinIdx, [](void *p) noexcept { delete[] (int *)p; });
  nb::capsule ownerMax(outMaxIdx, [](void *p) noexcept { delete[] (int *)p; });

  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MINMAXINDEX(0, size - 1, inReal.data(), optInTimePeriod,
                             &outBegIdx, &outNBElement, outMinIdx + lookback,
                             outMaxIdx + lookback);
  }
  check_ta_retcode(retCode, "TA_MINMAXINDEX");

  using IntArrayOUT = nb::ndarray<int, nb::numpy, nb::ndim<1>>;
  return nb::make_tuple(IntArrayOUT(outMinIdx, {size}, ownerMin),
                        IntArrayOUT(outMaxIdx, {size}, ownerMax));
}
