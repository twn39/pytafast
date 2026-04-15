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
  check_lengths("BETA", inReal0.shape(0), {inReal1.shape(0)});
  return apply_ta_func(inReal0.shape(0), TA_BETA_Lookback(optInTimePeriod), "TA_BETA",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_BETA(0, gsl::narrow<int>(inReal0.shape(0) - 1), inReal0.data(),
                        inReal1.data(), optInTimePeriod, outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// PEARSON'S CORRELATION COEFFICIENT (CORREL)
// ---------------------------------------------------------
DoubleArrayOUT correl(DoubleArrayIN inReal0, DoubleArrayIN inReal1,
                      int optInTimePeriod = 30) {
  if (inReal0.size() == 0 || inReal1.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("CORREL", inReal0.shape(0), {inReal1.shape(0)});
  return apply_ta_func(inReal0.shape(0), TA_CORREL_Lookback(optInTimePeriod), "TA_CORREL",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_CORREL(0, gsl::narrow<int>(inReal0.shape(0) - 1), inReal0.data(),
                          inReal1.data(), optInTimePeriod, outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// LINEAR REGRESSION (LINEARREG)
// ---------------------------------------------------------
DoubleArrayOUT linearreg(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_LINEARREG_Lookback(optInTimePeriod), "TA_LINEARREG",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_LINEARREG(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                             optInTimePeriod, outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// LINEAR REGRESSION ANGLE (LINEARREG_ANGLE)
// ---------------------------------------------------------
DoubleArrayOUT linearreg_angle(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_LINEARREG_ANGLE_Lookback(optInTimePeriod), "TA_LINEARREG_ANGLE",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_LINEARREG_ANGLE(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                                   optInTimePeriod, outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// LINEAR REGRESSION INTERCEPT (LINEARREG_INTERCEPT)
// ---------------------------------------------------------
DoubleArrayOUT linearreg_intercept(DoubleArrayIN inReal,
                                   int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_LINEARREG_INTERCEPT_Lookback(optInTimePeriod), "TA_LINEARREG_INTERCEPT",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_LINEARREG_INTERCEPT(
          0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), optInTimePeriod,
          outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// LINEAR REGRESSION SLOPE (LINEARREG_SLOPE)
// ---------------------------------------------------------
DoubleArrayOUT linearreg_slope(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_LINEARREG_SLOPE_Lookback(optInTimePeriod), "TA_LINEARREG_SLOPE",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_LINEARREG_SLOPE(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                                   optInTimePeriod, outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// TIME SERIES FORECAST (TSF)
// ---------------------------------------------------------
DoubleArrayOUT tsf(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_TSF_Lookback(optInTimePeriod), "TA_TSF",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_TSF(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), optInTimePeriod,
               outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// VARIANCE (VAR)
// ---------------------------------------------------------
DoubleArrayOUT var(DoubleArrayIN inReal, int optInTimePeriod = 5,
                   double optInNbDev = 1.0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_VAR_Lookback(optInTimePeriod, optInNbDev), "TA_VAR",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_VAR(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), optInTimePeriod,
               optInNbDev, outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// AVERAGE DEVIATION (AVGDEV)
// ---------------------------------------------------------
DoubleArrayOUT avgdev(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_AVGDEV_Lookback(optInTimePeriod), "TA_AVGDEV",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_AVGDEV(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), optInTimePeriod,
                  outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// HIGHEST VALUE (MAX)
// ---------------------------------------------------------
DoubleArrayOUT ta_max(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_MAX_Lookback(optInTimePeriod), "TA_MAX",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_MAX(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), optInTimePeriod,
               outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// LOWEST VALUE (MIN)
// ---------------------------------------------------------
DoubleArrayOUT ta_min(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_MIN_Lookback(optInTimePeriod), "TA_MIN",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_MIN(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), optInTimePeriod,
               outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// SUMMATION (SUM)
// ---------------------------------------------------------
DoubleArrayOUT ta_sum(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(inReal.shape(0), TA_SUM_Lookback(optInTimePeriod), "TA_SUM",
    [&](int* outBegIdx, int* outNBElement, double* outData) {
      return TA_SUM(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(), optInTimePeriod,
               outBegIdx, outNBElement, outData);
    });
}

// ---------------------------------------------------------
// MINMAX - Lowest and Highest values over period
// ---------------------------------------------------------
nb::tuple minmax(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  return apply_ta_func_2out(inReal.shape(0), TA_MINMAX_Lookback(optInTimePeriod), "TA_MINMAX",
    [&](int* outBegIdx, int* outNBElement, double* out1, double* out2) {
      return TA_MINMAX(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                          optInTimePeriod, outBegIdx, outNBElement, out1, out2);
    });
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
  if (lookback < 0) {
    throw std::invalid_argument(
        "TA_MINMAXINDEX: Invalid parameter (lookback < 0)");
  }
  auto [outMinIdx, ownerMin] = alloc_int_output(size, lookback);
  auto [outMaxIdx, ownerMax] = alloc_int_output(size, lookback);

  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_MINMAXINDEX(0, gsl::narrow<int>(size - 1), inReal.data(),
                       optInTimePeriod, &outBegIdx, &outNBElement,
                       outMinIdx.get() + lookback, outMaxIdx.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MINMAXINDEX");

  using IntArrayOUT = nb::ndarray<int, nb::numpy, nb::ndim<1>>;
  return nb::make_tuple(IntArrayOUT(outMinIdx.get(), {size}, ownerMin),
                        IntArrayOUT(outMaxIdx.get(), {size}, ownerMax));
}
