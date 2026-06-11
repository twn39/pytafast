#include "common.h"
#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------
// SIMPLE MOVING AVERAGE (SMA)
// ---------------------------------------------------------
DoubleArrayOUT sma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_SMA_Lookback(optInTimePeriod), "TA_SMA",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_SMA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// EXPONENTIAL MOVING AVERAGE (EMA)
// ---------------------------------------------------------
DoubleArrayOUT ema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_EMA_Lookback(optInTimePeriod), "TA_EMA",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_EMA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// BOLLINGER BANDS (BBANDS)
// ---------------------------------------------------------
nb::tuple bbands(DoubleArrayIN inReal, int optInTimePeriod = 5,
                 double optInNbDevUp = 2.0, double optInNbDevDn = 2.0,
                 int optInMAType = 0) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty, empty);
  }
  return apply_ta_func_3out(
      inReal.shape(0),
      TA_BBANDS_Lookback(optInTimePeriod, optInNbDevUp, optInNbDevDn,
                         static_cast<TA_MAType>(optInMAType)),
      "TA_BBANDS",
      [&](int *outBegIdx, int *outNBElement, double *outUpper,
          double *outMiddle, double *outLower) {
        return TA_BBANDS(0, gsl::narrow<int>(inReal.shape(0) - 1),
                         inReal.data(), optInTimePeriod, optInNbDevUp,
                         optInNbDevDn, static_cast<TA_MAType>(optInMAType),
                         outBegIdx, outNBElement, outUpper, outMiddle,
                         outLower);
      });
}

// ---------------------------------------------------------
// DOUBLE EXPONENTIAL MOVING AVERAGE (DEMA)
// ---------------------------------------------------------
DoubleArrayOUT dema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_DEMA_Lookback(optInTimePeriod), "TA_DEMA",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_DEMA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                       optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA)
// ---------------------------------------------------------
DoubleArrayOUT kama(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_KAMA_Lookback(optInTimePeriod), "TA_KAMA",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_KAMA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                       optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// MESA ADAPTIVE MOVING AVERAGE (MAMA)
// ---------------------------------------------------------
nb::tuple mama(DoubleArrayIN inReal, double optInFastLimit = 0.5,
               double optInSlowLimit = 0.05) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  return apply_ta_func_2out(
      inReal.shape(0), TA_MAMA_Lookback(optInFastLimit, optInSlowLimit),
      "TA_MAMA",
      [&](int *outBegIdx, int *outNBElement, double *outMAMA, double *outFAMA) {
        return TA_MAMA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                       optInFastLimit, optInSlowLimit, outBegIdx, outNBElement,
                       outMAMA, outFAMA);
      });
}

// ---------------------------------------------------------
// MOVING AVERAGE (MA)
// ---------------------------------------------------------
DoubleArrayOUT ma(DoubleArrayIN inReal, int optInTimePeriod = 30,
                  int optInMAType = 0) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0),
      TA_MA_Lookback(optInTimePeriod, static_cast<TA_MAType>(optInMAType)),
      "TA_MA", [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_MA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                     optInTimePeriod, static_cast<TA_MAType>(optInMAType),
                     outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// MIDPOINT OVER PERIOD (MIDPOINT)
// ---------------------------------------------------------
DoubleArrayOUT midpoint(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_MIDPOINT_Lookback(optInTimePeriod), "TA_MIDPOINT",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_MIDPOINT(0, gsl::narrow<int>(inReal.shape(0) - 1),
                           inReal.data(), optInTimePeriod, outBegIdx,
                           outNBElement, outData);
      });
}

// ---------------------------------------------------------
// PARABOLIC SAR (SAR)
// ---------------------------------------------------------
DoubleArrayOUT sar(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   double optInAcceleration = 0.02, double optInMaximum = 0.2) {
  if (inHigh.size() == 0 || inLow.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  check_lengths("SAR", inHigh.shape(0), {inLow.shape(0)});
  return apply_ta_func(
      inHigh.shape(0), TA_SAR_Lookback(optInAcceleration, optInMaximum),
      "TA_SAR", [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_SAR(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                      inLow.data(), optInAcceleration, optInMaximum, outBegIdx,
                      outNBElement, outData);
      });
}

// ---------------------------------------------------------
// TRIPLE EXPONENTIAL MOVING AVERAGE (T3)
// ---------------------------------------------------------
DoubleArrayOUT t3(DoubleArrayIN inReal, int optInTimePeriod = 5,
                  double optInVFactor = 0.7) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_T3_Lookback(optInTimePeriod, optInVFactor), "TA_T3",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_T3(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                     optInTimePeriod, optInVFactor, outBegIdx, outNBElement,
                     outData);
      });
}

// ---------------------------------------------------------
// TRIPLE EXPONENTIAL MOVING AVERAGE (TEMA)
// ---------------------------------------------------------
DoubleArrayOUT tema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_TEMA_Lookback(optInTimePeriod), "TA_TEMA",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_TEMA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                       optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// TRIANGULAR MOVING AVERAGE (TRIMA)
// ---------------------------------------------------------
DoubleArrayOUT trima(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_TRIMA_Lookback(optInTimePeriod), "TA_TRIMA",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_TRIMA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                        optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// WEIGHTED MOVING AVERAGE (WMA)
// ---------------------------------------------------------
DoubleArrayOUT wma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  return apply_ta_func(
      inReal.shape(0), TA_WMA_Lookback(optInTimePeriod), "TA_WMA",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_WMA(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// ZIGZAG INDICATOR (ZIGZAG)
// ---------------------------------------------------------
DoubleArrayOUT zigzag(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                      double change = 10.0, bool percent = true) {
  if (inHigh.size() == 0 || inLow.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("ZIGZAG", inHigh.shape(0), {inLow.shape(0)});
  size_t size = inHigh.shape(0);
  auto [outData, owner] = alloc_output(size, 0);
  gsl::span<double> out(outData.get(), size);
  std::fill(out.begin(), out.end(), NaN);

  if (size < 2) {
    return DoubleArrayOUT(out.data(), {size}, std::move(owner));
  }

  gsl::span<const double> hi(inHigh.data(), size);
  gsl::span<const double> lo(inLow.data(), size);

  {
    nb::gil_scoped_release release;

    int last_idx = 0;
    double last_val = (hi[0] + lo[0]) / 2.0;
    int trend = 0;

    out[0] = last_val;

    for (size_t i = 1; i < size; ++i) {
      if (trend == 0) {
        double up_diff =
            percent ? (hi[i] / last_val - 1.0) * 100.0 : (hi[i] - last_val);
        double down_diff =
            percent ? (last_val / lo[i] - 1.0) * 100.0 : (last_val - lo[i]);
        if (up_diff >= change) {
          trend = 1;
          last_val = hi[i];
          last_idx = gsl::narrow<int>(i);
          out[i] = last_val;
        } else if (down_diff >= change) {
          trend = -1;
          last_val = lo[i];
          last_idx = gsl::narrow<int>(i);
          out[i] = last_val;
        }
      } else if (trend == 1) {
        if (hi[i] > last_val) {
          out[last_idx] = NaN;
          last_val = hi[i];
          last_idx = gsl::narrow<int>(i);
          out[i] = last_val;
        } else if ((percent ? (last_val / lo[i] - 1.0) * 100.0
                            : (last_val - lo[i])) >= change) {
          trend = -1;
          last_val = lo[i];
          last_idx = gsl::narrow<int>(i);
          out[i] = last_val;
        }
      } else if (trend == -1) {
        if (lo[i] < last_val) {
          out[last_idx] = NaN;
          last_val = lo[i];
          last_idx = gsl::narrow<int>(i);
          out[i] = last_val;
        } else if ((percent ? (hi[i] / last_val - 1.0) * 100.0
                            : (hi[i] - last_val)) >= change) {
          trend = 1;
          last_val = hi[i];
          last_idx = gsl::narrow<int>(i);
          out[i] = last_val;
        }
      }
    }

    int prev_idx = 0;
    for (size_t i = 1; i < size; ++i) {
      if (!std::isnan(out[i])) {
        double start_val = out[prev_idx];
        double end_val = out[i];
        double step = (end_val - start_val) /
                      static_cast<double>(gsl::narrow<int>(i) - prev_idx);
        for (int j = prev_idx + 1; j < gsl::narrow<int>(i); ++j) {
          out[j] = start_val + (step * static_cast<double>(j - prev_idx));
        }
        prev_idx = gsl::narrow<int>(i);
      }
    }
    if (prev_idx < gsl::narrow<int>(size) - 1) {
      double start_val = out[prev_idx];
      double end_val;
      if (trend == 1) {
        end_val = hi[size - 1];
      } else if (trend == -1) {
        end_val = lo[size - 1];
      } else {
        end_val = (hi[size - 1] + lo[size - 1]) / 2.0;
      }
      double step = (end_val - start_val) /
                    static_cast<double>(gsl::narrow<int>(size) - 1 - prev_idx);
      for (size_t j = static_cast<size_t>(prev_idx) + 1; j < size; ++j) {
        out[j] = start_val +
                 (step * static_cast<double>(gsl::narrow<int>(j) - prev_idx));
      }
    }
  }

  return DoubleArrayOUT(out.data(), {size}, std::move(owner));
}

// ---------------------------------------------------------
// ARNAUD LEGOUX MOVING AVERAGE (ALMA)
// ---------------------------------------------------------
DoubleArrayOUT alma(DoubleArrayIN inReal, int optInTimePeriod = 9,
                    double optInOffset = 0.85, double optInSigma = 6.0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = optInTimePeriod - 1;
  auto [outData, owner] = alloc_output(size, lookback);
  gsl::span<double> out(outData.get(), size);
  if (size < static_cast<size_t>(optInTimePeriod)) {
    return DoubleArrayOUT(out.data(), {size}, std::move(owner));
  }

  std::vector<double> wts(optInTimePeriod);
  double m = std::floor(optInOffset * static_cast<double>(optInTimePeriod - 1));
  double s = static_cast<double>(optInTimePeriod) / optInSigma;
  double sum_w = 0.0;
  for (int i = 0; i < optInTimePeriod; ++i) {
    wts[i] = std::exp(-std::pow(static_cast<double>(i) - m, 2) / (2.0 * s * s));
    sum_w += wts[i];
  }
  if (sum_w != 0) {
    for (double &w : wts) {
      w /= sum_w;
    }
  }

  gsl::span<const double> in(inReal.data(), size);

  {
    nb::gil_scoped_release release;

    for (size_t i = static_cast<size_t>(optInTimePeriod) - 1; i < size; ++i) {
      double sum = 0.0;
      for (int j = 0; j < optInTimePeriod; ++j) {
        sum += in[i - static_cast<size_t>(optInTimePeriod - 1 - j)] *
               wts[static_cast<size_t>(j)];
      }
      out[i] = sum;
    }
  }
  return DoubleArrayOUT(out.data(), {size}, std::move(owner));
}

// ---------------------------------------------------------
// ELASTIC VOLUME WEIGHTED MOVING AVERAGE (EVWMA)
// ---------------------------------------------------------
DoubleArrayOUT evwma(DoubleArrayIN inReal, DoubleArrayIN inVolume,
                     int optInTimePeriod = 30) {
  if (inReal.size() == 0 || inVolume.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("EVWMA", inReal.shape(0), {inVolume.shape(0)});
  size_t size = inReal.shape(0);
  int lookback = optInTimePeriod - 1;
  auto [outData, owner] = alloc_output(size, lookback);
  gsl::span<double> out(outData.get(), size);
  if (size < static_cast<size_t>(optInTimePeriod)) {
    return DoubleArrayOUT(out.data(), {size}, std::move(owner));
  }

  gsl::span<const double> pr(inReal.data(), size);
  gsl::span<const double> vo(inVolume.data(), size);

  {
    nb::gil_scoped_release release;

    double volSum = 0.0;
    for (int i = 0; i < optInTimePeriod; i++) {
      volSum += vo[static_cast<size_t>(i)];
    }

    out[static_cast<size_t>(optInTimePeriod - 1)] =
        pr[static_cast<size_t>(optInTimePeriod - 1)];

    for (auto i = static_cast<size_t>(optInTimePeriod); i < size; i++) {
      volSum = (volSum + vo[i]) - vo[i - static_cast<size_t>(optInTimePeriod)];
      if (volSum > 0) {
        out[i] = (((volSum - vo[i]) * out[i - 1]) + (vo[i] * pr[i])) / volSum;
      } else {
        out[i] = out[i - 1];
      }
    }
  }
  return DoubleArrayOUT(out.data(), {size}, std::move(owner));
}

// ---------------------------------------------------------
// ZERO LAG EXPONENTIAL MOVING AVERAGE (ZLEMA)
// ---------------------------------------------------------
DoubleArrayOUT zlema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = optInTimePeriod - 1;
  auto [outData, owner] = alloc_output(size, lookback);
  gsl::span<double> out(outData.get(), size);
  if (size < static_cast<size_t>(optInTimePeriod)) {
    return DoubleArrayOUT(out.data(), {size}, std::move(owner));
  }

  gsl::span<const double> in(inReal.data(), size);

  {
    nb::gil_scoped_release release;

    double ratio = 2.0 / static_cast<double>(optInTimePeriod + 1);
    double seed = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(optInTimePeriod); i++) {
      seed += in[i] / static_cast<double>(optInTimePeriod);
    }

    out[static_cast<size_t>(optInTimePeriod - 1)] = seed;

    double lag = 1.0 / ratio;
    double wt = std::fmod(lag, 1.0);
    double w1 = 1.0 - wt;
    double r1 = 1.0 - ratio;

    for (auto i = static_cast<size_t>(optInTimePeriod); i < size; i++) {
      double loc_d = static_cast<double>(i) - lag;
      int loc = static_cast<int>(std::floor(loc_d));
      double value;
      if (loc >= 0 && (static_cast<size_t>(loc) + 1) < size) {
        value = (2.0 * in[i]) - ((w1 * in[static_cast<size_t>(loc)]) +
                                 (wt * in[static_cast<size_t>(loc) + 1]));
      } else {
        value = (2.0 * in[i]) - in[i > 0 ? i - 1 : 0];
      }
      out[i] = (ratio * value) + (r1 * out[i - 1]);
    }
  }
  return DoubleArrayOUT(out.data(), {size}, std::move(owner));
}