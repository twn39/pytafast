#include "common.h"
#include <cmath>
#include <vector>

// ---------------------------------------------------------
// SIMPLE MOVING AVERAGE (SMA)
// ---------------------------------------------------------
DoubleArrayOUT sma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_SMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_SMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_SMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// EXPONENTIAL MOVING AVERAGE (EMA)
// ---------------------------------------------------------
DoubleArrayOUT ema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_EMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_EMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_EMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
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
  size_t size = inReal.shape(0);
  int lookback = TA_BBANDS_Lookback(optInTimePeriod, optInNbDevUp, optInNbDevDn,
                         static_cast<TA_MAType>(optInMAType));
  auto [outUpper, ownerU] = alloc_output(size, lookback);
  auto [outMiddle, ownerM] = alloc_output(size, lookback);
  auto [outLower, ownerL] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_BBANDS(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
                        optInNbDevUp, optInNbDevDn,
                        static_cast<TA_MAType>(optInMAType), &outBegIdx,
                        &outNBElement, outUpper.get() + lookback,
                        outMiddle.get() + lookback, outLower.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_BBANDS");
  return nb::make_tuple(DoubleArrayOUT(outUpper.get(), {size}, ownerU),
                        DoubleArrayOUT(outMiddle.get(), {size}, ownerM),
                        DoubleArrayOUT(outLower.get(), {size}, ownerL));
}

// ---------------------------------------------------------
// DOUBLE EXPONENTIAL MOVING AVERAGE (DEMA)
// ---------------------------------------------------------
DoubleArrayOUT dema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_DEMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_DEMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_DEMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA)
// ---------------------------------------------------------
DoubleArrayOUT kama(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_KAMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_KAMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_KAMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
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
  size_t size = inReal.shape(0);
  int lookback = TA_MAMA_Lookback(optInFastLimit, optInSlowLimit);
  auto [outMAMA, ownerM] = alloc_output(size, lookback);
  auto [outFAMA, ownerF] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MAMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInFastLimit,
                      optInSlowLimit, &outBegIdx, &outNBElement,
                      outMAMA.get() + lookback, outFAMA.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MAMA");
  return nb::make_tuple(DoubleArrayOUT(outMAMA.get(), {size}, ownerM),
                        DoubleArrayOUT(outFAMA.get(), {size}, ownerF));
}

// ---------------------------------------------------------
// MOVING AVERAGE (MA)
// ---------------------------------------------------------
DoubleArrayOUT ma(DoubleArrayIN inReal, int optInTimePeriod = 30,
                  int optInMAType = 0) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback =
      TA_MA_Lookback(optInTimePeriod, static_cast<TA_MAType>(optInMAType));
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
                    static_cast<TA_MAType>(optInMAType), &outBegIdx,
                    &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// MIDPOINT OVER PERIOD (MIDPOINT)
// ---------------------------------------------------------
DoubleArrayOUT midpoint(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_MIDPOINT_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MIDPOINT(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                          &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MIDPOINT");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// PARABOLIC SAR (SAR)
// ---------------------------------------------------------
DoubleArrayOUT sar(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   double optInAcceleration = 0.02, double optInMaximum = 0.2) {
  if (inHigh.size() == 0 || inLow.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inHigh.shape(0);
  int lookback = TA_SAR_Lookback(optInAcceleration, optInMaximum);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_SAR(0, gsl::narrow<int>(size - 1), inHigh.data(), inLow.data(), optInAcceleration,
                     optInMaximum, &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_SAR");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// TRIPLE EXPONENTIAL MOVING AVERAGE (T3)
// ---------------------------------------------------------
DoubleArrayOUT t3(DoubleArrayIN inReal, int optInTimePeriod = 5,
                  double optInVFactor = 0.7) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_T3_Lookback(optInTimePeriod, optInVFactor);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_T3(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, optInVFactor,
                    &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_T3");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// TRIPLE EXPONENTIAL MOVING AVERAGE (TEMA)
// ---------------------------------------------------------
DoubleArrayOUT tema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_TEMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_TEMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_TEMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// TRIANGULAR MOVING AVERAGE (TRIMA)
// ---------------------------------------------------------
DoubleArrayOUT trima(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_TRIMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_TRIMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                       &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_TRIMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// WEIGHTED MOVING AVERAGE (WMA)
// ---------------------------------------------------------
DoubleArrayOUT wma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_WMA_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_WMA(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_WMA");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// ZIGZAG INDICATOR (ZIGZAG)
// ---------------------------------------------------------
DoubleArrayOUT zigzag(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                      double change = 10.0, bool percent = true) {
  if (inHigh.size() == 0 || inLow.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inHigh.shape(0);
  auto [outData, owner] = alloc_output(size, 0); 
  double *out = outData.get();
  std::fill(out, out + size, NaN);
  if (size < 2) return DoubleArrayOUT(out, {size}, owner);
  const double *hi = inHigh.data();
  const double *lo = inLow.data();
  int last_idx = 0;
  double last_val = (hi[0] + lo[0]) / 2.0;
  int trend = 0; 
  out[0] = last_val;
  for (size_t i = 1; i < size; ++i) {
    if (trend == 0) {
      double up_diff = percent ? (hi[i] / last_val - 1.0) * 100.0 : (hi[i] - last_val);
      double down_diff = percent ? (last_val / lo[i] - 1.0) * 100.0 : (last_val - lo[i]);
      if (up_diff >= change) { trend = 1; last_val = hi[i]; last_idx = (int)i; out[i] = last_val; }
      else if (down_diff >= change) { trend = -1; last_val = lo[i]; last_idx = (int)i; out[i] = last_val; }
    } else if (trend == 1) {
      if (hi[i] > last_val) { out[last_idx] = NaN; last_val = hi[i]; last_idx = (int)i; out[i] = last_val; }
      else if ((percent ? (last_val / lo[i] - 1.0) * 100.0 : (last_val - lo[i])) >= change) {
        trend = -1; last_val = lo[i]; last_idx = (int)i; out[i] = last_val;
      }
    } else if (trend == -1) {
      if (lo[i] < last_val) { out[last_idx] = NaN; last_val = lo[i]; last_idx = (int)i; out[i] = last_val; }
      else if ((percent ? (hi[i] / last_val - 1.0) * 100.0 : (hi[i] - last_val)) >= change) {
        trend = 1; last_val = hi[i]; last_idx = (int)i; out[i] = last_val;
      }
    }
  }
  int prev_idx = 0;
  for (size_t i = 1; i < size; ++i) {
    if (!std::isnan(out[i])) {
      double start_val = out[prev_idx];
      double end_val = out[i];
      double step = (end_val - start_val) / (double)(i - prev_idx);
      for (size_t j = prev_idx + 1; j < i; ++j) out[j] = start_val + step * (double)(j - prev_idx);
      prev_idx = (int)i;
    }
  }
  if (prev_idx < (int)size - 1) {
      double start_val = out[prev_idx];
      double end_val = (trend == 1) ? hi[size-1] : (trend == -1 ? lo[size-1] : (hi[size-1]+lo[size-1])/2.0);
      double step = (end_val - start_val) / (double)(size - 1 - prev_idx);
      for (size_t j = prev_idx + 1; j < size; ++j) out[j] = start_val + step * (double)(j - prev_idx);
  }
  return DoubleArrayOUT(out, {size}, owner);
}

// ---------------------------------------------------------
// ARNAUD LEGOUX MOVING AVERAGE (ALMA)
// ---------------------------------------------------------
DoubleArrayOUT alma(DoubleArrayIN inReal, int optInTimePeriod = 9,
                    double optInOffset = 0.85, double optInSigma = 6.0) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  auto [outData, owner] = alloc_output(size, 0);
  double *out = outData.get();
  std::fill(out, out + size, NaN);
  if (size < (size_t)optInTimePeriod) return DoubleArrayOUT(out, {size}, owner);
  std::vector<double> wts(optInTimePeriod);
  double m = std::floor(optInOffset * (optInTimePeriod - 1));
  double s = (double)optInTimePeriod / optInSigma;
  double sum_w = 0.0;
  for (int i = 0; i < optInTimePeriod; ++i) {
    wts[i] = std::exp(-std::pow(i - m, 2) / (2 * s * s));
    sum_w += wts[i];
  }
  if (sum_w != 0) { for (int i = 0; i < optInTimePeriod; ++i) wts[i] /= sum_w; }
  const double *in = inReal.data();
  for (size_t i = optInTimePeriod - 1; i < size; ++i) {
    double sum = 0.0;
    for (int j = 0; j < optInTimePeriod; ++j) sum += in[i - (optInTimePeriod - 1 - j)] * wts[j];
    out[i] = sum;
  }
  return DoubleArrayOUT(out, {size}, owner);
}

// ---------------------------------------------------------
// ELASTIC VOLUME WEIGHTED MOVING AVERAGE (EVWMA)
// ---------------------------------------------------------
DoubleArrayOUT evwma(DoubleArrayIN inReal, DoubleArrayIN inVolume, int optInTimePeriod = 30) {
  if (inReal.size() == 0 || inVolume.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  auto [outData, owner] = alloc_output(size, 0);
  double *out = outData.get();
  std::fill(out, out + size, NaN);
  if (size < (size_t)optInTimePeriod) return DoubleArrayOUT(out, {size}, owner);
  const double *pr = inReal.data();
  const double *vo = inVolume.data();
  double volSum = 0.0;
  for (int i = 0; i < optInTimePeriod; i++) volSum += vo[i];
  out[optInTimePeriod - 1] = pr[optInTimePeriod - 1];
  for (size_t i = optInTimePeriod; i < size; i++) {
    volSum = volSum + vo[i] - vo[i - optInTimePeriod];
    if (volSum > 0) out[i] = ((volSum - vo[i]) * out[i - 1] + vo[i] * pr[i]) / volSum;
    else out[i] = out[i - 1];
  }
  return DoubleArrayOUT(out, {size}, owner);
}

// ---------------------------------------------------------
// ZERO LAG EXPONENTIAL MOVING AVERAGE (ZLEMA)
// ---------------------------------------------------------
DoubleArrayOUT zlema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  auto [outData, owner] = alloc_output(size, 0);
  double *out = outData.get();
  std::fill(out, out + size, NaN);
  if (size < (size_t)optInTimePeriod) return DoubleArrayOUT(out, {size}, owner);
  const double *in = inReal.data();
  double ratio = 2.0 / (optInTimePeriod + 1);
  double seed = 0.0;
  for (int i = 0; i < optInTimePeriod; i++) seed += in[i] / optInTimePeriod;
  out[optInTimePeriod - 1] = seed;
  double lag = 1.0 / ratio;
  double wt = std::fmod(lag, 1.0);
  double w1 = 1.0 - wt;
  double r1 = 1.0 - ratio;
  for (size_t i = optInTimePeriod; i < size; i++) {
    double loc_d = (double)i - lag;
    int loc = (int)loc_d;
    double value;
    if (loc >= 0 && loc + 1 < (int)size) value = 2 * in[i] - (w1 * in[loc] + wt * in[loc + 1]);
    else value = 2 * in[i] - in[i > 0 ? i - 1 : 0];
    out[i] = ratio * value + r1 * out[i - 1];
  }
  return DoubleArrayOUT(out, {size}, owner);
}
