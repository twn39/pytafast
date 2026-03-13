#include "common.h"

// ---------------------------------------------------------
// RELATIVE STRENGTH INDEX (RSI)
// ---------------------------------------------------------
DoubleArrayOUT rsi(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_RSI_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_RSI: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_RSI(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
               &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_RSI");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// MOVING AVERAGE CONVERGENCE/DIVERGENCE (MACD)
// ---------------------------------------------------------
nb::tuple macd(DoubleArrayIN inReal, int optInFastPeriod = 12,
               int optInSlowPeriod = 26, int optInSignalPeriod = 9) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback =
      TA_MACD_Lookback(optInFastPeriod, optInSlowPeriod, optInSignalPeriod);
  auto [outMACD, ownerM] = alloc_output(size, lookback);
  auto [outSignal, ownerS] = alloc_output(size, lookback);
  auto [outHist, ownerH] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_MACD(0, gsl::narrow<int>(size - 1), inReal.data(),
                        optInFastPeriod, optInSlowPeriod, optInSignalPeriod,
                        &outBegIdx, &outNBElement, outMACD.get() + lookback,
                        outSignal.get() + lookback, outHist.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_MACD");
  }
  return nb::make_tuple(DoubleArrayOUT(outMACD.get(), {size}, ownerM),
                        DoubleArrayOUT(outSignal.get(), {size}, ownerS),
                        DoubleArrayOUT(outHist.get(), {size}, ownerH));
}

// ---------------------------------------------------------
// MACD WITH CONTROLLABLE MA TYPE (MACDEXT)
// ---------------------------------------------------------
nb::tuple macdext(DoubleArrayIN inReal, int optInFastPeriod = 12,
                  int optInFastMAType = 0, int optInSlowPeriod = 26,
                  int optInSlowMAType = 0, int optInSignalPeriod = 9,
                  int optInSignalMAType = 0) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MACDEXT_Lookback(
      optInFastPeriod, static_cast<TA_MAType>(optInFastMAType), optInSlowPeriod,
      static_cast<TA_MAType>(optInSlowMAType), optInSignalPeriod,
      static_cast<TA_MAType>(optInSignalMAType));
  if (lookback < 0) {
    throw std::invalid_argument("TA_MACDEXT: Invalid parameter (lookback < 0)");
  }
  auto [outMACD, ownerM] = alloc_output(size, lookback);
  auto [outSignal, ownerS] = alloc_output(size, lookback);
  auto [outHist, ownerH] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_MACDEXT(0, gsl::narrow<int>(size - 1), inReal.data(),
                   optInFastPeriod, static_cast<TA_MAType>(optInFastMAType),
                   optInSlowPeriod, static_cast<TA_MAType>(optInSlowMAType),
                   optInSignalPeriod, static_cast<TA_MAType>(optInSignalMAType),
                   &outBegIdx, &outNBElement, outMACD.get() + lookback,
                   outSignal.get() + lookback, outHist.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MACDEXT");
  return nb::make_tuple(DoubleArrayOUT(outMACD.get(), {size}, ownerM),
                        DoubleArrayOUT(outSignal.get(), {size}, ownerS),
                        DoubleArrayOUT(outHist.get(), {size}, ownerH));
}

// ---------------------------------------------------------
// MACD FIX (12, 26, 9)
// ---------------------------------------------------------
nb::tuple macdfix(DoubleArrayIN inReal, int optInSignalPeriod = 9) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MACDFIX_Lookback(optInSignalPeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_MACDFIX: Invalid parameter (lookback < 0)");
  }
  auto [outMACD, ownerM] = alloc_output(size, lookback);
  auto [outSignal, ownerS] = alloc_output(size, lookback);
  auto [outHist, ownerH] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_MACDFIX(0, gsl::narrow<int>(size - 1), inReal.data(),
                           optInSignalPeriod, &outBegIdx, &outNBElement,
                           outMACD.get() + lookback, outSignal.get() + lookback,
                           outHist.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_MACDFIX");
  }
  return nb::make_tuple(DoubleArrayOUT(outMACD.get(), {size}, ownerM),
                        DoubleArrayOUT(outSignal.get(), {size}, ownerS),
                        DoubleArrayOUT(outHist.get(), {size}, ownerH));
}

// ---------------------------------------------------------
// STOCHASTIC (STOCH)
// ---------------------------------------------------------
nb::tuple stoch(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                DoubleArrayIN inClose, int optInFastK_Period = 5,
                int optInSlowK_Period = 3, int optInSlowK_MAType = 0,
                int optInSlowD_Period = 3, int optInSlowD_MAType = 0) {
  if (inHigh.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  check_lengths("STOCH", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_STOCH_Lookback(optInFastK_Period, optInSlowK_Period,
                                   static_cast<TA_MAType>(optInSlowK_MAType),
                                   optInSlowD_Period,
                                   static_cast<TA_MAType>(optInSlowD_MAType));
  if (lookback < 0) {
    throw std::invalid_argument("TA_STOCH: Invalid parameter (lookback < 0)");
  }
  auto [outSlowK, ownerK] = alloc_output(size, lookback);
  auto [outSlowD, ownerD] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_STOCH(
          0, gsl::narrow<int>(size - 1), inHigh.data(), inLow.data(),
          inClose.data(), optInFastK_Period, optInSlowK_Period,
          static_cast<TA_MAType>(optInSlowK_MAType), optInSlowD_Period,
          static_cast<TA_MAType>(optInSlowD_MAType), &outBegIdx, &outNBElement,
          outSlowK.get() + lookback, outSlowD.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_STOCH");
  }
  return nb::make_tuple(DoubleArrayOUT(outSlowK.get(), {size}, ownerK),
                        DoubleArrayOUT(outSlowD.get(), {size}, ownerD));
}

// ---------------------------------------------------------
// STOCHASTIC FAST (STOCHF)
// ---------------------------------------------------------
nb::tuple stochf(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                 DoubleArrayIN inClose, int optInFastK_Period = 5,
                 int optInFastD_Period = 3, int optInFastD_MAType = 0) {
  if (inHigh.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  check_lengths("STOCHF", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_STOCHF_Lookback(optInFastK_Period, optInFastD_Period,
                                    static_cast<TA_MAType>(optInFastD_MAType));
  if (lookback < 0) {
    throw std::invalid_argument("TA_STOCHF: Invalid parameter (lookback < 0)");
  }
  auto [outFastK, ownerK] = alloc_output(size, lookback);
  auto [outFastD, ownerD] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_STOCHF(
          0, gsl::narrow<int>(size - 1), inHigh.data(), inLow.data(),
          inClose.data(), optInFastK_Period, optInFastD_Period,
          static_cast<TA_MAType>(optInFastD_MAType), &outBegIdx, &outNBElement,
          outFastK.get() + lookback, outFastD.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_STOCHF");
  }
  return nb::make_tuple(DoubleArrayOUT(outFastK.get(), {size}, ownerK),
                        DoubleArrayOUT(outFastD.get(), {size}, ownerD));
}

// ---------------------------------------------------------
// STOCHASTIC RSI (STOCHRSI)
// ---------------------------------------------------------
nb::tuple stochrsi(DoubleArrayIN inReal, int optInTimePeriod = 14,
                   int optInFastK_Period = 5, int optInFastD_Period = 3,
                   int optInFastD_MAType = 0) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_STOCHRSI_Lookback(
      optInTimePeriod, optInFastK_Period, optInFastD_Period,
      static_cast<TA_MAType>(optInFastD_MAType));
  if (lookback < 0) {
    throw std::invalid_argument(
        "TA_STOCHRSI: Invalid parameter (lookback < 0)");
  }
  auto [outFastK, ownerK] = alloc_output(size, lookback);
  auto [outFastD, ownerD] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_STOCHRSI(
          0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
          optInFastK_Period, optInFastD_Period,
          static_cast<TA_MAType>(optInFastD_MAType), &outBegIdx, &outNBElement,
          outFastK.get() + lookback, outFastD.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_STOCHRSI");
  }
  return nb::make_tuple(DoubleArrayOUT(outFastK.get(), {size}, ownerK),
                        DoubleArrayOUT(outFastD.get(), {size}, ownerD));
}

// ---------------------------------------------------------
// RATE OF CHANGE (ROC)
// ---------------------------------------------------------
DoubleArrayOUT roc(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_ROC_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_ROC: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ROC(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
               &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_ROC");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// RATE OF CHANGE PERCENTAGE (ROCP)
// ---------------------------------------------------------
DoubleArrayOUT rocp(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_ROCP_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_ROCP: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ROCP(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
                &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_ROCP");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// RATE OF CHANGE RATIO (ROCR)
// ---------------------------------------------------------
DoubleArrayOUT rocr(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_ROCR_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_ROCR: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ROCR(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
                &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_ROCR");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// RATE OF CHANGE RATIO 100 (ROCR100)
// ---------------------------------------------------------
DoubleArrayOUT rocr100(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_ROCR100_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_ROCR100: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_ROCR100(0, gsl::narrow<int>(size - 1), inReal.data(),
                           optInTimePeriod, &outBegIdx, &outNBElement,
                           outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_ROCR100");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// MOMENTUM (MOM)
// ---------------------------------------------------------
DoubleArrayOUT mom(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MOM_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_MOM: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_MOM(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
               &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_MOM");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// CHANDE MOMENTUM OSCILLATOR (CMO)
// ---------------------------------------------------------
DoubleArrayOUT cmo(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_CMO_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_CMO: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_CMO(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
               &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_CMO");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// ABSOLUTE PRICE OSCILLATOR (APO)
// ---------------------------------------------------------
DoubleArrayOUT apo(DoubleArrayIN inReal, int optInFastPeriod = 12,
                   int optInSlowPeriod = 26, int optInMAType = 0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_APO_Lookback(optInFastPeriod, optInSlowPeriod,
                                 static_cast<TA_MAType>(optInMAType));
  if (lookback < 0) {
    throw std::invalid_argument("TA_APO: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_APO(0, gsl::narrow<int>(size - 1), inReal.data(), optInFastPeriod,
               optInSlowPeriod, static_cast<TA_MAType>(optInMAType), &outBegIdx,
               &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_APO");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// PERCENTAGE PRICE OSCILLATOR (PPO)
// ---------------------------------------------------------
DoubleArrayOUT ppo(DoubleArrayIN inReal, int optInFastPeriod = 12,
                   int optInSlowPeriod = 26, int optInMAType = 0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_PPO_Lookback(optInFastPeriod, optInSlowPeriod,
                                 static_cast<TA_MAType>(optInMAType));
  if (lookback < 0) {
    throw std::invalid_argument("TA_PPO: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_PPO(0, gsl::narrow<int>(size - 1), inReal.data(), optInFastPeriod,
               optInSlowPeriod, static_cast<TA_MAType>(optInMAType), &outBegIdx,
               &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_PPO");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// TRIPLE SMOOTHED EXPONENTIAL MOVING AVERAGE (TRIX)
// ---------------------------------------------------------
DoubleArrayOUT trix(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  size_t size = inReal.shape(0);
  int lookback = TA_TRIX_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_TRIX: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_TRIX(0, gsl::narrow<int>(size - 1), inReal.data(), optInTimePeriod,
                &outBegIdx, &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_TRIX");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// AROON (AROON)
// ---------------------------------------------------------
nb::tuple aroon(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  check_lengths("AROON", inHigh.shape(0), {inLow.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_AROON_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_AROON: Invalid parameter (lookback < 0)");
  }
  auto [outDown, ownerD] = alloc_output(size, lookback);
  auto [outUp, ownerU] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode =
          TA_AROON(0, gsl::narrow<int>(size - 1), inHigh.data(), inLow.data(),
                   optInTimePeriod, &outBegIdx, &outNBElement,
                   outDown.get() + lookback, outUp.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_AROON");
  }
  return nb::make_tuple(DoubleArrayOUT(outDown.get(), {size}, ownerD),
                        DoubleArrayOUT(outUp.get(), {size}, ownerU));
}

// ---------------------------------------------------------
// AROON OSCILLATOR (AROONOSC)
// ---------------------------------------------------------
DoubleArrayOUT aroonosc(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("AROONOSC", inHigh.shape(0), {inLow.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_AROONOSC_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument(
        "TA_AROONOSC: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_AROONOSC(0, gsl::narrow<int>(size - 1), inHigh.data(),
                            inLow.data(), optInTimePeriod, &outBegIdx,
                            &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_AROONOSC");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// AVERAGE DIRECTIONAL MOVEMENT INDEX (ADX)
// ---------------------------------------------------------
DoubleArrayOUT adx(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("ADX", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_ADX_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_ADX: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_ADX(0, gsl::narrow<int>(size - 1), inHigh.data(),
                       inLow.data(), inClose.data(), optInTimePeriod,
                       &outBegIdx, &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_ADX");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// AVERAGE DIRECTIONAL MOVEMENT INDEX RATING (ADXR)
// ---------------------------------------------------------
DoubleArrayOUT adxr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                    DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("ADXR", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_ADXR_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_ADXR: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_ADXR(0, gsl::narrow<int>(size - 1), inHigh.data(),
                        inLow.data(), inClose.data(), optInTimePeriod,
                        &outBegIdx, &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_ADXR");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// DIRECTIONAL MOVEMENT INDEX (DX)
// ---------------------------------------------------------
DoubleArrayOUT dx(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                  DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("DX", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_DX_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_DX: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_DX(0, gsl::narrow<int>(size - 1), inHigh.data(),
                      inLow.data(), inClose.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_DX");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// MINUS DIRECTIONAL INDICATOR (MINUS_DI)
// ---------------------------------------------------------
DoubleArrayOUT minus_di(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("MINUS_DI", inHigh.shape(0),
                {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_MINUS_DI_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument(
        "TA_MINUS_DI: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode =
          TA_MINUS_DI(0, gsl::narrow<int>(size - 1), inHigh.data(),
                      inLow.data(), inClose.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_MINUS_DI");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// MINUS DIRECTIONAL MOVEMENT (MINUS_DM)
// ---------------------------------------------------------
DoubleArrayOUT minus_dm(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("MINUS_DM", inHigh.shape(0), {inLow.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_MINUS_DM_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument(
        "TA_MINUS_DM: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_MINUS_DM(0, gsl::narrow<int>(size - 1), inHigh.data(),
                            inLow.data(), optInTimePeriod, &outBegIdx,
                            &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_MINUS_DM");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// PLUS DIRECTIONAL INDICATOR (PLUS_DI)
// ---------------------------------------------------------
DoubleArrayOUT plus_di(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                       DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("PLUS_DI", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_PLUS_DI_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_PLUS_DI: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_PLUS_DI(0, gsl::narrow<int>(size - 1), inHigh.data(),
                           inLow.data(), inClose.data(), optInTimePeriod,
                           &outBegIdx, &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_PLUS_DI");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// PLUS DIRECTIONAL MOVEMENT (PLUS_DM)
// ---------------------------------------------------------
DoubleArrayOUT plus_dm(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                       int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("PLUS_DM", inHigh.shape(0), {inLow.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_PLUS_DM_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_PLUS_DM: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_PLUS_DM(0, gsl::narrow<int>(size - 1), inHigh.data(),
                           inLow.data(), optInTimePeriod, &outBegIdx,
                           &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_PLUS_DM");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// WILLIAMS' %R (WILLR)
// ---------------------------------------------------------
DoubleArrayOUT willr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                     DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("WILLR", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_WILLR_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_WILLR: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_WILLR(0, gsl::narrow<int>(size - 1), inHigh.data(),
                         inLow.data(), inClose.data(), optInTimePeriod,
                         &outBegIdx, &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_WILLR");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// MONEY FLOW INDEX (MFI)
// ---------------------------------------------------------
DoubleArrayOUT mfi(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, DoubleArrayIN inVolume,
                   int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("MFI", inHigh.shape(0),
                {inLow.shape(0), inClose.shape(0), inVolume.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_MFI_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_MFI: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode =
          TA_MFI(0, gsl::narrow<int>(size - 1), inHigh.data(), inLow.data(),
                 inClose.data(), inVolume.data(), optInTimePeriod, &outBegIdx,
                 &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_MFI");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// COMMODITY CHANNEL INDEX (CCI)
// ---------------------------------------------------------
DoubleArrayOUT cci(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("CCI", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback = TA_CCI_Lookback(optInTimePeriod);
  if (lookback < 0) {
    throw std::invalid_argument("TA_CCI: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_CCI(0, gsl::narrow<int>(size - 1), inHigh.data(),
                       inLow.data(), inClose.data(), optInTimePeriod,
                       &outBegIdx, &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_CCI");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// ULTIMATE OSCILLATOR (ULTOSC)
// ---------------------------------------------------------
DoubleArrayOUT ultosc(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                      DoubleArrayIN inClose, int optInTimePeriod1 = 7,
                      int optInTimePeriod2 = 14, int optInTimePeriod3 = 28) {
  if (inHigh.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("ULTOSC", inHigh.shape(0), {inLow.shape(0), inClose.shape(0)});
  size_t size = inHigh.shape(0);
  int lookback =
      TA_ULTOSC_Lookback(optInTimePeriod1, optInTimePeriod2, optInTimePeriod3);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_ULTOSC(0, gsl::narrow<int>(size - 1), inHigh.data(),
                        inLow.data(), inClose.data(), optInTimePeriod1,
                        optInTimePeriod2, optInTimePeriod3, &outBegIdx,
                        &outNBElement, outData.get() + lookback);
  }
  check_ta_retcode(retCode, "TA_ULTOSC");
  return DoubleArrayOUT(outData.get(), {size}, owner);
}

// ---------------------------------------------------------
// BALANCE OF POWER (BOP)
// ---------------------------------------------------------
DoubleArrayOUT bop(DoubleArrayIN inOpen, DoubleArrayIN inHigh,
                   DoubleArrayIN inLow, DoubleArrayIN inClose) {
  if (inOpen.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  check_lengths("BOP", inOpen.shape(0),
                {inHigh.shape(0), inLow.shape(0), inClose.shape(0)});
  size_t size = inOpen.shape(0);
  int lookback = TA_BOP_Lookback();
  if (lookback < 0) {
    throw std::invalid_argument("TA_BOP: Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = TA_BOP(0, gsl::narrow<int>(size - 1), inOpen.data(),
                       inHigh.data(), inLow.data(), inClose.data(), &outBegIdx,
                       &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, "TA_BOP");
  }
  return DoubleArrayOUT(outData.get(), {size}, owner);
}
