// Momentum Indicators: RSI, MACD, MACDEXT, MACDFIX, ROC, ROCP, ROCR, ROCR100,
// STOCH, STOCHF, STOCHRSI, MOM, CMO, APO, PPO, TRIX, AROON, AROONOSC,
// ADX, ADXR, DX, MINUS_DI, MINUS_DM, PLUS_DI, PLUS_DM, WILLR, MFI,
// CCI, ULTOSC, BOP
#include "common.h"

// ---------------------------------------------------------
// RELATIVE STRENGTH INDEX
// ---------------------------------------------------------
DoubleArrayOUT rsi(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());

  size_t size = inReal.shape(0);
  int lookback = TA_RSI_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_RSI(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_RSI");

  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MACD
// ---------------------------------------------------------
nb::tuple macd(DoubleArrayIN inReal, int optInFastPeriod = 12,
               int optInSlowPeriod = 26, int optInSignalPeriod = 9) {
  if (inReal.size() == 0) {
    return nb::make_tuple(DoubleArrayOUT(nullptr, {0}, nb::handle()),
                          DoubleArrayOUT(nullptr, {0}, nb::handle()),
                          DoubleArrayOUT(nullptr, {0}, nb::handle()));
  }

  size_t size = inReal.shape(0);
  int lookback =
      TA_MACD_Lookback(optInFastPeriod, optInSlowPeriod, optInSignalPeriod);

  auto [outMACD, owner1] = alloc_output(size, lookback);
  auto [outSignal, owner2] = alloc_output(size, lookback);
  auto [outHist, owner3] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_MACD(0, size - 1, inReal.data(), optInFastPeriod, optInSlowPeriod,
                optInSignalPeriod, &outBegIdx, &outNBElement,
                outMACD + lookback, outSignal + lookback, outHist + lookback);
  }
  check_ta_retcode(retCode, "TA_MACD");

  return nb::make_tuple(DoubleArrayOUT(outMACD, {size}, owner1),
                        DoubleArrayOUT(outSignal, {size}, owner2),
                        DoubleArrayOUT(outHist, {size}, owner3));
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
  int lookback =
      TA_MACDEXT_Lookback(optInFastPeriod, (TA_MAType)optInFastMAType,
                          optInSlowPeriod, (TA_MAType)optInSlowMAType,
                          optInSignalPeriod, (TA_MAType)optInSignalMAType);
  auto [outMACD, ownerM] = alloc_output(size, lookback);
  auto [outSignal, ownerS] = alloc_output(size, lookback);
  auto [outHist, ownerH] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MACDEXT(
        0, size - 1, inReal.data(), optInFastPeriod, (TA_MAType)optInFastMAType,
        optInSlowPeriod, (TA_MAType)optInSlowMAType, optInSignalPeriod,
        (TA_MAType)optInSignalMAType, &outBegIdx, &outNBElement,
        outMACD + lookback, outSignal + lookback, outHist + lookback);
  }
  check_ta_retcode(retCode, "TA_MACDEXT");
  return nb::make_tuple(DoubleArrayOUT(outMACD, {size}, ownerM),
                        DoubleArrayOUT(outSignal, {size}, ownerS),
                        DoubleArrayOUT(outHist, {size}, ownerH));
}

// ---------------------------------------------------------
// MACD FIX 12/26 (MACDFIX)
// ---------------------------------------------------------
nb::tuple macdfix(DoubleArrayIN inReal, int optInSignalPeriod = 9) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty, empty);
  }
  size_t size = inReal.shape(0);
  int lookback = TA_MACDFIX_Lookback(optInSignalPeriod);
  auto [outMACD, ownerM] = alloc_output(size, lookback);
  auto [outSignal, ownerS] = alloc_output(size, lookback);
  auto [outHist, ownerH] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MACDFIX(0, size - 1, inReal.data(), optInSignalPeriod,
                         &outBegIdx, &outNBElement, outMACD + lookback,
                         outSignal + lookback, outHist + lookback);
  }
  check_ta_retcode(retCode, "TA_MACDFIX");
  return nb::make_tuple(DoubleArrayOUT(outMACD, {size}, ownerM),
                        DoubleArrayOUT(outSignal, {size}, ownerS),
                        DoubleArrayOUT(outHist, {size}, ownerH));
}

// ---------------------------------------------------------
// RATE OF CHANGE (ROC)
// ---------------------------------------------------------
DoubleArrayOUT roc(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_ROC_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_ROC(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ROC");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// RATE OF CHANGE PERCENTAGE (ROCP)
// ---------------------------------------------------------
DoubleArrayOUT rocp(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_ROCP_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_ROCP(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ROCP");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// RATE OF CHANGE RATIO (ROCR)
// ---------------------------------------------------------
DoubleArrayOUT rocr(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_ROCR_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_ROCR(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ROCR");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// RATE OF CHANGE RATIO 100 SCALE (ROCR100)
// ---------------------------------------------------------
DoubleArrayOUT rocr100(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_ROCR100_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_ROCR100(0, size - 1, inReal.data(), optInTimePeriod,
                         &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ROCR100");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// STOCHASTIC (STOCH)
// ---------------------------------------------------------
nb::tuple stoch(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                DoubleArrayIN inClose, int optInFastK_Period = 5,
                int optInSlowK_Period = 3, int optInSlowK_MAType = 0,
                int optInSlowD_Period = 3, int optInSlowD_MAType = 0) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0) {
    return nb::make_tuple(DoubleArrayOUT(nullptr, {0}, nb::handle()),
                          DoubleArrayOUT(nullptr, {0}, nb::handle()));
  }
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");

  size_t size = inHigh.shape(0);
  int lookback = TA_STOCH_Lookback(
      optInFastK_Period, optInSlowK_Period, (TA_MAType)optInSlowK_MAType,
      optInSlowD_Period, (TA_MAType)optInSlowD_MAType);

  auto [outSlowK, owner1] = alloc_output(size, lookback);
  auto [outSlowD, owner2] = alloc_output(size, lookback);

  int outBegIdx = 0;
  int outNBElement = 0;

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_STOCH(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                       optInFastK_Period, optInSlowK_Period,
                       (TA_MAType)optInSlowK_MAType, optInSlowD_Period,
                       (TA_MAType)optInSlowD_MAType, &outBegIdx, &outNBElement,
                       outSlowK + lookback, outSlowD + lookback);
  }
  check_ta_retcode(retCode, "TA_STOCH");

  return nb::make_tuple(DoubleArrayOUT(outSlowK, {size}, owner1),
                        DoubleArrayOUT(outSlowD, {size}, owner2));
}

// ---------------------------------------------------------
// STOCHASTIC FAST (STOCHF)
// ---------------------------------------------------------
nb::tuple stochf(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                 DoubleArrayIN inClose, int optInFastK_Period = 5,
                 int optInFastD_Period = 3, int optInFastD_MAType = 0) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty);
  }
  size_t size = inHigh.shape(0);
  int lookback = TA_STOCHF_Lookback(optInFastK_Period, optInFastD_Period,
                                    (TA_MAType)optInFastD_MAType);
  auto [outFastK, ownerK] = alloc_output(size, lookback);
  auto [outFastD, ownerD] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_STOCHF(0, size - 1, inHigh.data(), inLow.data(),
                        inClose.data(), optInFastK_Period, optInFastD_Period,
                        (TA_MAType)optInFastD_MAType, &outBegIdx, &outNBElement,
                        outFastK + lookback, outFastD + lookback);
  }
  check_ta_retcode(retCode, "TA_STOCHF");
  return nb::make_tuple(DoubleArrayOUT(outFastK, {size}, ownerK),
                        DoubleArrayOUT(outFastD, {size}, ownerD));
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
  int lookback =
      TA_STOCHRSI_Lookback(optInTimePeriod, optInFastK_Period,
                           optInFastD_Period, (TA_MAType)optInFastD_MAType);
  auto [outFastK, ownerK] = alloc_output(size, lookback);
  auto [outFastD, ownerD] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_STOCHRSI(
        0, size - 1, inReal.data(), optInTimePeriod, optInFastK_Period,
        optInFastD_Period, (TA_MAType)optInFastD_MAType, &outBegIdx,
        &outNBElement, outFastK + lookback, outFastD + lookback);
  }
  check_ta_retcode(retCode, "TA_STOCHRSI");
  return nb::make_tuple(DoubleArrayOUT(outFastK, {size}, ownerK),
                        DoubleArrayOUT(outFastD, {size}, ownerD));
}

// ---------------------------------------------------------
// MOMENTUM (MOM)
// ---------------------------------------------------------
DoubleArrayOUT mom(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_MOM_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MOM(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MOM");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// CHANDE MOMENTUM OSCILLATOR (CMO)
// ---------------------------------------------------------
DoubleArrayOUT cmo(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_CMO_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_CMO(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_CMO");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// ABSOLUTE PRICE OSCILLATOR (APO)
// ---------------------------------------------------------
DoubleArrayOUT apo(DoubleArrayIN inReal, int optInFastPeriod = 12,
                   int optInSlowPeriod = 26, int optInMAType = 0) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback =
      TA_APO_Lookback(optInFastPeriod, optInSlowPeriod, (TA_MAType)optInMAType);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_APO(0, size - 1, inReal.data(), optInFastPeriod,
                     optInSlowPeriod, (TA_MAType)optInMAType, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_APO");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// PERCENTAGE PRICE OSCILLATOR (PPO)
// ---------------------------------------------------------
DoubleArrayOUT ppo(DoubleArrayIN inReal, int optInFastPeriod = 12,
                   int optInSlowPeriod = 26, int optInMAType = 0) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback =
      TA_PPO_Lookback(optInFastPeriod, optInSlowPeriod, (TA_MAType)optInMAType);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_PPO(0, size - 1, inReal.data(), optInFastPeriod,
                     optInSlowPeriod, (TA_MAType)optInMAType, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_PPO");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// 1-DAY RATE-OF-CHANGE (TRIX)
// ---------------------------------------------------------
DoubleArrayOUT trix(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inReal.shape(0);
  int lookback = TA_TRIX_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_TRIX(0, size - 1, inReal.data(), optInTimePeriod, &outBegIdx,
                      &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_TRIX");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// AROON (AROON)
// ---------------------------------------------------------
nb::tuple aroon(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0) {
    return nb::make_tuple(DoubleArrayOUT(nullptr, {0}, nb::handle()),
                          DoubleArrayOUT(nullptr, {0}, nb::handle()));
  }
  if (inHigh.shape(0) != inLow.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_AROON_Lookback(optInTimePeriod);
  auto [outDown, owner1] = alloc_output(size, lookback);
  auto [outUp, owner2] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_AROON(0, size - 1, inHigh.data(), inLow.data(),
                       optInTimePeriod, &outBegIdx, &outNBElement,
                       outDown + lookback, outUp + lookback);
  }
  check_ta_retcode(retCode, "TA_AROON");
  return nb::make_tuple(DoubleArrayOUT(outDown, {size}, owner1),
                        DoubleArrayOUT(outUp, {size}, owner2));
}

// ---------------------------------------------------------
// AROON OSCILLATOR (AROONOSC)
// ---------------------------------------------------------
DoubleArrayOUT aroonosc(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_AROONOSC_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_AROONOSC(0, size - 1, inHigh.data(), inLow.data(), optInTimePeriod,
                    &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_AROONOSC");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// AVERAGE DIRECTIONAL MOVEMENT INDEX (ADX)
// ---------------------------------------------------------
DoubleArrayOUT adx(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_ADX_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ADX(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
               optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ADX");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// AVERAGE DIRECTIONAL MOVEMENT INDEX RATING (ADXR)
// ---------------------------------------------------------
DoubleArrayOUT adxr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                    DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inHigh.shape(0);
  int lookback = TA_ADXR_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ADXR(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ADXR");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// DIRECTIONAL MOVEMENT INDEX (DX)
// ---------------------------------------------------------
DoubleArrayOUT dx(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                  DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_DX_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_DX(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
              optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_DX");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MINUS DIRECTIONAL INDICATOR (MINUS_DI)
// ---------------------------------------------------------
DoubleArrayOUT minus_di(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_MINUS_DI_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MINUS_DI(0, size - 1, inHigh.data(), inLow.data(),
                          inClose.data(), optInTimePeriod, &outBegIdx,
                          &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MINUS_DI");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MINUS DIRECTIONAL MOVEMENT (MINUS_DM)
// ---------------------------------------------------------
DoubleArrayOUT minus_dm(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                        int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_MINUS_DM_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_MINUS_DM(0, size - 1, inHigh.data(), inLow.data(), optInTimePeriod,
                    &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MINUS_DM");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// PLUS DIRECTIONAL INDICATOR (PLUS_DI)
// ---------------------------------------------------------
DoubleArrayOUT plus_di(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                       DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_PLUS_DI_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_PLUS_DI(0, size - 1, inHigh.data(), inLow.data(),
                         inClose.data(), optInTimePeriod, &outBegIdx,
                         &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_PLUS_DI");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// PLUS DIRECTIONAL MOVEMENT (PLUS_DM)
// ---------------------------------------------------------
DoubleArrayOUT plus_dm(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                       int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_PLUS_DM_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_PLUS_DM(0, size - 1, inHigh.data(), inLow.data(), optInTimePeriod,
                   &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_PLUS_DM");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// WILLIAMS %R (WILLR)
// ---------------------------------------------------------
DoubleArrayOUT willr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                     DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_WILLR_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_WILLR(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                       optInTimePeriod, &outBegIdx, &outNBElement,
                       outData + lookback);
  }
  check_ta_retcode(retCode, "TA_WILLR");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// MONEY FLOW INDEX (MFI)
// ---------------------------------------------------------
DoubleArrayOUT mfi(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, DoubleArrayIN inVolume,
                   int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0 ||
      inVolume.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) ||
      inHigh.shape(0) != inClose.shape(0) ||
      inHigh.shape(0) != inVolume.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_MFI_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_MFI(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                     inVolume.data(), optInTimePeriod, &outBegIdx,
                     &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_MFI");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// COMMODITY CHANNEL INDEX (CCI)
// ---------------------------------------------------------
DoubleArrayOUT cci(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback = TA_CCI_Lookback(optInTimePeriod);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_CCI(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
               optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_CCI");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// ULTIMATE OSCILLATOR (ULTOSC)
// ---------------------------------------------------------
DoubleArrayOUT ultosc(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                      DoubleArrayIN inClose, int optInTimePeriod1 = 7,
                      int optInTimePeriod2 = 14, int optInTimePeriod3 = 28) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  size_t size = inHigh.shape(0);
  int lookback =
      TA_ULTOSC_Lookback(optInTimePeriod1, optInTimePeriod2, optInTimePeriod3);
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0;
  int outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ULTOSC(0, size - 1, inHigh.data(), inLow.data(), inClose.data(),
                  optInTimePeriod1, optInTimePeriod2, optInTimePeriod3,
                  &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ULTOSC");
  return DoubleArrayOUT(outData, {size}, owner);
}

// ---------------------------------------------------------
// BALANCE OF POWER (BOP)
// ---------------------------------------------------------
DoubleArrayOUT bop(DoubleArrayIN inOpen, DoubleArrayIN inHigh,
                   DoubleArrayIN inLow, DoubleArrayIN inClose) {
  if (inOpen.size() == 0 || inHigh.size() == 0 || inLow.size() == 0 ||
      inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  size_t size = inOpen.shape(0);
  int lookback = TA_BOP_Lookback();
  auto [outData, owner] = alloc_output(size, lookback);
  int outBegIdx = 0, outNBElement = 0;
  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_BOP(0, size - 1, inOpen.data(), inHigh.data(), inLow.data(),
               inClose.data(), &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_BOP");
  return DoubleArrayOUT(outData, {size}, owner);
}
