#include "common.h"

// ---------------------------------------------------------
// RELATIVE STRENGTH INDEX (RSI)
// ---------------------------------------------------------
DoubleArrayOUT rsi(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0), TA_RSI_Lookback(optInTimePeriod), "TA_RSI",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_RSI(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInTimePeriod, outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func_3out(
      inReal.shape(0),
      TA_MACD_Lookback(optInFastPeriod, optInSlowPeriod, optInSignalPeriod),
      "TA_MACD",
      [&](int *outBegIdx, int *outNBElement, double *out1, double *out2,
          double *out3) {
        return TA_MACD(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                       optInFastPeriod, optInSlowPeriod, optInSignalPeriod,
                       outBegIdx, outNBElement, out1, out2, out3);
      });
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
  return apply_ta_func_3out(
      inReal.shape(0),
      TA_MACDEXT_Lookback(
          optInFastPeriod, static_cast<TA_MAType>(optInFastMAType),
          optInSlowPeriod, static_cast<TA_MAType>(optInSlowMAType),
          optInSignalPeriod, static_cast<TA_MAType>(optInSignalMAType)),
      "TA_MACDEXT",
      [&](int *outBegIdx, int *outNBElement, double *out1, double *out2,
          double *out3) {
        return TA_MACDEXT(
            0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
            optInFastPeriod, static_cast<TA_MAType>(optInFastMAType),
            optInSlowPeriod, static_cast<TA_MAType>(optInSlowMAType),
            optInSignalPeriod, static_cast<TA_MAType>(optInSignalMAType),
            outBegIdx, outNBElement, out1, out2, out3);
      });
}

// ---------------------------------------------------------
// MACD FIX (12, 26, 9)
// ---------------------------------------------------------
nb::tuple macdfix(DoubleArrayIN inReal, int optInSignalPeriod = 9) {
  if (inReal.size() == 0) {
    auto empty = DoubleArrayOUT(nullptr, {0}, nb::handle());
    return nb::make_tuple(empty, empty, empty);
  }
  return apply_ta_func_3out(
      inReal.shape(0), TA_MACDFIX_Lookback(optInSignalPeriod), "TA_MACDFIX",
      [&](int *outBegIdx, int *outNBElement, double *out1, double *out2,
          double *out3) {
        return TA_MACDFIX(0, gsl::narrow<int>(inReal.shape(0) - 1),
                          inReal.data(), optInSignalPeriod, outBegIdx,
                          outNBElement, out1, out2, out3);
      });
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
  return apply_ta_func_2out(
      inHigh.shape(0),
      TA_STOCH_Lookback(optInFastK_Period, optInSlowK_Period,
                        static_cast<TA_MAType>(optInSlowK_MAType),
                        optInSlowD_Period,
                        static_cast<TA_MAType>(optInSlowD_MAType)),
      "TA_STOCH",
      [&](int *outBegIdx, int *outNBElement, double *out1, double *out2) {
        return TA_STOCH(
            0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
            inLow.data(), inClose.data(), optInFastK_Period, optInSlowK_Period,
            static_cast<TA_MAType>(optInSlowK_MAType), optInSlowD_Period,
            static_cast<TA_MAType>(optInSlowD_MAType), outBegIdx, outNBElement,
            out1, out2);
      });
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
  return apply_ta_func_2out(
      inHigh.shape(0),
      TA_STOCHF_Lookback(optInFastK_Period, optInFastD_Period,
                         static_cast<TA_MAType>(optInFastD_MAType)),
      "TA_STOCHF",
      [&](int *outBegIdx, int *outNBElement, double *out1, double *out2) {
        return TA_STOCHF(0, gsl::narrow<int>(inHigh.shape(0) - 1),
                         inHigh.data(), inLow.data(), inClose.data(),
                         optInFastK_Period, optInFastD_Period,
                         static_cast<TA_MAType>(optInFastD_MAType), outBegIdx,
                         outNBElement, out1, out2);
      });
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
  return apply_ta_func_2out(
      inReal.shape(0),
      TA_STOCHRSI_Lookback(optInTimePeriod, optInFastK_Period,
                           optInFastD_Period,
                           static_cast<TA_MAType>(optInFastD_MAType)),
      "TA_STOCHRSI",
      [&](int *outBegIdx, int *outNBElement, double *out1, double *out2) {
        return TA_STOCHRSI(0, gsl::narrow<int>(inReal.shape(0) - 1),
                           inReal.data(), optInTimePeriod, optInFastK_Period,
                           optInFastD_Period,
                           static_cast<TA_MAType>(optInFastD_MAType), outBegIdx,
                           outNBElement, out1, out2);
      });
}

// ---------------------------------------------------------
// RATE OF CHANGE (ROC)
// ---------------------------------------------------------
DoubleArrayOUT roc(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0), TA_ROC_Lookback(optInTimePeriod), "TA_ROC",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_ROC(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// RATE OF CHANGE PERCENTAGE (ROCP)
// ---------------------------------------------------------
DoubleArrayOUT rocp(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0), TA_ROCP_Lookback(optInTimePeriod), "TA_ROCP",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_ROCP(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                       optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// RATE OF CHANGE RATIO (ROCR)
// ---------------------------------------------------------
DoubleArrayOUT rocr(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0), TA_ROCR_Lookback(optInTimePeriod), "TA_ROCR",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_ROCR(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                       optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// RATE OF CHANGE RATIO 100 (ROCR100)
// ---------------------------------------------------------
DoubleArrayOUT rocr100(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0), TA_ROCR100_Lookback(optInTimePeriod), "TA_ROCR100",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_ROCR100(0, gsl::narrow<int>(inReal.shape(0) - 1),
                          inReal.data(), optInTimePeriod, outBegIdx,
                          outNBElement, outData);
      });
}

// ---------------------------------------------------------
// MOMENTUM (MOM)
// ---------------------------------------------------------
DoubleArrayOUT mom(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0), TA_MOM_Lookback(optInTimePeriod), "TA_MOM",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_MOM(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// CHANDE MOMENTUM OSCILLATOR (CMO)
// ---------------------------------------------------------
DoubleArrayOUT cmo(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0), TA_CMO_Lookback(optInTimePeriod), "TA_CMO",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_CMO(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInTimePeriod, outBegIdx, outNBElement, outData);
      });
}

// ---------------------------------------------------------
// ABSOLUTE PRICE OSCILLATOR (APO)
// ---------------------------------------------------------
DoubleArrayOUT apo(DoubleArrayIN inReal, int optInFastPeriod = 12,
                   int optInSlowPeriod = 26, int optInMAType = 0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0),
      TA_APO_Lookback(optInFastPeriod, optInSlowPeriod,
                      static_cast<TA_MAType>(optInMAType)),
      "TA_APO", [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_APO(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInFastPeriod, optInSlowPeriod,
                      static_cast<TA_MAType>(optInMAType), outBegIdx,
                      outNBElement, outData);
      });
}

// ---------------------------------------------------------
// PERCENTAGE PRICE OSCILLATOR (PPO)
// ---------------------------------------------------------
DoubleArrayOUT ppo(DoubleArrayIN inReal, int optInFastPeriod = 12,
                   int optInSlowPeriod = 26, int optInMAType = 0) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0),
      TA_PPO_Lookback(optInFastPeriod, optInSlowPeriod,
                      static_cast<TA_MAType>(optInMAType)),
      "TA_PPO", [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_PPO(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                      optInFastPeriod, optInSlowPeriod,
                      static_cast<TA_MAType>(optInMAType), outBegIdx,
                      outNBElement, outData);
      });
}

// ---------------------------------------------------------
// TRIPLE SMOOTHED EXPONENTIAL MOVING AVERAGE (TRIX)
// ---------------------------------------------------------
DoubleArrayOUT trix(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }
  return apply_ta_func(
      inReal.shape(0), TA_TRIX_Lookback(optInTimePeriod), "TA_TRIX",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_TRIX(0, gsl::narrow<int>(inReal.shape(0) - 1), inReal.data(),
                       optInTimePeriod, outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func_2out(
      inHigh.shape(0), TA_AROON_Lookback(optInTimePeriod), "TA_AROON",
      [&](int *outBegIdx, int *outNBElement, double *out1, double *out2) {
        return TA_AROON(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                        inLow.data(), optInTimePeriod, outBegIdx, outNBElement,
                        out1, out2);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_AROONOSC_Lookback(optInTimePeriod), "TA_AROONOSC",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_AROONOSC(0, gsl::narrow<int>(inHigh.shape(0) - 1),
                           inHigh.data(), inLow.data(), optInTimePeriod,
                           outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_ADX_Lookback(optInTimePeriod), "TA_ADX",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_ADX(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                      inLow.data(), inClose.data(), optInTimePeriod, outBegIdx,
                      outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_ADXR_Lookback(optInTimePeriod), "TA_ADXR",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_ADXR(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                       inLow.data(), inClose.data(), optInTimePeriod, outBegIdx,
                       outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_DX_Lookback(optInTimePeriod), "TA_DX",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_DX(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                     inLow.data(), inClose.data(), optInTimePeriod, outBegIdx,
                     outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_MINUS_DI_Lookback(optInTimePeriod), "TA_MINUS_DI",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_MINUS_DI(0, gsl::narrow<int>(inHigh.shape(0) - 1),
                           inHigh.data(), inLow.data(), inClose.data(),
                           optInTimePeriod, outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_MINUS_DM_Lookback(optInTimePeriod), "TA_MINUS_DM",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_MINUS_DM(0, gsl::narrow<int>(inHigh.shape(0) - 1),
                           inHigh.data(), inLow.data(), optInTimePeriod,
                           outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_PLUS_DI_Lookback(optInTimePeriod), "TA_PLUS_DI",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_PLUS_DI(0, gsl::narrow<int>(inHigh.shape(0) - 1),
                          inHigh.data(), inLow.data(), inClose.data(),
                          optInTimePeriod, outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_PLUS_DM_Lookback(optInTimePeriod), "TA_PLUS_DM",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_PLUS_DM(0, gsl::narrow<int>(inHigh.shape(0) - 1),
                          inHigh.data(), inLow.data(), optInTimePeriod,
                          outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_WILLR_Lookback(optInTimePeriod), "TA_WILLR",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_WILLR(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                        inLow.data(), inClose.data(), optInTimePeriod,
                        outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_MFI_Lookback(optInTimePeriod), "TA_MFI",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_MFI(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                      inLow.data(), inClose.data(), inVolume.data(),
                      optInTimePeriod, outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0), TA_CCI_Lookback(optInTimePeriod), "TA_CCI",
      [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_CCI(0, gsl::narrow<int>(inHigh.shape(0) - 1), inHigh.data(),
                      inLow.data(), inClose.data(), optInTimePeriod, outBegIdx,
                      outNBElement, outData);
      });
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
  return apply_ta_func(
      inHigh.shape(0),
      TA_ULTOSC_Lookback(optInTimePeriod1, optInTimePeriod2, optInTimePeriod3),
      "TA_ULTOSC", [&](int *outBegIdx, int *outNBElement, double *outData) {
        return TA_ULTOSC(0, gsl::narrow<int>(inHigh.shape(0) - 1),
                         inHigh.data(), inLow.data(), inClose.data(),
                         optInTimePeriod1, optInTimePeriod2, optInTimePeriod3,
                         outBegIdx, outNBElement, outData);
      });
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
  return apply_ta_func(inOpen.shape(0), TA_BOP_Lookback(), "TA_BOP",
                       [&](int *outBegIdx, int *outNBElement, double *outData) {
                         return TA_BOP(0, gsl::narrow<int>(inOpen.shape(0) - 1),
                                       inOpen.data(), inHigh.data(),
                                       inLow.data(), inClose.data(), outBegIdx,
                                       outNBElement, outData);
                       });
}
