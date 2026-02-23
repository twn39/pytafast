#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <string>
#include <ta_libc.h>

namespace nb = nanobind;

// Define a type alias for our preferred input array type
using DoubleArrayIN =
    nb::ndarray<nb::numpy, const double, nb::c_contig, nb::ndim<1>>;
// Define a type alias for out output numpy array
using DoubleArrayOUT = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// Check return codes
void check_ta_retcode(TA_RetCode code, const char *func) {
  if (code != TA_SUCCESS) {
    throw std::runtime_error(
        std::string(func) + " failed with TA_RetCode: " + std::to_string(code));
  }
}

// ---------------------------------------------------------
// SIMPLE MOVING AVERAGE
// ---------------------------------------------------------
DoubleArrayOUT sma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0) {
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }

  int startIdx = 0;
  int endIdx = inReal.shape(0) - 1;

  // Allocate the output numpy array. It will have the same size as input.
  // We will initialize it with NaNs so the leading empty values are NaN.
  double *outData = new double[inReal.shape(0)];
  for (size_t i = 0; i < inReal.shape(0); ++i)
    outData[i] = std::numeric_limits<double>::quiet_NaN();

  int outBegIdx = 0;
  int outNBElement = 0;

  int lookback = TA_SMA_Lookback(optInTimePeriod);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_SMA(startIdx, endIdx, inReal.data(), optInTimePeriod,
                     &outBegIdx, &outNBElement, outData + lookback);
  }

  check_ta_retcode(retCode, "TA_SMA");

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });

  return DoubleArrayOUT(outData, {inReal.shape(0)}, owner);
}

// ---------------------------------------------------------
// EXPONENTIAL MOVING AVERAGE
// ---------------------------------------------------------
DoubleArrayOUT ema(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  int startIdx = 0;
  int endIdx = inReal.shape(0) - 1;

  double *outData = new double[inReal.shape(0)];
  for (size_t i = 0; i < inReal.shape(0); ++i)
    outData[i] = std::numeric_limits<double>::quiet_NaN();

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_EMA_Lookback(optInTimePeriod);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_EMA(startIdx, endIdx, inReal.data(), optInTimePeriod,
                     &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_EMA");

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });
  return DoubleArrayOUT(outData, {inReal.shape(0)}, owner);
}

// ---------------------------------------------------------
// RELATIVE STRENGTH INDEX
// ---------------------------------------------------------
DoubleArrayOUT rsi(DoubleArrayIN inReal, int optInTimePeriod = 14) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  int startIdx = 0;
  int endIdx = inReal.shape(0) - 1;

  double *outData = new double[inReal.shape(0)];
  for (size_t i = 0; i < inReal.shape(0); ++i)
    outData[i] = std::numeric_limits<double>::quiet_NaN();

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_RSI_Lookback(optInTimePeriod);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_RSI(startIdx, endIdx, inReal.data(), optInTimePeriod,
                     &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_RSI");

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });
  return DoubleArrayOUT(outData, {inReal.shape(0)}, owner);
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
  int startIdx = 0;
  int endIdx = inReal.shape(0) - 1;

  double *outMACD = new double[inReal.shape(0)];
  double *outSignal = new double[inReal.shape(0)];
  double *outHist = new double[inReal.shape(0)];

  for (size_t i = 0; i < inReal.shape(0); ++i) {
    outMACD[i] = std::numeric_limits<double>::quiet_NaN();
    outSignal[i] = std::numeric_limits<double>::quiet_NaN();
    outHist[i] = std::numeric_limits<double>::quiet_NaN();
  }

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback =
      TA_MACD_Lookback(optInFastPeriod, optInSlowPeriod, optInSignalPeriod);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_MACD(startIdx, endIdx, inReal.data(), optInFastPeriod,
                optInSlowPeriod, optInSignalPeriod, &outBegIdx, &outNBElement,
                outMACD + lookback, outSignal + lookback, outHist + lookback);
  }
  check_ta_retcode(retCode, "TA_MACD");

  nb::capsule owner1(outMACD, [](void *p) noexcept { delete[] (double *)p; });
  nb::capsule owner2(outSignal, [](void *p) noexcept { delete[] (double *)p; });
  nb::capsule owner3(outHist, [](void *p) noexcept { delete[] (double *)p; });

  return nb::make_tuple(DoubleArrayOUT(outMACD, {inReal.shape(0)}, owner1),
                        DoubleArrayOUT(outSignal, {inReal.shape(0)}, owner2),
                        DoubleArrayOUT(outHist, {inReal.shape(0)}, owner3));
}

// ---------------------------------------------------------
// BOLLINGER BANDS
// ---------------------------------------------------------
nb::tuple bbands(DoubleArrayIN inReal, int optInTimePeriod = 5,
                 double optInNbDevUp = 2.0, double optInNbDevDn = 2.0,
                 int optInMAType = 0) {
  if (inReal.size() == 0) {
    return nb::make_tuple(DoubleArrayOUT(nullptr, {0}, nb::handle()),
                          DoubleArrayOUT(nullptr, {0}, nb::handle()),
                          DoubleArrayOUT(nullptr, {0}, nb::handle()));
  }
  int startIdx = 0;
  int endIdx = inReal.shape(0) - 1;

  double *outUpper = new double[inReal.shape(0)];
  double *outMiddle = new double[inReal.shape(0)];
  double *outLower = new double[inReal.shape(0)];

  for (size_t i = 0; i < inReal.shape(0); ++i) {
    outUpper[i] = std::numeric_limits<double>::quiet_NaN();
    outMiddle[i] = std::numeric_limits<double>::quiet_NaN();
    outLower[i] = std::numeric_limits<double>::quiet_NaN();
  }

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_BBANDS_Lookback(optInTimePeriod, optInNbDevUp, optInNbDevDn,
                                    (TA_MAType)optInMAType);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_BBANDS(startIdx, endIdx, inReal.data(), optInTimePeriod,
                        optInNbDevUp, optInNbDevDn, (TA_MAType)optInMAType,
                        &outBegIdx, &outNBElement, outUpper + lookback,
                        outMiddle + lookback, outLower + lookback);
  }
  check_ta_retcode(retCode, "TA_BBANDS");

  nb::capsule owner1(outUpper, [](void *p) noexcept { delete[] (double *)p; });
  nb::capsule owner2(outMiddle, [](void *p) noexcept { delete[] (double *)p; });
  nb::capsule owner3(outLower, [](void *p) noexcept { delete[] (double *)p; });

  return nb::make_tuple(DoubleArrayOUT(outUpper, {inReal.shape(0)}, owner1),
                        DoubleArrayOUT(outMiddle, {inReal.shape(0)}, owner2),
                        DoubleArrayOUT(outLower, {inReal.shape(0)}, owner3));
}

// ---------------------------------------------------------
// AVERAGE TRUE RANGE (ATR)
// ---------------------------------------------------------
DoubleArrayOUT atr(DoubleArrayIN inHigh, DoubleArrayIN inLow,
                   DoubleArrayIN inClose, int optInTimePeriod = 14) {
  if (inHigh.size() == 0 || inLow.size() == 0 || inClose.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inHigh.shape(0) != inLow.shape(0) || inHigh.shape(0) != inClose.shape(0))
    throw std::runtime_error("Input lengths must match");
  int startIdx = 0;
  int endIdx = inHigh.shape(0) - 1;

  double *outData = new double[inHigh.shape(0)];
  for (size_t i = 0; i < inHigh.shape(0); ++i)
    outData[i] = std::numeric_limits<double>::quiet_NaN();

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_ATR_Lookback(optInTimePeriod);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ATR(startIdx, endIdx, inHigh.data(), inLow.data(), inClose.data(),
               optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ATR");

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });
  return DoubleArrayOUT(outData, {inHigh.shape(0)}, owner);
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
  int startIdx = 0;
  int endIdx = inHigh.shape(0) - 1;

  double *outData = new double[inHigh.shape(0)];
  for (size_t i = 0; i < inHigh.shape(0); ++i)
    outData[i] = std::numeric_limits<double>::quiet_NaN();

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_ADX_Lookback(optInTimePeriod);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_ADX(startIdx, endIdx, inHigh.data(), inLow.data(), inClose.data(),
               optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ADX");

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });
  return DoubleArrayOUT(outData, {inHigh.shape(0)}, owner);
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
  int startIdx = 0;
  int endIdx = inHigh.shape(0) - 1;

  double *outData = new double[inHigh.shape(0)];
  for (size_t i = 0; i < inHigh.shape(0); ++i)
    outData[i] = std::numeric_limits<double>::quiet_NaN();

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_CCI_Lookback(optInTimePeriod);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode =
        TA_CCI(startIdx, endIdx, inHigh.data(), inLow.data(), inClose.data(),
               optInTimePeriod, &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_CCI");

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });
  return DoubleArrayOUT(outData, {inHigh.shape(0)}, owner);
}

// ---------------------------------------------------------
// ON BALANCE VOLUME (OBV)
// ---------------------------------------------------------
DoubleArrayOUT obv(DoubleArrayIN inReal, DoubleArrayIN inVolume) {
  if (inReal.size() == 0 || inVolume.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  if (inReal.shape(0) != inVolume.shape(0))
    throw std::runtime_error("Input lengths must match");
  int startIdx = 0;
  int endIdx = inReal.shape(0) - 1;

  double *outData = new double[inReal.shape(0)];
  for (size_t i = 0; i < inReal.shape(0); ++i)
    outData[i] = std::numeric_limits<double>::quiet_NaN();

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_OBV_Lookback();

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_OBV(startIdx, endIdx, inReal.data(), inVolume.data(),
                     &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_OBV");

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });
  return DoubleArrayOUT(outData, {inReal.shape(0)}, owner);
}

// ---------------------------------------------------------
// RATE OF CHANGE (ROC)
// ---------------------------------------------------------
DoubleArrayOUT roc(DoubleArrayIN inReal, int optInTimePeriod = 10) {
  if (inReal.size() == 0)
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  int startIdx = 0;
  int endIdx = inReal.shape(0) - 1;

  double *outData = new double[inReal.shape(0)];
  for (size_t i = 0; i < inReal.shape(0); ++i)
    outData[i] = std::numeric_limits<double>::quiet_NaN();

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_ROC_Lookback(optInTimePeriod);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_ROC(startIdx, endIdx, inReal.data(), optInTimePeriod,
                     &outBegIdx, &outNBElement, outData + lookback);
  }
  check_ta_retcode(retCode, "TA_ROC");

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });
  return DoubleArrayOUT(outData, {inReal.shape(0)}, owner);
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
  int startIdx = 0;
  int endIdx = inHigh.shape(0) - 1;

  double *outSlowK = new double[inHigh.shape(0)];
  double *outSlowD = new double[inHigh.shape(0)];

  for (size_t i = 0; i < inHigh.shape(0); ++i) {
    outSlowK[i] = std::numeric_limits<double>::quiet_NaN();
    outSlowD[i] = std::numeric_limits<double>::quiet_NaN();
  }

  int outBegIdx = 0;
  int outNBElement = 0;
  int lookback = TA_STOCH_Lookback(
      optInFastK_Period, optInSlowK_Period, (TA_MAType)optInSlowK_MAType,
      optInSlowD_Period, (TA_MAType)optInSlowD_MAType);

  TA_RetCode retCode;
  {
    nb::gil_scoped_release release;
    retCode = TA_STOCH(startIdx, endIdx, inHigh.data(), inLow.data(),
                       inClose.data(), optInFastK_Period, optInSlowK_Period,
                       (TA_MAType)optInSlowK_MAType, optInSlowD_Period,
                       (TA_MAType)optInSlowD_MAType, &outBegIdx, &outNBElement,
                       outSlowK + lookback, outSlowD + lookback);
  }
  check_ta_retcode(retCode, "TA_STOCH");

  nb::capsule owner1(outSlowK, [](void *p) noexcept { delete[] (double *)p; });
  nb::capsule owner2(outSlowD, [](void *p) noexcept { delete[] (double *)p; });

  return nb::make_tuple(DoubleArrayOUT(outSlowK, {inHigh.shape(0)}, owner1),
                        DoubleArrayOUT(outSlowD, {inHigh.shape(0)}, owner2));
}

// Helper to initialize and shutdown TA-lib
void initialize() {
  TA_RetCode retcode = TA_Initialize();
  check_ta_retcode(retcode, "TA_Initialize");
}

void shutdown() {
  TA_RetCode retcode = TA_Shutdown();
  check_ta_retcode(retcode, "TA_Shutdown");
}

NB_MODULE(pytafast_ext, m) {
  m.doc() = "TA-Lib wrapper using nanobind";

  nb::enum_<TA_MAType>(m, "MAType")
      .value("SMA", TA_MAType_SMA)
      .value("EMA", TA_MAType_EMA)
      .value("WMA", TA_MAType_WMA)
      .value("DEMA", TA_MAType_DEMA)
      .value("TEMA", TA_MAType_TEMA)
      .value("TRIMA", TA_MAType_TRIMA)
      .value("KAMA", TA_MAType_KAMA)
      .value("MAMA", TA_MAType_MAMA)
      .value("T3", TA_MAType_T3);

  m.def("SMA", &sma, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("EMA", &ema, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 30);
  m.def("RSI", &rsi, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("MACD", &macd, nb::arg("inReal").noconvert(),
        nb::arg("optInFastPeriod") = 12, nb::arg("optInSlowPeriod") = 26,
        nb::arg("optInSignalPeriod") = 9);
  m.def("BBANDS", &bbands, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 5, nb::arg("optInNbDevUp") = 2.0,
        nb::arg("optInNbDevDn") = 2.0, nb::arg("optInMAType") = 0);

  // Additional Indicators
  m.def("ATR", &atr, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("ADX", &adx, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("CCI", &cci, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInTimePeriod") = 14);
  m.def("OBV", &obv, nb::arg("inReal").noconvert(),
        nb::arg("inVolume").noconvert());
  m.def("ROC", &roc, nb::arg("inReal").noconvert(),
        nb::arg("optInTimePeriod") = 10);
  m.def("STOCH", &stoch, nb::arg("inHigh").noconvert(),
        nb::arg("inLow").noconvert(), nb::arg("inClose").noconvert(),
        nb::arg("optInFastK_Period") = 5, nb::arg("optInSlowK_Period") = 3,
        nb::arg("optInSlowK_MAType") = 0, nb::arg("optInSlowD_Period") = 3,
        nb::arg("optInSlowD_MAType") = 0);

  m.def("initialize", &initialize);
  m.def("shutdown", &shutdown);
}
