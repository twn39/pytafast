#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <string>
#include <ta_libc.h>

namespace nb = nanobind;

// Define a type alias for our preferred input array type
using DoubleArrayIN = nb::ndarray<double, nb::c_contig, nb::device::cpu>;
// Define a type alias for out output numpy array
using DoubleArrayOUT = nb::ndarray<nb::numpy, double>;

// Helper to check return codes
void check_ta_retcode(TA_RetCode code, const char *func) {
  if (code != TA_SUCCESS) {
    throw std::runtime_error(
        std::string(func) + " failed with TA_RetCode: " + std::to_string(code));
  }
}

DoubleArrayOUT sma(DoubleArrayIN inReal, int optInTimePeriod = 30) {
  if (inReal.ndim() != 1) {
    throw std::runtime_error("Input must be a 1D array");
  }

  int startIdx = 0;
  int endIdx = static_cast<int>(inReal.shape(0)) - 1;

  if (endIdx < 0) {
    // Empty array, return an empty array
    return DoubleArrayOUT(nullptr, {0}, nb::handle());
  }

  size_t outSize = inReal.shape(0);
  double *outData = new double[outSize];
  // Fill the array with NaNs since TA-Lib doesn't fill the beginning
  for (size_t i = 0; i < outSize; ++i) {
    outData[i] = std::numeric_limits<double>::quiet_NaN();
  }

  int outBegIdx = 0;
  int outNBElement = 0;

  int lookback = TA_SMA_Lookback(optInTimePeriod);

  TA_RetCode retCode = TA_SMA(startIdx, endIdx, inReal.data(), optInTimePeriod,
                              &outBegIdx, &outNBElement, outData + lookback);

  check_ta_retcode(retCode, "TA_SMA");

  // Make sure it alignes exactly
  if (outBegIdx != lookback) {
    // Edge case, need to move data
    // Actually this shouldn't happen for SMA
  }

  nb::capsule owner(outData, [](void *p) noexcept { delete[] (double *)p; });

  return DoubleArrayOUT(outData, {outSize}, owner);
}

void initialize() {
  TA_RetCode retCode = TA_Initialize();
  check_ta_retcode(retCode, "TA_Initialize");
}

void shutdown() { TA_Shutdown(); }

NB_MODULE(pytalib_ext, m) {
  m.def("initialize", &initialize, "Initialize TA-Lib");
  m.def("shutdown", &shutdown, "Shutdown TA-Lib");

  m.def("SMA", &sma, "Simple Moving Average", nb::arg("inReal"),
        nb::arg("timeperiod") = 30);
}
