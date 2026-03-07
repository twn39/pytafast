#pragma once

#include <gsl/gsl>
#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <string>
#include <ta_libc.h>

namespace nb = nanobind;

// Type aliases for numpy array I/O
using DoubleArrayIN =
    nb::ndarray<nb::numpy, const double, nb::c_contig, nb::ndim<1>>;
using DoubleArrayOUT = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

static const double NaN = std::numeric_limits<double>::quiet_NaN();

// Check TA-Lib return codes
inline void check_ta_retcode(TA_RetCode code, const char *func) {
  if (code != TA_SUCCESS) {
    throw std::runtime_error(
        std::string(func) + " failed with TA_RetCode: " + std::to_string(code));
  }
}

// Helper: allocate a double array, wrap in capsule, fill lookback region with
// NaN
struct AllocResult {
  gsl::not_null<double *> data;
  nb::capsule owner;
};

inline AllocResult alloc_output(size_t size, int lookback) {
  auto *data = new double[size];
  nb::capsule owner(data, [](void *p) noexcept { delete[] (double *)p; });

  // Only fill the lookback region which won't be touched by TA-Lib
  if (lookback > 0 && size > 0) {
    gsl::span<double> data_span(data,
                                std::min(static_cast<size_t>(lookback), size));
    std::fill(data_span.begin(), data_span.end(), NaN);
  }

  return {gsl::make_not_null(data), std::move(owner)};
}

struct AllocIntResult {
  gsl::not_null<int *> data;
  nb::capsule owner;
};

inline AllocIntResult alloc_int_output(size_t size, int lookback) {
  auto *data = new int[size];
  nb::capsule owner(data, [](void *p) noexcept { delete[] (int *)p; });

  if (lookback > 0 && size > 0) {
    gsl::span<int> data_span(data,
                             std::min(static_cast<size_t>(lookback), size));
    std::fill(data_span.begin(), data_span.end(), 0);
  }

  return {gsl::make_not_null(data), std::move(owner)};
}
