#pragma once

#include <gsl/gsl>
#include <limits>
#include <memory>
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

// Helper: allocate a double array, wrap in capsule, fill lookback prefix with
// NaN. Only the [0, lookback) region needs initializing; TA-Lib fills
// [lookback, size).
struct AllocResult {
  gsl::not_null<double *> data;
  nb::capsule owner;
};

inline AllocResult alloc_output(size_t size, int lookback) {
  std::unique_ptr<double[]> data(new double[size]);
  nb::capsule owner(data.get(), [](void *p) noexcept { delete[] (double *)p; });

  // Only fill the lookback prefix with NaN; TA-Lib writes all remaining
  // elements.
  if (lookback > 0 && size > 0) {
    std::fill(data.get(),
              data.get() + std::min(static_cast<size_t>(lookback), size), NaN);
  }

  return {gsl::make_not_null(data.release()), std::move(owner)};
}

struct AllocIntResult {
  gsl::not_null<int *> data;
  nb::capsule owner;
};

inline AllocIntResult alloc_int_output(size_t size, int /*lookback*/) {
  // Value-initialize (zero-fill) the entire array via new int[size]().
  // This is correct for both the NaN-prefix region and the TA-Lib output
  // region.
  std::unique_ptr<int[]> data(new int[size]());
  nb::capsule owner(data.get(), [](void *p) noexcept { delete[] (int *)p; });
  return {gsl::make_not_null(data.release()), std::move(owner)};
}

// Helper: validate that all array sizes in the list equal `expected`.
// Usage: check_lengths("funcname", size, {inLow.shape(0), inClose.shape(0)});
inline void check_lengths(const char *func, size_t expected,
                          std::initializer_list<size_t> others) {
  for (size_t s : others) {
    if (s != expected) {
      throw std::runtime_error(std::string(func) +
                               ": input array lengths must match");
    }
  }
}
