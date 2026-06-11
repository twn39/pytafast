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
// nb::device::cpu allows accepting numpy, pytorch, jax arrays directly without
// copy
using DoubleArrayIN =
    nb::ndarray<nb::device::cpu, const double, nb::c_contig, nb::ndim<1>>;
using DoubleArrayOUT = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

class ta_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Check TA-Lib return codes
inline void check_ta_retcode(TA_RetCode code, const char *func) {
  if (code != TA_SUCCESS) {
    throw ta_error(std::string(func) +
                   " failed with TA_RetCode: " + std::to_string(code));
  }
}

// Helper: allocate a double array, wrap in capsule, fill lookback prefix with
// NaN.
struct AllocResult {
  gsl::not_null<double *> data;
  nb::capsule owner;
};

inline AllocResult alloc_output(size_t size, int lookback) {
  std::unique_ptr<double[]> data(new double[size]);

  if (lookback > 0 && size > 0) {
    std::fill(data.get(),
              data.get() + std::min(static_cast<size_t>(lookback), size), NaN);
  }

  double *raw_ptr = data.release();
  nb::capsule owner(
      raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });

  return {gsl::make_not_null(raw_ptr), std::move(owner)};
}

struct AllocIntResult {
  gsl::not_null<int *> data;
  nb::capsule owner;
};

inline AllocIntResult alloc_int_output(size_t size, int /*lookback*/) {
  std::unique_ptr<int[]> data(new int[size]());
  int *raw_ptr = data.release();
  nb::capsule owner(raw_ptr,
                    [](void *p) noexcept { delete[] static_cast<int *>(p); });
  return {gsl::make_not_null(raw_ptr), std::move(owner)};
}

// Helper: validate that all array sizes in the list equal `expected`.
inline void check_lengths(const char *func, size_t expected,
                          std::initializer_list<size_t> others) {
  for (size_t s : others) {
    if (s != expected) {
      throw std::invalid_argument(std::string(func) +
                                  ": input array lengths must match");
    }
  }
}

// --- Boilerplate Abstraction Templates ---

template <typename Func>
DoubleArrayOUT apply_ta_func(size_t size, int lookback, const char *name,
                             Func &&compute_func) {
  if (lookback < 0) {
    throw std::invalid_argument(std::string(name) +
                                ": Invalid parameter (lookback < 0)");
  }
  auto [outData, owner] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode =
          compute_func(&outBegIdx, &outNBElement, outData.get() + lookback);
    }
    check_ta_retcode(retCode, name);
  }
  return DoubleArrayOUT(outData.get(), {size}, std::move(owner));
}

template <typename Func>
nb::tuple apply_ta_func_2out(size_t size, int lookback, const char *name,
                             Func &&compute_func) {
  if (lookback < 0) {
    throw std::invalid_argument(std::string(name) +
                                ": Invalid parameter (lookback < 0)");
  }
  auto [out1, owner1] = alloc_output(size, lookback);
  auto [out2, owner2] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = compute_func(&outBegIdx, &outNBElement, out1.get() + lookback,
                             out2.get() + lookback);
    }
    check_ta_retcode(retCode, name);
  }
  return nb::make_tuple(DoubleArrayOUT(out1.get(), {size}, std::move(owner1)),
                        DoubleArrayOUT(out2.get(), {size}, std::move(owner2)));
}

template <typename Func>
nb::tuple apply_ta_func_3out(size_t size, int lookback, const char *name,
                             Func &&compute_func) {
  if (lookback < 0) {
    throw std::invalid_argument(std::string(name) +
                                ": Invalid parameter (lookback < 0)");
  }
  auto [out1, owner1] = alloc_output(size, lookback);
  auto [out2, owner2] = alloc_output(size, lookback);
  auto [out3, owner3] = alloc_output(size, lookback);
  if (size > static_cast<size_t>(lookback)) {
    int outBegIdx = 0, outNBElement = 0;
    TA_RetCode retCode;
    {
      nb::gil_scoped_release release;
      retCode = compute_func(&outBegIdx, &outNBElement, out1.get() + lookback,
                             out2.get() + lookback, out3.get() + lookback);
    }
    check_ta_retcode(retCode, name);
  }
  return nb::make_tuple(DoubleArrayOUT(out1.get(), {size}, std::move(owner1)),
                        DoubleArrayOUT(out2.get(), {size}, std::move(owner2)),
                        DoubleArrayOUT(out3.get(), {size}, std::move(owner3)));
}