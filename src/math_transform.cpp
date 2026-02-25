// Math Transforms: ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10,
// SIN, SINH, SQRT, TAN, TANH
#include "common.h"

// Helper macro for single-input no-param transforms
#define MATH_TRANSFORM_FUNC(NAME, TA_FUNC)                                     \
  DoubleArrayOUT NAME(DoubleArrayIN inReal) {                                  \
    if (inReal.size() == 0) return DoubleArrayOUT(nullptr, {0}, nb::handle()); \
    size_t size = inReal.shape(0);                                             \
    int lookback = TA_FUNC##_Lookback();                                       \
    auto [outData, owner] = alloc_output(size, lookback);                      \
    int outBegIdx = 0, outNBElement = 0;                                       \
    TA_RetCode retCode;                                                        \
    {                                                                          \
      nb::gil_scoped_release release;                                          \
      retCode = TA_FUNC(0, size - 1, inReal.data(), &outBegIdx, &outNBElement, \
                        outData + lookback);                                   \
    }                                                                          \
    check_ta_retcode(retCode, #TA_FUNC);                                       \
    return DoubleArrayOUT(outData, {size}, owner);                             \
  }

MATH_TRANSFORM_FUNC(ta_acos, TA_ACOS)
MATH_TRANSFORM_FUNC(ta_asin, TA_ASIN)
MATH_TRANSFORM_FUNC(ta_atan, TA_ATAN)
MATH_TRANSFORM_FUNC(ta_ceil, TA_CEIL)
MATH_TRANSFORM_FUNC(ta_cos, TA_COS)
MATH_TRANSFORM_FUNC(ta_cosh, TA_COSH)
MATH_TRANSFORM_FUNC(ta_exp, TA_EXP)
MATH_TRANSFORM_FUNC(ta_floor, TA_FLOOR)
MATH_TRANSFORM_FUNC(ta_ln, TA_LN)
MATH_TRANSFORM_FUNC(ta_log10, TA_LOG10)
MATH_TRANSFORM_FUNC(ta_sin, TA_SIN)
MATH_TRANSFORM_FUNC(ta_sinh, TA_SINH)
MATH_TRANSFORM_FUNC(ta_sqrt, TA_SQRT)
MATH_TRANSFORM_FUNC(ta_tan, TA_TAN)
MATH_TRANSFORM_FUNC(ta_tanh, TA_TANH)

#undef MATH_TRANSFORM_FUNC
