#ifndef TFKERNELS_CORE_TYPES_H
#define TFKERNELS_CORE_TYPES_H


#include <stdint.h>

namespace lng {

#ifdef FLOAT_SYS_IS_32_BIT
  typedef float FloatSys;
  const FloatSys FL100 = 100.0f;
  const FloatSys FL10 = 10.0f;
  const FloatSys FL2 = 2.0f;
  const FloatSys FL1 = 1.0f;
  const FloatSys FL05 = 0.5f;
  const FloatSys FL0 = 0.0f;

#else
  typedef double FloatSys;
  const FloatSys FL100 = 100.0;
  const FloatSys FL10 = 10.0;
  const FloatSys FL2 = 2.0;
  const FloatSys FL1 = 1.0;
  const FloatSys FL05 = 0.5;
  const FloatSys FL0 = 0.0;

#endif

inline FloatSys to_float_sys(float val) { return (FloatSys)val; }
inline FloatSys to_float_sys(double val) { return (FloatSys)val; }
inline FloatSys to_float_sys(int val) { return (FloatSys)val; }
inline FloatSys to_float_sys(long val) { return (FloatSys)val; }

typedef int32_t IntSys;

typedef float Flt32;
typedef double Flt64;

#define INSTANTIATE_WITH_ALL_FLT_TYPES(m) m(Flt32) m(Flt64)

}


#endif //TFKERNELS_CORE_TYPES_H
