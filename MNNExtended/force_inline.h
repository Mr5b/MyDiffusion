#ifndef FORCE_INLINE_H
#define FORCE_INLINE_H

namespace MyTensor::Utils
{

#if defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE __attribute__((always_inline, flatten)) inline
    #define UNROLL_LOOP _Pragma("GCC unroll 4")
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
    #define UNROLL_LOOP __pragma(unroll(4))
#else
    #define FORCE_INLINE inline
    #define UNROLL_LOOP
#endif

}//namespace MyTensor::Utils

#endif