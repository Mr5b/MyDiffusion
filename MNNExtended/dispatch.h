#ifndef MY_MODULE_DISPATCH_H
#define MY_MODULE_DISPATCH_H

#include <stdexcept>
#include <cstdint>
#include <string>

#include <MNN/HalideRuntime.h>

#include "half.h"

namespace MyTensor::Utils
{

inline void print_halide_type(const halide_type_t& type)
{
    
}

#define DISPATCH_SWITCH(TYPE, NAME, ...)                                    \
[&]                                                                         \
{                                                                           \
    const int type_key = (static_cast<int>((TYPE).code) << 8) | (TYPE).bits; \
    switch (type_key)                                                       \
    {                                                                       \
        __VA_ARGS__                                                         \
        default:                                                            \
            throw std::runtime_error                                       \
            (                                                               \
                NAME + std::string(" not implemented for dtype (code=") +  \
                std::to_string((TYPE).code) + ", bits=" +                  \
                std::to_string((TYPE).bits) + ")"                          \
            );                                                              \
    }                                                                       \
}()

#define DISPATCH_CASE_FLOAT_TYPES(LAMBDA)                                   \
    case ((halide_type_float << 8) | 16):                                   \
    {                                                                       \
        using scalar_t = MyTensor::Utils::Float16;                          \
        LAMBDA();                                                           \
    } break;                                                                \
    case ((halide_type_float << 8) | 32):                                   \
    {                                                                       \
        using scalar_t = float;                                             \
        LAMBDA();                                                           \
    } break;                                                                \
    case ((halide_type_float << 8) | 64):                                   \
    {                                                                       \
        using scalar_t = double;                                            \
        LAMBDA();                                                           \
    } break;

#define DISPATCH_CASE_INT_TYPES(LAMBDA)                                     \
    case ((halide_type_int << 8) | 32):                                     \
    {                                                                       \
        using scalar_t = int32_t;                                           \
        LAMBDA();                                                           \
    } break;                                                                \
    case ((halide_type_uint << 8) | 8):                                     \
    {                                                                       \
        using scalar_t = uint8_t;                                           \
        LAMBDA();                                                           \
    } break;

#define DISPATCH_CASE_ALL_TYPES(LAMBDA)                                     \
    DISPATCH_CASE_FLOAT_TYPES(LAMBDA)                                       \
    DISPATCH_CASE_INT_TYPES(LAMBDA)

#define DISPATCH_ALL_TYPES(TYPE, NAME, LAMBDA)                              \
    DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_ALL_TYPES(LAMBDA))

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, LAMBDA)                            \
    DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(LAMBDA))

} // namespace MyTensor::Utils

#endif