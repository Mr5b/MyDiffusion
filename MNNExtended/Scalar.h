#ifndef SCALAR_H
#define SCALAR_H

#include <cstdint>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <limits>

#include <MNN/HalideRuntime.h>
#include "half.h"

namespace MyTensor::Utils
{

#define MY_FORALL_SCALAR_TYPES(_)   \
    _(int, Int)                     \
    _(uint8_t, UInt8)               \
    _(int64_t, Long)                \
    _(double, Double)               \
    _(float, Float)                 \
    _(bool, Bool)                   \
    _(std::complex<double>, ComplexDouble)\
    _(Float16, Half)

#define MY_FORALL_SCALAR_TYPES_WITH_COMPLEX(_)   \
    MY_FORALL_SCALAR_TYPES(_)                    \
    _(uint16_t, UInt16)                          \
    _(uint32_t, UInt32)                          \
    _(uint64_t, UInt64)
    ///////////////////////////////////////

class Scalar
{

public:
    enum class Tag { HAS_d, HAS_i, HAS_u, HAS_z, HAS_b };

    Scalar() : tag(Tag::HAS_i)
    {
        v.i = 0;
    }

    template <typename T>
    Scalar(T value)
    {
        init(value);
    }


    Scalar(uint64_t value)
    {
        if (value > static_cast<uint64_t>(INT64_MAX))
        {
            tag = Tag::HAS_u;
            v.u = value;
        }
        else
        {
            tag = Tag::HAS_i;
            v.i = static_cast<int64_t>(value);
        }
    }

    Scalar(const Scalar& other) : tag(other.tag)
    {
        switch (tag)
        {
            case Tag::HAS_d: v.d = other.v.d; break;
            case Tag::HAS_i: v.i = other.v.i; break;
            case Tag::HAS_u: v.u = other.v.u; break;
            case Tag::HAS_z: v.z = other.v.z; break;
            case Tag::HAS_b: v.b = other.v.b; break;
        }
    }

    Scalar(Scalar&& other) noexcept : tag(other.tag)
    {
        switch (tag)
        {
            case Tag::HAS_d: v.d = other.v.d; break;
            case Tag::HAS_i: v.i = other.v.i; break;
            case Tag::HAS_u: v.u = other.v.u; break;
            case Tag::HAS_z: v.z = std::move(other.v.z); break;
            case Tag::HAS_b: v.b = other.v.b; break;
        }
        other.tag = Tag::HAS_i;
        other.v.i = 0;
    }

    bool isIntegral(bool includeBool = false) const
    {
        return (tag == Tag::HAS_i || tag == Tag::HAS_u) &&
        !(includeBool && isBoolean());
    }

    bool isFloatingPoint() const { return tag == Tag::HAS_d; }
    bool isBoolean() const      { return tag == Tag::HAS_b; }
    bool isComplex() const      { return tag == Tag::HAS_z; }

    /*int64_t toInt() const
    {
        if (tag != Tag::HAS_i) throwBadTypeError("int64_t");
        return v.i;
    }*/

    /*double toDouble() const
    {
        if (tag != Tag::HAS_d) throwBadTypeError("double");
        return v.d;
    }*/

    /*bool toBool() const
    {
        if (tag != Tag::HAS_b) throwBadTypeError("bool");
        return v.b;
    }*/

    /*template <typename T>
    T to() const
    {
        if constexpr (std::is_same_v<T, int64_t>) return toInt();
        else if constexpr (std::is_same_v<T, double>) return toDouble();
        else if constexpr (std::is_same_v<T, bool>) return toBool();
        else static_assert(!std::is_same_v<T, T>, "Unsupported type");
    }*/
    
    template <typename T>
    T to() const = delete;


    template <typename T>
    struct is_complex : std::false_type {};
    
    template <typename T>
    struct is_complex<std::complex<T>> : std::true_type {};
    
    template <typename T>
    static constexpr bool is_complex_v = is_complex<T>::value;

    #define DEFINE_ACCESSOR(type, name)                 \
    type to##name() const                               \
    {                                                   \
        if (tag == Tag::HAS_d)                          \
        {                                               \
            if constexpr (is_complex_v<type>)           \
            {                                           \
                throw std::runtime_error("Cannot convert double to complex"); \
            }                                           \
            return checked_convert<type, double>(v.d);  \
        }                                               \
        else if (tag == Tag::HAS_i)                     \
        {                                               \
            if constexpr (is_complex_v<type>)           \
            {                                           \
                throw std::runtime_error("Cannot convert int to complex"); \
            }                                           \
            return checked_convert<type, int64_t>(v.i); \
        }                                               \
        else if (tag == Tag::HAS_u)                     \
        {                                               \
            if constexpr (is_complex_v<type>)           \
            {                                           \
                throw std::runtime_error("Cannot convert uint to complex"); \
            }                                           \
            return checked_convert<type, uint64_t>(v.u);\
        }                                               \
        else if (tag == Tag::HAS_z)                     \
        {                                               \
            if constexpr (!is_complex_v<type>)          \
            {                                           \
                throw std::runtime_error("Cannot convert complex to non-complex"); \
            }                                           \
            return checked_convert<type, std::complex<double>>(v.z);\
        }                                               \
        else if (tag == Tag::HAS_b)                     \
        {                                               \
            if constexpr (is_complex_v<type>)           \
            {                                           \
                throw std::runtime_error("Cannot convert bool to complex"); \
            }                                           \
            return checked_convert<type, bool>(v.b);    \
        }                                               \
        throw std::runtime_error("Invalid scalar tag"); \
    }

    MY_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ACCESSOR)
    #undef DEFINE_ACCESSOR

    halide_type_t type() const
    {
        if (isBoolean())
        {
            return halide_type_t(halide_type_uint, 1);   // bool
        }
        else if (isIntegral(false))
        {
            if (tag == Tag::HAS_u)
            {
                return halide_type_t(halide_type_uint, 64); // uint64
            }
            return halide_type_t(halide_type_int, 64);      // int64
        }
        else if (isFloatingPoint())
        {
            return halide_type_t(halide_type_float, 64);    // double
        }
        else if (isComplex())
        {
            
#ifdef halide_type_complex
            return halide_type_t(halide_type_complex, 128); // complex<double> 128 bits
#else
            return halide_type_t(halide_type_float, 64);
#endif
        }
        else
        {
            return halide_type_t(halide_type_handle, 0);//Unknown
        }
    }

private:
    union v_t
    {
        double d;
        int64_t i;
        uint64_t u;
        std::complex<double> z;
        bool b;

        v_t() : i(0) {}
    }v;

    Tag tag;

    template <typename T>
    void init(T value)
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            tag = Tag::HAS_d;
            v.d = static_cast<double>(value);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            tag = Tag::HAS_b;
            v.b = value;
        }
        else if constexpr (std::is_integral_v<T>)
        {
            tag = Tag::HAS_i;
            v.i = static_cast<int64_t>(value);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>)
        {
            tag = Tag::HAS_z;
            v.z = value;
        }
        else
        {
            static_assert(!std::is_same_v<T, T>, "Unsupported type");
        }
    }
    
    /*template <typename To, typename From>
    static To checked_convert(From value)
    {
        if constexpr (std::is_integral_v<To> && std::is_floating_point_v<From>)
        {
            if
            (
                value < static_cast<From>(std::numeric_limits<To>::min()) ||
                value > static_cast<From>(std::numeric_limits<To>::max())
            )
            {
                throw std::overflow_error("Value overflows target type");
            }
        }
        return static_cast<To>(value);
    }*/
    
    template <typename To, typename From>
    static To checked_convert(From value)
    {
        if constexpr (is_complex_v<To> && is_complex_v<From>)
        {
            return static_cast<To>(value);
        }
        else if constexpr (!is_complex_v<To> && !is_complex_v<From>)
        {
            if constexpr (std::is_integral_v<To> && std::is_floating_point_v<From>)
            {
                if
                (
                    value < static_cast<From>(std::numeric_limits<To>::min()) ||
                    value > static_cast<From>(std::numeric_limits<To>::max())
                )
                {
                    throw std::overflow_error("Value overflows target type");
                }
            }
            return static_cast<To>(value);
        }
        else
        {
            throw std::logic_error("Invalid type conversion");
        }
    }

    void throwBadTypeError(const char* type) const
    {
        throw std::runtime_error("Scalar is not of type " + std::string(type));
    }
};

#define DEFINE_TO(T, name)          \
template <>                         \
inline T Scalar::to<T>() const      \
{                                   \
    return to##name();              \
}

MY_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_TO)
#undef DEFINE_TO

}//namespace MyTensor::Utils

#endif