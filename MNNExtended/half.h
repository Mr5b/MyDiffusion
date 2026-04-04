#ifndef HALF_H
#define HALF_H


#include <cstdint>
#include <iostream>
#include <cmath>
#include <type_traits>
#include <limits>

#include "force_inline.h"


#if defined(__STDCPP_FLOAT16_T__) && __cplusplus >= 202002L
    #define FLOAT16_STANDARD_CXX23 1
    #include <stdfloat>
    #define FLOAT16_ARM_NATIVE 0
    #define FLOAT16_COMPILER_EXT 0
#elif (defined(__ARM_FP16_FORMAT_IEEE) && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)) || \
      defined(__ARM_NEON_FP16) || \
      (defined(__ARM_FP) && (__ARM_FP & 0x2)) || \
      (defined(__aarch64__) && defined(__clang__))
    #define FLOAT16_STANDARD_CXX23 0
    #define FLOAT16_ARM_NATIVE 1
    #define FLOAT16_COMPILER_EXT 0
#elif defined(__FLT16_MANT_DIG__) || defined(__FLOAT16__) || defined(__SIZEOF_FLOAT16__)
    #define FLOAT16_STANDARD_CXX23 0
    #define FLOAT16_ARM_NATIVE 0
    #define FLOAT16_COMPILER_EXT 1
#else
    #define FLOAT16_STANDARD_CXX23 0
    #define FLOAT16_ARM_NATIVE 0
    #define FLOAT16_COMPILER_EXT 0
#endif

namespace MyTensor::Utils
{

class Float16
{
public:
    FORCE_INLINE Float16() noexcept = default;
    
    FORCE_INLINE explicit Float16(float value) noexcept :
    #if FLOAT16_STANDARD_CXX23
        native_(static_cast<std::float16_t>(value))
    #elif FLOAT16_ARM_NATIVE
        native_(static_cast<__fp16>(value))
    #elif FLOAT16_COMPILER_EXT
        native_(static_cast<_Float16>(value))
    #else
        bits_(float_to_half(value))
    #endif
    {
    /*#if FLOAT16_STANDARD_CXX23
        std::cout << "FLOAT16_STANDARD_CXX23" << std::endl;
    #elif FLOAT16_ARM_NATIVE
        std::cout << "FLOAT16_ARM_NATIVE" << std::endl;
    #elif FLOAT16_COMPILER_EXT
        std::cout << "FLOAT16_COMPILER_EXT" << std::endl;
    #else
        std::cout << "NONE" << std::endl;
    #endif*/
    }
    
    FORCE_INLINE static Float16 from_bits(uint16_t bits) noexcept
    {
        #if FLOAT16_STANDARD_CXX23
            return Float16(reinterpret_cast<const std::float16_t&>(bits));
        #elif FLOAT16_ARM_NATIVE
            return Float16(bit_cast_to_native(bits));
        #elif FLOAT16_COMPILER_EXT
            return bit_cast_from_bits(bits);
        #else
            Float16 result;
            result.bits_ = bits;
            return result;
        #endif
    }
    
    FORCE_INLINE uint16_t bits() const noexcept
    {
        #if FLOAT16_STANDARD_CXX23
            return reinterpret_cast<const uint16_t&>(native_);
        #elif FLOAT16_ARM_NATIVE
            uint16_t result;
            __builtin_memcpy(&result, &native_, sizeof(result));
            return result;
        #elif FLOAT16_COMPILER_EXT
            return bit_cast_to_bits();
        #else
            return bits_;
        #endif
    }
    
    FORCE_INLINE explicit operator float() const noexcept
    {
        #if FLOAT16_STANDARD_CXX23
            return static_cast<float>(native_);
        #elif FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return native_;
        #else
            return half_to_float(bits_);
        #endif
    }
    
    FORCE_INLINE Float16 operator+(Float16 rhs) const noexcept
    {
        #if FLOAT16_STANDARD_CXX23
            return Float16(native_ + rhs.native_);
        #elif FLOAT16_ARM_NATIVE
            return Float16(native_ + rhs.native_);
        #elif FLOAT16_COMPILER_EXT
            return Float16(native_ + rhs.native_);
        #else
            return Float16
            (
                static_cast<float>(*this) + 
                static_cast<float>(rhs)
            );
        #endif
    }

    FORCE_INLINE Float16 operator-(Float16 rhs) const noexcept
    {
        #if FLOAT16_STANDARD_CXX23
            return Float16(native_ - rhs.native_);
        #elif FLOAT16_ARM_NATIVE
            return Float16(native_ - rhs.native_);
        #elif FLOAT16_COMPILER_EXT
            return Float16(native_ - rhs.native_);
        #else
            return Float16
            (
                static_cast<float>(*this) - 
                static_cast<float>(rhs)
            );
        #endif
    }

    FORCE_INLINE Float16 operator*(Float16 rhs) const noexcept
    {
        #if FLOAT16_STANDARD_CXX23
            return Float16(native_ * rhs.native_);
        #elif FLOAT16_ARM_NATIVE
            return Float16(native_ * rhs.native_);
        #elif FLOAT16_COMPILER_EXT
            return Float16(native_ * rhs.native_);
        #else
            return Float16
            (
                static_cast<float>(*this) * 
                static_cast<float>(rhs)
            );
        #endif
    }

    FORCE_INLINE Float16 operator/(Float16 rhs) const noexcept
    {
        #if FLOAT16_STANDARD_CXX23
            return Float16(native_ / rhs.native_);
        #elif FLOAT16_ARM_NATIVE
            return Float16(native_ / rhs.native_);
        #elif FLOAT16_COMPILER_EXT
            return Float16(native_ / rhs.native_);
        #else
            return Float16
            (
                static_cast<float>(*this) / 
                static_cast<float>(rhs)
            );
        #endif
    }
    
    FORCE_INLINE bool operator==(Float16 rhs) const noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return native_ == rhs.native_;
        #else
            return
            (bits_ == rhs.bits_) || 
            (is_nan() && rhs.is_nan());
        #endif
    }
    
    FORCE_INLINE bool operator!=(Float16 rhs) const noexcept
    {
        return !(*this == rhs);
    }
    
    FORCE_INLINE bool is_nan() const noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return native_ != native_;
        #else
            return (bits_ & 0x7FFF) > 0x7C00;
        #endif
    }
    
    FORCE_INLINE Float16& operator+=(Float16 rhs) noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            native_ += rhs.native_;
        #else
            float result = static_cast<float>(*this) + static_cast<float>(rhs);
            bits_ = float_to_half(result);
        #endif
        return *this;
    }
    
    FORCE_INLINE Float16& operator-=(Float16 rhs) noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            native_ -= rhs.native_;
        #else
            float result = static_cast<float>(*this) - static_cast<float>(rhs);
            bits_ = float_to_half(result);
        #endif
        return *this;
    }
    
    FORCE_INLINE Float16& operator*=(Float16 rhs) noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            native_ *= rhs.native_;
        #else
            float result = static_cast<float>(*this) * static_cast<float>(rhs);
            bits_ = float_to_half(result);
        #endif
        return *this;
    }
    
    FORCE_INLINE Float16& operator/=(Float16 rhs) noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            native_ /= rhs.native_;
        #else
            float result = static_cast<float>(*this) / static_cast<float>(rhs);
            bits_ = float_to_half(result);
        #endif
        return *this;
    }
    
    FORCE_INLINE bool operator<(Float16 rhs) const noexcept {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return native_ < rhs.native_;
        #else
            if (is_nan() || rhs.is_nan()) return false;
            return static_cast<float>(*this) < static_cast<float>(rhs);
        #endif
    }
    
    FORCE_INLINE bool operator>(Float16 rhs) const noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return native_ > rhs.native_;
        #else
            if (is_nan() || rhs.is_nan()) return false;
            return static_cast<float>(*this) > static_cast<float>(rhs);
        #endif
    }
    
    FORCE_INLINE bool operator<=(Float16 rhs) const noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return native_ <= rhs.native_;
        #else
            if (is_nan() || rhs.is_nan()) return false;
            return static_cast<float>(*this) <= static_cast<float>(rhs);
        #endif
    }
    
    FORCE_INLINE bool operator>=(Float16 rhs) const noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return native_ >= rhs.native_;
        #else
            if (is_nan() || rhs.is_nan()) return false;
            return static_cast<float>(*this) >= static_cast<float>(rhs);
        #endif
    }
    
    FORCE_INLINE bool is_inf() const noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return std::isinf(static_cast<float>(native_));
        #else
            // 位模式检测：指数全1 + 尾数全0
            return (bits_ & 0x7FFF) == 0x7C00;
        #endif
    }
    
    FORCE_INLINE bool is_finite() const noexcept
    {
        #if FLOAT16_STANDARD_CXX23 || FLOAT16_ARM_NATIVE || FLOAT16_COMPILER_EXT
            return std::isfinite(static_cast<float>(native_));
        #else
            return (bits_ & 0x7C00) != 0x7C00;
        #endif
    }
    
    friend FORCE_INLINE std::ostream& operator<<(std::ostream& os, Float16 value)
    {
        return os << static_cast<float>(value);
    }

private:
    #if FLOAT16_STANDARD_CXX23
        std::float16_t native_{0};
        
        FORCE_INLINE static uint16_t bit_cast_from_native() noexcept
        {
            return reinterpret_cast<const uint16_t&>(native_);
        }
        
        FORCE_INLINE static std::float16_t bit_cast_to_native(uint16_t bits) noexcep
        {
            return reinterpret_cast<const std::float16_t&>(bits);
        }
        
    #elif FLOAT16_ARM_NATIVE
        __fp16 native_{0};
        
        FORCE_INLINE static uint16_t bit_cast_from_native(__fp16 value) noexcept
        {
            uint16_t result;
            __builtin_memcpy(&result, &value, sizeof(result));
            return result;
        }
        
        FORCE_INLINE static __fp16 bit_cast_to_native(uint16_t bits) noexcept
        {
            __fp16 result;
            __builtin_memcpy(&result, &bits, sizeof(bits));
            return result;
        }
        
    #elif FLOAT16_COMPILER_EXT
        _Float16 native_;
        
        FORCE_INLINE static Float16 bit_cast_from_bits(uint16_t bits) noexcept
        {
            Float16 result;
            __builtin_memcpy(&result.native_, &bits, sizeof(bits));
            return result;
        }
        
        FORCE_INLINE uint16_t bit_cast_to_bits() const noexcept
        {
            uint16_t result;
            __builtin_memcpy(&result, &native_, sizeof(native_));
            return result;
        }
    #else
        uint16_t bits_{0};
        
        FORCE_INLINE static uint16_t float_to_half(float f) noexcept
        {
            uint32_t x;
            std::memcpy(&x, &f, sizeof(f));
    
            const uint32_t sign = (x >> 31) & 0x1;
            uint32_t exponent = (x >> 23) & 0xFF;
            uint32_t mantissa = x & 0x7FFFFF;
    
            if (exponent == 0xFF) return (sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0);
    
            int32_t exp32 = static_cast<int32_t>(exponent) - 127;
            int32_t exp16 = exp32 + 15;
    
            if (exp16 >= 31) return (sign << 15) | 0x7C00;
    
            if (exp16 <= 0)
            {
                if (exp16 < -10) return sign << 15;
        
                mantissa |= 0x800000;
                uint32_t shift = static_cast<uint32_t>(14 - exp16);
                mantissa = (mantissa + (1 << (shift - 1))) >> shift;
                return (sign << 15) | mantissa;
            }
    
            mantissa += 0x1000;
    
            if (mantissa & 0x800000)
            {
                mantissa = 0;
                exp16++;
                if (exp16 >= 31) return (sign << 15) | 0x7C00;
            }
    
            return (sign << 15) | (exp16 << 10) | (mantissa >> 13);
        }
        
        FORCE_INLINE static float half_to_float(uint16_t h) noexcept
        {
            const uint32_t sign = (h >> 15) & 0x1;
            uint32_t exponent = (h >> 10) & 0x1F;
            uint32_t mantissa = h & 0x3FF;
    
            if (exponent == 0x1F)
            {
                uint32_t f = (sign << 31) | 0x7F800000 | (mantissa << 13);
                float result;
                std::memcpy(&result, &f, sizeof(result));
                return result;
            }
    
            if (exponent != 0)
            {
                exponent += 112;
            }
            else if (mantissa != 0)
            {
                exponent = 113;
                while ((mantissa & 0x400) == 0)
                {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3FF;
            }
    
            uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
            float result;
            std::memcpy(&result, &f, sizeof(result));
            return result;
        }
    #endif
};


static_assert(sizeof(Float16) == 2, "Float16 must be 2 bytes");
static_assert(alignof(Float16) <= 2, "Float16 alignment must be <= 2 bytes");
static_assert(std::is_trivially_copyable_v<Float16>, 
              "Float16 must be trivially copyable");
}//MyTensor::Utils
  

namespace std
{
    template <>
    struct numeric_limits<MyTensor::Utils::Float16>
    {
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr float_denorm_style has_denorm = denorm_present;
        static constexpr bool has_denorm_loss = false;
        
        static constexpr float_round_style round_style = round_to_nearest;
        static constexpr bool is_iec559 = true;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        
        static constexpr int digits = 11;
        static constexpr int digits10 = 3;
        static constexpr int max_digits10 = 5;
        
        static constexpr int radix = 2;
        
        static constexpr int min_exponent = -13;
        static constexpr int min_exponent10 = -4;
        static constexpr int max_exponent = 16;
        static constexpr int max_exponent10 = 4;
        
        /*static constexpr MyTensor::Utils::Float16 min() noexcept
        { 
            return MyTensor::Utils::Float16(0x0400);
        }
        static constexpr MyTensor::Utils::Float16 max() noexcept
        { 
            return MyTensor::Utils::Float16(0x7BFF);
        }
        static constexpr MyTensor::Utils::Float16 lowest() noexcept
        { 
            return MyTensor::Utils::Float16(0xFBFF);
        }
        static constexpr MyTensor::Utils::Float16 epsilon() noexcept
        { 
            return MyTensor::Utils::Float16(0x1400);
        }
        static constexpr MyTensor::Utils::Float16 round_error() noexcept
        { 
            return MyTensor::Utils::Float16(0x3800);
        }
        static constexpr MyTensor::Utils::Float16 infinity() noexcept
        { 
            return MyTensor::Utils::Float16(0x7C00);
        }
        static constexpr MyTensor::Utils::Float16 quiet_NaN() noexcept
        { 
            return MyTensor::Utils::Float16(0x7E00);
        }
        static constexpr MyTensor::Utils::Float16 signaling_NaN() noexcept
        { 
            return MyTensor::Utils::Float16(0x7D00);
        }
        static constexpr MyTensor::Utils::Float16 denorm_min() noexcept
        { 
            return MyTensor::Utils::Float16(0x0001);
        }*/
    
        // 使用类型双关直接创建Float16对象
        static constexpr MyTensor::Utils::Float16 min() noexcept
        { 
            return bit_cast_to_float16(0x0400);
        }
        
        static constexpr MyTensor::Utils::Float16 max() noexcept
        { 
            return bit_cast_to_float16(0x7BFF);
        }
        
        static constexpr MyTensor::Utils::Float16 lowest() noexcept
        { 
            return bit_cast_to_float16(0xFBFF);
        }
        
        static constexpr MyTensor::Utils::Float16 epsilon() noexcept
        { 
            return bit_cast_to_float16(0x1400);
        }
        
        static constexpr MyTensor::Utils::Float16 round_error() noexcept
        { 
            return bit_cast_to_float16(0x3800);
        }
        
        static constexpr MyTensor::Utils::Float16 infinity() noexcept
        { 
            return bit_cast_to_float16(0x7C00);
        }
        
        static constexpr MyTensor::Utils::Float16 quiet_NaN() noexcept
        { 
            return bit_cast_to_float16(0x7E00);
        }
        
        static constexpr MyTensor::Utils::Float16 signaling_NaN() noexcept
        { 
            return bit_cast_to_float16(0x7D00);
        }
        
        static constexpr MyTensor::Utils::Float16 denorm_min() noexcept
        { 
            return bit_cast_to_float16(0x0001);
        }
        
    private:
        static constexpr MyTensor::Utils::Float16 bit_cast_to_float16(uint16_t raw) noexcept
        {
            static_assert(sizeof(MyTensor::Utils::Float16) == sizeof(raw), 
                         "Size mismatch for type punning");
            
            #ifdef __clang__
                union
                {
                    uint16_t as_uint;
                    MyTensor::Utils::Float16 as_float16;
                } pun = { .as_uint = raw };
                return pun.as_float16;
            
            #elif __cplusplus >= 202002L
                MyTensor::Utils::Float16 result;
                std::memcpy(&result, &raw, sizeof(raw));
                return result;
            
            #else
                /*struct Storage
                {
                    unsigned char data[sizeof(MyTensor::Utils::Float16)];
                };
                
                Storage from = { { *(reinterpret_cast<unsigned char(*)[sizeof(raw)]>(&raw)) } };
                Storage to{};
                for (size_t i = 0; i < sizeof(from); ++i)
                {
                    to.data[i] = from.data[i];
                }
                return *reinterpret_cast<MyTensor::Utils::Float16*>(&to);*/

                union
                {
                    uint16_t as_uint;
                    MyTensor::Utils::Float16 as_float16;
                } pun = { raw };

                //pun.as_uint = raw;
                return pun.as_float16;

            #endif
        }
    };
}

   
#endif
