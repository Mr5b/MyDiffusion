#include <cstdint>
#include <climits>
#include <cfloat>
#include <MNN/HalideRuntime.h>

namespace DonNotKnowHowToNameIt
{

double halide_type_max(const halide_type_t& type)
{
    switch (type.code)
    {
        case 0:
        {
            if (type.bits <= 0 || type.bits > 64) return 0.0;
            if (type.bits == 64) return 9223372036854775807.0;
            return (double)((1ULL << (type.bits - 1)) - 1);
        }
        case 1:
        {
            if (type.bits <= 0 || type.bits > 64) return 0.0;
            if (type.bits == 64) return (double)UINT64_MAX;
            return (double)((1ULL << type.bits) - 1);
        }
        case 2:
        {
            if (type.bits == 16) return 65504.0;
            if (type.bits == 32) return (double)FLT_MAX;
            if (type.bits == 64) return DBL_MAX;
            return 0.0;
        }
        case 4:
        {
            return 3.38953139e38;
        }
        case 3:
        {
            return (double)UINTPTR_MAX;
        }
        default:
            return 0.0;
    }
}


double halide_type_min(const halide_type_t& type)
{
    switch (type.code)
    {
        case 0:
        {
            if (type.bits <= 0 || type.bits > 64) return 0.0;
            if (type.bits == 64) return -9223372036854775807.0 - 1.0;
            return -(double)((1ULL << (type.bits - 1)));
        }
        case 1:
            return 0.0;
        case 2:
        {
            if (type.bits == 16) return -65504.0;
            if (type.bits == 32) return -(double)FLT_MAX;
            if (type.bits == 64) return -DBL_MAX;
            return 0.0;
        }
        case 4:
            return -3.38953139e38;
        case 3:
            return 0.0;
        default:
            return 0.0;
    }
}

}//namespace DonNotKnowHowToNameIt