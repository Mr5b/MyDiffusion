#ifndef MYMODULE_ASSERT_H
#define MYMODULE_ASSERT_H

#include <stdexcept>
#include <string>

namespace DonNotKnowHowToNameIt
{

#ifndef MY_MODULE_NO_DEBUG
#ifdef NDEBUG
#define MY_MODULE_NO_DEBUG
#endif
#endif

#ifdef MY_MODULE_NO_DEBUG
    #define MY_ASSERT(cond, msg)
#else
    #define MY_ASSERT(cond, msg)                                            \
    do                                                                  \
    {                                                                   \
        if (!(cond))                                                    \
        {                                                               \
            fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", \
                    (msg), __FILE__, __LINE__);                         \
            abort();                                                    \
        }                                                               \
    }                                                                   \
    while(0)
#endif
}//namespace DonNotKnowHowToNameIt


#endif