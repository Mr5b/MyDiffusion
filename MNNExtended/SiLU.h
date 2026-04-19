#ifndef MY_MODULE_SILU
#define MY_MODULE_SILU

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{

class SiLU : public MyModule
{
public:
    
    virtual std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs) override
    {
        VARPS outputs;
        outputs.reserve(inputs.size());
        for (auto x : inputs) outputs.emplace_back(_Silu(x));
        return outputs;
    }
};

}//namespace DonNotKnowHowToNameIt

#endif