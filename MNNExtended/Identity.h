#ifndef MY_MODULE_IDENTITY_H
#define MY_MODULE_IDENTITY_H

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{

using namespace MNN::Express;

class Identity : public MyModule
{
public:
    Identity() = default;

    virtual std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs) override
    {
        return inputs;
    }
};

}//namespace DonNotKnowHowToNameIt

#endif