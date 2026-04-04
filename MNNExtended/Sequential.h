#ifndef MY_MODULE_SEQUETIAL_H
#define MY_MODULE_SEQUETIAL_H

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <iterator>
#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{
using namespace MNN::Express;

class Sequential : public MyModule
{
public:
    Sequential(std::vector<std::shared_ptr<Module>> children)
    {
        for (int idx = 0, n = children.size(); idx < n; idx++)
        {
            children[idx]->setName(std::to_string(idx));
        }
        mChildren = children;
    }
    
    Sequential(std::vector<std::pair<std::string, std::shared_ptr<Module>>> children)
    {
        for (auto module : children)
        {
            module.second->setName(module.first);
        }
        mChildren.clear();
        std::transform
        (
            children.begin(),
            children.end(),
            std::back_inserter(mChildren),
            [](const auto& pair) { return pair.second; }
        );
    }
    
    /*virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        std::vector<VARP> x = inputs;
        for (auto m : mChildren)
        {
            x = m->onForward(x);
        }
        return x;
    }*/
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        auto it = mChildren.begin();
        if (it == mChildren.end())
        {
            return std::vector<VARP>(inputs);
        }

        std::vector<VARP> x = (*it)->onForward(inputs);

        for (++it; it != mChildren.end(); ++it)
        {
            x = (*it)->onForward(x);
        }
        return x;
    }
};

}//namespace DonNotKnowHowToNameIt

#endif