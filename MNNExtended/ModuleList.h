#ifndef MY_MODULE_MODULELIST_H
#define MY_MODULE_MODULELIST_H

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{

using namespace MNN::Express;

class ModuleList : public MyModule
{
public:
    using ModulePtr = std::shared_ptr<Module>;
    using Container = std::vector<std::shared_ptr<Module>>;
    
    using iterator = Container::iterator;
    using const_iterator = Container::const_iterator;
    
    const_iterator begin() const { return mChildren.begin(); }
    const_iterator end() const { return mChildren.end(); }
    const_iterator cbegin() const { return mChildren.cbegin(); }
    const_iterator cend()   const { return mChildren.cend(); }

    ModuleList(std::vector<std::shared_ptr<Module>> children)
    {
        for (int idx = 0, n = children.size(); idx < n; idx++)
        {
            children[idx]->setName(std::to_string(idx));
        }
        mChildren = children;
    }
    
    ModuleList() = default;
    
    void emplace_back(std::shared_ptr<Module> module)
    {
        module->setName(std::to_string(mChildren.size()));
        mChildren.emplace_back(module);
    }
    
    iterator insert(const_iterator pos, const ModulePtr& value)
    {
        iterator result = mChildren.insert(pos, value);
        for (int idx = 0, n = mChildren.size(); idx < n; idx++)
        {
            mChildren[idx]->setName(std::to_string(idx));
        }
        return result;
    }
    
    void reserve(size_t size)
    {
        mChildren.reserve(size);
    }
    
    virtual std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs) override
    {
        return inputs;
    }
};

}//namespace DonNotKnowHowToNameIt

#endif