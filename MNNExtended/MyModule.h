#ifndef MY_MODULE_H
#define MY_MODULE_H

#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include "file_mapping.h"
#include "MNNSafetensorLoader.h"
#include "Scalar.h"

/*namespace MNN
{
namespace Express
{


struct Expr::Inside
{
    Inside(int outputSize);
    Inside(Tensor* tensor, bool own = false);
    ~ Inside();
    std::vector<Variable::Info> mOutputInfos;
    std::vector<Tensor*> mOutputTensors;
    Executor::Requirement mReq;
    std::shared_ptr<Executor::ComputeCache> mCache;
    int mCacheOffset = 0;
    bool mInfoDirty = true;
    bool mContentDirty = true;
    bool mOwnTensor = true;
    Tensor* mHostTensor = nullptr;
    std::shared_ptr<Backend> mHoldBackend;
};

}//namespace Express
}//MNN
*/
namespace DonNotKnowHowToNameIt
{

void print_shape(MNN::Express::VARP x)
{
    const INTS& dims = x->getInfo()->dim;
    for (auto x : dims) std::cout << x << " ";
    std::cout << std::endl;
}

void print_stride(MNN::Express::VARP x)
{
    auto dim_size = x->getInfo()->dim.size();
    auto x_tensox = x->getTensor();
    for (int i = 0; i < dim_size; i++) std::cout << x_tensox->stride(i) << " ";
    std::cout << std::endl;
}

class MyModule : public MNN::Express::Module
{
public:
    enum class DtypePolicy
    {
        STRICT,
        AS_DEFINITION,
        AS_FILE
    };

    int register_parameter
    (
        const std::string& name,
        MNN::Express::VARP tensor
    )
    {
        tensor->setName(name);
        return addParameter(tensor);     
    }
    
    void register_module
    (
        const std::string& name,
        const std::shared_ptr<Module>& children
    )
    {
        children->setName(name);
        //return registerModel({children});
        mChildren.emplace_back(children);
    }
    
    virtual void load_from_safetensors
    (
        const SafetensorLoader& loader,
        const std::string& prefix = "",
        SafetensorLoader::ShapeMode shape_mode = SafetensorLoader::ShapeMode::LOOSE,
        DtypePolicy dtype_policy = DtypePolicy::AS_FILE,
        bool allow_missing_tensors = false
    )
    {
        return
        load_from_safetensors_recursive
        (
            loader,
            prefix,
            shape_mode,
            dtype_policy,
            allow_missing_tensors
        );
    }
    
    void load_from_safetensors_recursive
    (
        const SafetensorLoader& loader,
        const std::string& prefix = "",
        SafetensorLoader::ShapeMode shape_mode = SafetensorLoader::ShapeMode::LOOSE,
        DtypePolicy dtype_policy = DtypePolicy::AS_FILE,
        bool allow_missing_tensors = false
    )
    {
        for (int i = 0, n = mParameters.size(); i < n; i++)
        {
            auto param = mParameters[i];
            std::string param_name = param->name();
            const std::string full_key = prefix + param->name();//std::cout << full_key << std::endl;
            
            if (loader.contains(full_key))
            {
                Dimensionformat format = param->getInfo()->order;
                
                std::pair
                <
                    MNN::Express::EXPRP,
                    std::shared_ptr<MyTensor::Utils::FileMapping>
                > PAIR =
                    loader.get_tensor(full_key, format, param->getInfo()->dim, shape_mode);
                
                /*auto inside      = PAIR.first->inside();
                auto mTensor = inside->mOutputTensors[0];
                if (nullptr != inside->mCache)
                {
                    mTensor =
                        inside->mCache->getSession()->getTensor(inside->mCacheOffset);
                }*/
                
                VARP temp = Variable::create(PAIR.first);
                
                auto mTensor = temp->getTensor();
                
                if (mTensor->getType() != param->getTensor()->getType())
                {
                    if (dtype_policy == DtypePolicy::STRICT)
                    {
                        throw std::runtime_error("Data type mismatch: " + full_key);
                    }
                    else if (dtype_policy == DtypePolicy::AS_DEFINITION)
                    {
                        auto info = *(param->getInfo());
                        /*MNN::Express::EXPRP new_expr = Expr::create(std::move(info), PAIR.second->data(), VARP::CONSTANT, Expr::MemoryType::COPY);*/
                        MNN::Express::VARP d = MNN::Express::_Cast(temp, info.type);
                        auto old_expr_pair = param->expr();
                        param->setExpr(d->expr().first, old_expr_pair.second);
                        
                    }
                    else
                    {
                        auto old_expr_pair = param->expr();
                        param->setExpr(PAIR.first, old_expr_pair.second);
                    }
                }
                else
                {
                    auto old_expr_pair = param->expr();
                    param->setExpr(PAIR.first, old_expr_pair.second);
                }
                
                mappings_[i] = PAIR.second;
                param->setName(param_name);
            }
            else
            {
                if (allow_missing_tensors)
                {
                    std::cout << "Warning: Missing parameter: " << full_key << std::endl;
                }
                else
                {
                    throw std::runtime_error("Missing parameter: " + full_key);
                }
            }
        }
    
        for (auto module : mChildren)
        {
            const std::string child_prefix = prefix + module->name() + ".";
            reinterpret_cast<MyModule*>(module.get())->load_from_safetensors(loader, child_prefix, shape_mode, dtype_policy, allow_missing_tensors);
        }
    }
    
    std::unordered_map<std::string, MNN::Express::VARP> get_parameters_recursive()
    {
        std::unordered_map<std::string, MNN::Express::VARP> map;
        get_parameters_recursive(map, "");
        return map;
    }
    
    
    virtual void get_parameters_recursive
    (
        std::unordered_map<std::string, MNN::Express::VARP>& map,
        const std::string& prefix = ""
    )
    {
        for (auto t : mParameters)
        {
            map[prefix + t->name()] = t;
        }
        for (auto m : mChildren)
        {
            const std::string child_prefix = prefix + m->name() + ".";
            static_cast<MyModule*>(m.get())->get_parameters_recursive(map, child_prefix);
        }
    }
    
    virtual std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs) override
    {
        return inputs;
    }
    
    std::shared_ptr<Module> getChildren(size_t idx)
    {
        return mChildren[idx];
    }
    
    
protected:
    std::unordered_map<int, std::shared_ptr<MyTensor::Utils::FileMapping>> mappings_;
};


}//namespace DonNotKnowHowToNameIt
#endif