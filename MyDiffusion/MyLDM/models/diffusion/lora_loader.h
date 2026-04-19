#ifndef MYDIFFUSION_LORA_LOADER
#define MYDIFFUSION_LORA_LOADER

#include "lora.h"
#include <MyModule.h>

namespace IDK = DonNotKnowHowToNameIt;
using namespace MNN::Express;

std::vector<std::shared_ptr<MyTensor::Utils::FileMapping>> lora_load
(
    std::unordered_map<std::string, MNN::Express::VARP> parameters,
    IDK::SafetensorLoader loader,
    bool blocking = true
)
{
    std::vector<std::shared_ptr<MyTensor::Utils::FileMapping>> mapping;
    
    /*auto sf_metadata = loader.get_safetensors_metadata();
    auto global_metadata = sf_metadata->get_global_metadata();*/
    
    /*float scale = std::stoi(sf_metadata["ss_network_alpha"]） / std::stoi(sf_metadata["ss_network_dim"];
    std::cout << "scale: " << scale << std::endl;*/
    /*std::cout << "ss_network_alpha: " << sf_metadata["ss_network_alpha"] << std::endl;
    std::cout << "ss_network_dim: " << sf_metadata["ss_network_dim"] << std::endl;*/
    
    for (auto param_pair : parameters)
    {
        VARP param_var = param_pair.second;
        std::string param_name = param_var->name();
        auto param_info = param_var->getInfo();
        LoRAKeySet keys = convertSD15KeyToLoRAKeys(param_pair.first);
        if (loader.contains(keys.up))
        {
            using PAIR =
                std::pair
                <
                    MNN::Express::EXPRP,
                    std::shared_ptr<MyTensor::Utils::FileMapping>
                >;
                
            PAIR up_PAIR = loader.get_tensor(keys.up, param_info->order);
                    
            PAIR down_PAIR = loader.get_tensor(keys.down, param_info->order);
            
            VARP up = Variable::create(up_PAIR.first);
            VARP down = Variable::create(down_PAIR.first);
            
            VARP delta;
            
            bool isConv = up->getInfo()->dim.size() == 4;
            
            if (isConv)
            {
                up = _Squeeze(up, {2, 3});
                down = _Squeeze(down, {2, 3});
            }
            else
            {
                if (up->getInfo()->dim.size() != 2) throw std::runtime_error("乱来" + param_name);
            }
            delta = _MatMul(up, down);
            
            if (isConv) delta = _Unsqueeze(delta, {2, 3});
            
            
            if (loader.contains(keys.alpha))
            {
                PAIR alpha_PAIR = loader.get_tensor(keys.alpha, param_info->order);
                
                VARP alpha = Variable::create(alpha_PAIR.first);
                VARP rank = _Scalar<float>(float(down->getInfo()->dim[0]));
                
                delta = delta * (alpha / rank);
                
                mapping.emplace_back(alpha_PAIR.second);
            }
            
            mapping.emplace_back(up_PAIR.second);
            mapping.emplace_back(down_PAIR.second);
            
            VARP new_w = param_var + delta;
            MY_ASSERT(new_w.get() != nullptr && new_w->getInfo() != nullptr, "nullptr!!!!!");
            
            if (blocking) new_w.fix(VARP::CONSTANT);
            
            //auto old_expr_pair = ;
            param_var->setExpr(new_w->expr().first, param_var->expr().second);
            param_var->setName(param_name);
            
            std::cout << "lora: " << param_pair.first << std::endl;
        }
    }
    return mapping;
}


#endif