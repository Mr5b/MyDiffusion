#ifndef MY_MODULE_LAYERNORM_H
#define MY_MODULE_LAYERNORM_H

#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{

using namespace MNN::Express;

class LayerNorm : public MyModule
{
public:

    LayerNorm
    (
        const INTS& normalized_shape,
        halide_type_t dtype = halide_type_of<float>(),
        float eps = 1e-05,
        bool affine = true
    ) :
        normalized_shape_(normalized_shape),
        eps_(eps),
        affine_(affine)
    {
        if (affine)
        {
            weight_ = fillValue(1, normalized_shape, NCHW, dtype);
            bias_ = fillValue(0, normalized_shape, NCHW, dtype);
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        
        INTS output_shape = x->getInfo()->dim;
        
        int num_norm_dims = normalized_shape_.size();
        INTS reduce_dims;
        
        {
            int x_rank = output_shape.size();
            reduce_dims.reserve(num_norm_dims);
            for (int i = x_rank - num_norm_dims; i < x_rank; i++)
            {
                reduce_dims.emplace_back(i);
            }
        }
        
        VARP mean = _ReduceMean(x, reduce_dims, true);
        
        VARP centerd = x - mean;
        VARP var = _ReduceMean(_Square(centerd), reduce_dims, true);
        
        VARP normalized = centerd / _Sqrt(_BiasAdd(var, _Scalar(eps_)));
        
        if (!affine_)
        {
            return {normalized};
        }
        
        weight_.fix(VARP::TRAINABLE);
        bias_.fix(VARP::TRAINABLE);
            
        VARP out = _BiasAdd((normalized * weight_), bias_);
        return {out};
    }
    
    /*virtual void load_from_safetensors
    (
        const SafetensorLoader& loader,
        const std::string& prefix = "",
        SafetensorLoader::ShapeMode shape_mode = SafetensorLoader::ShapeMode::LOOSE,
        DtypePolicy dtype_policy = DtypePolicy::AS_FILE,
        bool allow_missing_tensors = false
    ) override
    {
        return
        load_from_safetensors_recursive
        (
            loader,
            prefix,
            SafetensorLoader::ShapeMode::LOOSE,
            dtype_policy,
            allow_missing_tensors
        );
    }*/
    
    INTS normalized_shape_;
    bool affine_;
    float eps_;
    VARP weight_, bias_;
};

}//namespace DonNotKnowHowToNameIt

#endif