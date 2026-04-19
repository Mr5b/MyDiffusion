#ifndef MY_MODULE_GROUPNORM_H
#define MY_MODULE_GROUPNORM_H

#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{

using namespace MNN::Express;

class GroupNorm : public MyModule
{
public:
    GroupNorm
    (
        int num_groups,
        int num_channels,
        float eps = 1e-05,
        bool affine = true,
        Dimensionformat format = Dimensionformat::NCHW,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        num_groups_(num_groups),
        eps_(eps),
        affine_(affine)
    {
        MY_ASSERT
        (
            num_channels % num_groups_ == 0,
            std::string("num_channels must be divisible by num_groups. num_channels: "
            + std::to_string(num_channels)
            + " num_groups_: "
            + std::to_string(num_groups_)).c_str()
        );
        
        if (affine)
        {
            INTS shape = {num_channels};
            /*if (format == Dimensionformat::NHWC)
            {
                shape = {num_channels};
            }
            else
            {
                shape = {1, num_channels, 1, 1};
            }*/
            weight_ = fillValue(1, shape, format, dtype);
            bias_ = fillValue(0, shape, format, dtype);
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        
        INTS output_shape = x->getInfo()->dim;
        MY_ASSERT(output_shape.size() == 4, "");      

        const MNN::Tensor* x_tensor = x->getTensor();
        int N = x_tensor->batch();
        int C = x_tensor->channel();
        int H = x_tensor->height();
        int W = x_tensor->width();
        int group_channels = C / num_groups_;
        INTS shape_5d = {N, num_groups_, group_channels};
        INTS reduce_dims;
    
        Dimensionformat format = x->getInfo()->order;
    
        MY_ASSERT(C % num_groups_ == 0, "");
    
        if (format == Dimensionformat::NHWC)
        {
            shape_5d.insert(shape_5d.begin() + 1, {H, W});
            reduce_dims = {1, 2, 4};
        }
        else
        {
            shape_5d.insert(shape_5d.end(), {H, W});
            reduce_dims = {2, 3, 4};
        }
    
        VARP x_5d = _Reshape(x, shape_5d, format);
        
        VARP mean = _ReduceMean(x_5d, reduce_dims, true);
        
        VARP centerd = x_5d - mean;
        VARP var = _ReduceMean(_Square(centerd), reduce_dims, true);
        
        VARP normalized = centerd / _Sqrt(_BiasAdd(var, _Scalar(eps_)));
            
        VARP normalized_4d = _Reshape(normalized, output_shape, format);
        
        if (!affine_)
        {
            return {normalized_4d};
        }
        
        MY_ASSERT(format == weight_->getInfo()->order, "");
        MY_ASSERT
        (
            (
                weight_->getTensor()->elementSize() == C ||
                weight_->getTensor()->elementSize() == 1
            ),
            ""
        );
        
        /*weight_.fix(VARP::TRAINABLE);
        bias_.fix(VARP::TRAINABLE);*/
            
        VARP out = _BiasAdd((normalized_4d * _Unsqueeze(weight_, {1, 2})), _Unsqueeze(bias_, {1, 2}));
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
    
    int num_groups_;
    bool affine_;
    float eps_;
    VARP weight_, bias_;
};

}//namespace DonNotKnowHowToNameIt

#endif