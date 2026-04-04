#ifndef MY_MODULE_LINEAR
#define MY_MODULE_LINEAR

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{
using namespace MNN::Express;

class Linear : public MyModule
{
public:
    Linear
    (
        int in_features, int out_features, bool bias = true,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        weight_(fillValue(0, {out_features, in_features}, Dimensionformat::NCHW, dtype)),
        bias_(bias ? fillValue(0, {out_features}, Dimensionformat::NCHW, dtype) : nullptr),
        has_bias_(bias)
    {
        register_parameter("weight", weight_);
        if (bias) register_parameter("bias", bias_);
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        weight_.fix(VARP::TRAINABLE);
        if (bias_.get()) bias_.fix(VARP::TRAINABLE);
    
        VARP x = inputs[0];
        auto input_info = x->getInfo();
        const INTS& input_shape = input_info->dim;
        const INTS& weight_shape = weight_->getInfo()->dim;
        INTS flat;
        int in_features = weight_shape[1];
        
        MY_ASSERT(input_shape.back() == in_features, "");
        
        int batch_size = 1;
        for (int i = 0, n = input_shape.size() - 1; i < n; i++)
        {
            batch_size *= input_shape[i];
        }
        flat = {batch_size, in_features};
        
        VARP x_flat = _Reshape(x, flat, input_info->order);
        VARP y = _MatMul(x_flat, weight_, false, true);
        //y = _Add(y, bias_);
        if (has_bias_) y = _BiasAdd(y, bias_);
        
        INTS output_shape = input_shape;
        output_shape[output_shape.size() - 1] = weight_shape[0];
        y = _Reshape(y, output_shape);
        return {y};
    }

    VARP weight_, bias_;
    bool has_bias_;
};

}//namespace DonNotKnowHowToNameIt

#endif