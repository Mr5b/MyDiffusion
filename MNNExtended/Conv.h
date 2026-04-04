#ifndef MY_MODULE_CONV
#define MY_MODULE_CONV

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{
template <size_t D>
std::array<int, D> make_filled_array(int value)
{
    std::array<int, D> arr;
    arr.fill(value);
    return arr;
}



template <size_t D>
struct ConvNdOptions
{
    int in_channels;
    int out_channels;
    std::array<int, D> kernel_size_;
    std::array<int, D> stride_;
    std::array<int, D> padding_;
    std::array<int, D> dilation_;
    int groups_ = 1;
    bool bias_ = true;
    PaddingMode padding_mode_ = PaddingMode::CAFFE;

    ConvNdOptions
    (
        int in_channels, 
        int out_channels, 
        std::array<int, D> kernel_size
    ) : in_channels(in_channels),
        out_channels(out_channels),
        kernel_size_(kernel_size),
        stride_(make_filled_array<D>(1)),
        padding_(make_filled_array<D>(0)),
        dilation_(make_filled_array<D>(1))
        {}

    ConvNdOptions& stride(std::array<int, D> stride)
    {
        stride_ = stride;
        return *this;
    }
    
    ConvNdOptions& padding(std::array<int, D> padding)
    {
        padding_ = padding;
        return *this;
    }
    
    ConvNdOptions& dilation(std::array<int, D> dilation)
    {
        dilation_ = dilation;
        return *this;
    }
    
    ConvNdOptions& bias(bool bias)
    {
        bias_ = bias;
        return *this;
    }
};

template <size_t D, typename Derived>
class ConvNd : public MyModule
{
public:
    explicit ConvNd
    (
        ConvNdOptions<D> options,
        Dimensionformat format = NCHW,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        options_(std::move(options))
    {
        int out_c = options_.out_channels;
        std::vector<int> weight_dims =
            {out_c, (options_.in_channels / options_.groups_)};
            
        weight_dims.reserve(2 + D);
        
        if (format == NHWC)
        {
            weight_dims.insert
            (
                weight_dims.begin() + 1,
                options.kernel_size_.begin(),
                options.kernel_size_.end()
            );
        }
        else
        {
            weight_dims.insert
            (
                weight_dims.begin() + 2,
                options.kernel_size_.begin(),
                options.kernel_size_.end()
            );
        }
        
        weight_ = fillValue(0, weight_dims, format, dtype);
        register_parameter("weight", weight_);
        
        if (options_.bias_)
        {
            bias_ = fillValue(0, {out_c}, format, dtype);
            register_parameter("bias", bias_);
        }
        else
        {
            bias_ = createScalar(0, format, dtype);
        }
    }
    
    VARP weight_, bias_;
    ConvNdOptions<D> options_;
};

class Conv2d : public ConvNd<2, Conv2d>
{
public:
    explicit Conv2d
    (
        ConvNdOptions<2> options,
        Dimensionformat format = NCHW,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        ConvNd<2, Conv2d>(std::move(options), format, dtype)
    {}
    
    /*explicit Conv2d
    (
        int input_channels,
        int output_channels,
        std::array<int, 2> kernel_size
    ) :
    Conv2dImpl
    (
            ConvNdOptions<2>(input_channels, output_channels, kernel_size)
    ) {}*/
    
    virtual std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs) override
    {
        weight_.fix(VARP::TRAINABLE);
        bias_.fix(VARP::TRAINABLE);
    
        auto x = inputs[0];
        MY_ASSERT(x->getInfo()->dim.size() == 4, "");
        MY_ASSERT(x->getTensor()->channel() == weight_->getTensor()->channel() * options_.groups_, "");
        
        x =
            _Conv
            (
                weight_,
                bias_,
                x,
                options_.padding_mode_,
                {options_.stride_[0], options_.stride_[1]},
                {options_.dilation_[0], options_.dilation_[1]},
                options_.groups_,
                {options_.padding_[0], options_.padding_[1]}
            );
        
        return {x};
    }
};

}//namespace DonNotNowHowToNameIt

#endif