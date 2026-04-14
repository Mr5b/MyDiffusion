#ifndef MYLDM_DIFFUSIONMODULES_MODEL
#define MYLDM_DIFFUSIONMODULES_MODEL

#include <cmath>
#include <MyModules.h>
#include "../attention.h"
#include "util.h"
#include <variant>

namespace MyLDM
{
namespace OpenaiModel
{

namespace IDK = DonNotKnowHowToNameIt;

struct Upsample : public IDK::MyModule
{
    std::shared_ptr<IDK::MyModule> conv_{nullptr};
    
    Upsample
    (
        int channels,
        int out_channels = -1,
        int padding = 1,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        conv_
        (
            std::make_shared<IDK::Conv2d>
            (
                IDK::ConvNdOptions<2>(channels, out_channels, {3, 3})
                    .padding({padding, padding}),
                Dimensionformat::NCHW,
                dtype
            )
        )
    {
        register_module("conv", conv_);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        x = _Interp({x}, 2, 2, 0, 0, MNN::Express::InterpolationMethod::NEAREST, false);
        x = conv_->forward(x);
        return {x};
    }
};

struct Downsample : public IDK::MyModule
{
public:
    std::shared_ptr<IDK::MyModule> op_{nullptr};
    
    Downsample
    (
        int channels,
        int out_channels = -1,
        int padding = 1,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        op_
        (
            std::make_shared<IDK::Conv2d>
            (
                IDK::ConvNdOptions<2>(channels, out_channels, {3, 3})
                    .stride({2, 2})
                    .padding({padding, padding}),
                Dimensionformat::NCHW,
                dtype
            )
        )
    {
        register_module("op", op_);
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
            
        x = op_->forward(x);
        
        return {x};
    }
};


struct GuessWhoIAm : public IDK::MyModule
{
    enum struct Type : char
    {
        TimestepBlock,
        SpatialTransformer,
        Nothing
    };

    std::shared_ptr<IDK::MyModule> module_{nullptr};
    Type type = Type::Nothing;
    
    
    /*template <class TypeOfModule, class... Args>
    GuessWhoIAm(Args&&... args) :
        module_(std::make_shared<TypeOfModule>(std::forward<Args>(args)...))
    {}*/

    template <typename TypeOfModule, class... Args>
    GuessWhoIAm(std::in_place_type_t<TypeOfModule>, Args&&... args) :
        module_(std::make_shared<TypeOfModule>(std::forward<Args>(args)...))
    {}
        
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        return module_->onForward(inputs);
    }
    
    virtual void load_from_safetensors
    (
        const SafetensorLoader& loader,
        const std::string& prefix = "",
        SafetensorLoader::ShapeMode shape_mode = SafetensorLoader::ShapeMode::LOOSE,
        DtypePolicy dtype_policy = DtypePolicy::AS_FILE,
        bool allow_missing_tensors = false
    ) override
    {
        module_->load_from_safetensors
        (
            loader,
            prefix,
            shape_mode,
            dtype_policy,
            allow_missing_tensors
        );
    }
};

template<class TypeOfModule, class... Args>
std::shared_ptr<GuessWhoIAm> makeGuessWhoIAm(Args&&... args)
{
    return std::make_shared<GuessWhoIAm>(std::in_place_type<TypeOfModule>, std::forward<Args>(args)...);
}

std::shared_ptr<GuessWhoIAm> makeSpatialTransformer
(
    int in_channels,
    int n_heads,
    int d_head,
    int depth = 1,
    int context_dim = -1,
    halide_type_t dtype = halide_type_of<float>()
)
{
    std::shared_ptr<GuessWhoIAm> r =
        makeGuessWhoIAm<SpatialTransformer>
        (
            in_channels,
            n_heads,
            d_head,
            depth,
            context_dim,
            dtype
        );
    
    r->type = GuessWhoIAm::Type::SpatialTransformer;
    return r;
}

template<class TypeOfModule, class... Args>
std::shared_ptr<GuessWhoIAm> makeTimestepBlock(Args&&... args)
{
    std::shared_ptr<GuessWhoIAm> r =
        makeGuessWhoIAm<TypeOfModule>(std::forward<Args>(args)...);
    r->type = GuessWhoIAm::Type::TimestepBlock;
    return r;
}

struct TimestepEmbedSequential : public IDK::Sequential
{
    TimestepEmbedSequential(std::vector<std::shared_ptr<Module>> children) :
        IDK::Sequential(std::move(children)) {}
        
    TimestepEmbedSequential(std::vector<std::pair<std::string, std::shared_ptr<Module>>> children) :
        IDK::Sequential(std::move(children)) {}
        
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        VARP emb = inputs[1];
        VARP context = inputs[2];
        
        for (auto m : mChildren)
        {
            GuessWhoIAm* layer = reinterpret_cast<GuessWhoIAm*>(m.get());
            if (layer->type == GuessWhoIAm::Type::TimestepBlock)
            {
                x = layer->onForward({x, emb})[0];
            }
            else if (layer->type == GuessWhoIAm::Type::SpatialTransformer)
            {
                x = layer->onForward({x, context})[0];
            }
            else
            {
                x = layer->forward(x);
            }
        }
        return {x};
    }
};
    /*def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x*/

struct ResBlock : public IDK::MyModule
{
public:
    int channels_, out_channels_;
    
    std::shared_ptr<IDK::MyModule> in_layers_{nullptr};
    std::shared_ptr<IDK::MyModule> emb_layers_{nullptr};
    std::shared_ptr<IDK::MyModule> out_layers_{nullptr};
    std::shared_ptr<IDK::MyModule> skip_connection_{nullptr};

    ResBlock
    (
        int channels,
        int emb_channels,
        int out_channels = -1,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        channels_(channels),
        out_channels_((out_channels < 0) ? channels : out_channels),
        in_layers_
        (
            std::make_shared<IDK::Sequential>
            (
                std::vector<std::shared_ptr<Module>>
                {
                    std::make_shared<IDK::GroupNorm>
                    (
                        32, channels, 1e-05, true, Dimensionformat::NCHW, dtype
                    ),
                    std::make_shared<IDK::SiLU>(),
                    std::make_shared<IDK::Conv2d>
                    (
                        IDK::ConvNdOptions<2>(channels_, out_channels_, {3, 3})
                            .padding({1, 1}),
                        Dimensionformat::NCHW,
                        dtype
                    )
                }
            )
        ),
        emb_layers_
        (
            std::make_shared<IDK::Sequential>
            (
                std::vector<std::shared_ptr<Module>>
                {
                    std::make_shared<IDK::SiLU>(),
                    std::make_shared<IDK::Linear>(emb_channels, out_channels_, true, dtype)
                }
            )
        ),
        out_layers_
        (
            std::make_shared<IDK::Sequential>
            (
                std::vector<std::pair<std::string, std::shared_ptr<Module>>>
                {
                    std::make_pair<std::string, std::shared_ptr<Module>>
                    (
                        "0",
                        std::make_shared<IDK::GroupNorm>
                        (
                            32, out_channels_, 1e-05, true, Dimensionformat::NCHW, dtype
                        )
                    ),
                    std::make_pair<std::string, std::shared_ptr<Module>>
                    (
                        "1",
                        std::make_shared<IDK::SiLU>()
                    ),
                    std::make_pair<std::string, std::shared_ptr<Module>>
                    (
                        "3",
                        std::make_shared<IDK::Conv2d>
                        (
                            IDK::ConvNdOptions<2>(out_channels_, out_channels_, {3, 3})
                            .padding({1, 1}),
                            Dimensionformat::NCHW,
                            dtype
                        )
                    )
                }
            )
        )
    {
        if (out_channels_ == channels_)
        {
            skip_connection_ = std::make_shared<IDK::Identity>();
        }
        else
        {
            skip_connection_ =
                std::make_shared<IDK::Conv2d>
                (
                    IDK::ConvNdOptions<2>(channels_, out_channels_, {1, 1}),
                    Dimensionformat::NCHW,
                    dtype
                );
        }
        
        register_module("in_layers", in_layers_);
        register_module("emb_layers", emb_layers_);
        register_module("out_layers", out_layers_);
        register_module("skip_connection", skip_connection_);
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        VARP emb = inputs[1];
        VARP h = in_layers_->forward(x);
        VARP emb_out = emb_layers_->forward(emb);
        
        auto h_info = h->getInfo();
        
        {
            halide_type_t h_dtype = h_info->type;
            emb_out = 
                h_dtype == emb_out->getInfo()->type
                ? emb_out
                : _Cast(emb_out, h_dtype);
        }
        
        {
            int emb_out_dim_size = emb_out->getInfo()->dim.size();
            while (emb_out_dim_size < h_info->dim.size())
            {
                emb_out = _Unsqueeze(emb_out, {emb_out_dim_size});
                emb_out_dim_size += 1;
            }
        }
        
        h = h + emb_out;
        h = out_layers_->forward(h);
        
        return {skip_connection_->forward(x) + h};
    }
};



struct UNetModel : public IDK::MyModule
{
    int model_channels_;
    halide_type_t dtype_;
    std::shared_ptr<IDK::MyModule> time_embed_{nullptr};
    std::shared_ptr<IDK::ModuleList> input_blocks_{nullptr};
    std::shared_ptr<TimestepEmbedSequential> middle_block_{nullptr};
    std::shared_ptr<IDK::ModuleList> output_blocks_{nullptr};
    std::shared_ptr<IDK::MyModule> out_{nullptr};
    
    UNetModel
    (
        int in_channels,
        int model_channels,
        int out_channels,
        int num_res_blocks,
        int num_heads = -1,
        int transformer_depth = 1,
        int context_dim = -1,
        std::vector<int> attention_resolutions = {},
        std::vector<int> channel_mult = {1, 2, 4, 8},
        halide_type_t dtype = halide_type_of<float>()
    ) :
        dtype_(dtype),
        model_channels_(model_channels),
        time_embed_
        (
            std::make_shared<Sequential>
            (
                std::vector<std::shared_ptr<Module>>
                {
                    std::make_shared<IDK::Linear>(model_channels_, model_channels_ * 4, true, dtype),
                    std::make_shared<IDK::SiLU>(),
                    std::make_shared<IDK::Linear>(model_channels_ * 4, model_channels_ * 4, true, dtype)
                }
            )
        ),
        input_blocks_
        (
            std::make_shared<IDK::ModuleList>
            (
                std::vector<std::shared_ptr<Module>>
                {
                    std::make_shared<TimestepEmbedSequential>
                    (
                        std::vector<std::shared_ptr<Module>>
                        {
                            makeTimestepBlock<IDK::Conv2d>
                            (
                                IDK::ConvNdOptions<2>(in_channels, model_channels_, {3, 3})
                                    .padding({1, 1}),
                                Dimensionformat::NCHW,
                                dtype
                            )
                        }
                    )
                }
            )
        ),
        output_blocks_(std::make_shared<IDK::ModuleList>())
    {
        int ch = model_channels_;
        int _feature_size = model_channels_;
        int ds = 1;
        std::vector<int> input_block_chans = {model_channels_};
        int dim_head;
        
        for (int level = 0, len = channel_mult.size(); level < len; level++)
        {
            int mult = channel_mult[level];
            for (int _ = 0; _ < num_res_blocks; _++)
            {
                std::vector<std::shared_ptr<Module>> layers =
                    {
                        makeTimestepBlock<ResBlock>
                        (
                            ch,
                            model_channels_ * 4,
                            mult * model_channels_,
                            dtype
                        )
                    };
                
                ch = mult * model_channels_;
                if
                (
                    std::find
                    (
                        attention_resolutions.begin(),
                        attention_resolutions.end(),
                        ds
                    ) != attention_resolutions.end()
                )
                {
                    dim_head = ch / num_heads;
                    layers.emplace_back
                    (
                        makeSpatialTransformer
                        (
                            ch,
                            num_heads,
                            dim_head,
                            transformer_depth,
                            context_dim,
                            dtype
                        )
                    );
                }
                input_blocks_->emplace_back(std::make_shared<TimestepEmbedSequential>(layers));
                _feature_size += ch;
                input_block_chans.emplace_back(ch);
            }
            if (level != channel_mult.size() - 1)
            {
                int out_ch = ch;
                input_blocks_->emplace_back
                (
                    std::make_shared<TimestepEmbedSequential>
                    (
                        std::vector<std::shared_ptr<Module>>
                        {
                            makeGuessWhoIAm<Downsample>(ch, out_ch, 1, dtype)
                        }
                    )
                );
                ch = out_ch;
                input_block_chans.emplace_back(ch);
                ds *= 2;
                _feature_size += ch;
            }
        }
        dim_head = ch / num_heads;
        middle_block_ =
            std::make_shared<TimestepEmbedSequential>
            (
                std::vector<std::shared_ptr<Module>>
                {
                    makeTimestepBlock<ResBlock>
                    (
                        ch,
                        model_channels_ * 4,
                        ch,
                        dtype
                    ),
                    makeSpatialTransformer
                    (
                        ch,
                        num_heads,
                        dim_head,
                        transformer_depth,
                        context_dim,
                        dtype
                    ),
                    makeTimestepBlock<ResBlock>
                    (
                        ch,
                        model_channels_ * 4,
                        ch,
                        dtype
                    )
                }
            );
        _feature_size += ch;
        
        for (int level = channel_mult.size() - 1; level >= 0; level--)
        {
            int mult = channel_mult[level];
            for (int i = 0, n = num_res_blocks + 1; i < n; i++)
            {
                int ich = input_block_chans.back();
                input_block_chans.pop_back();
                std::vector<std::shared_ptr<Module>> layers =
                    {
                        makeTimestepBlock<ResBlock>
                        (
                            ch + ich,
                            model_channels_ * 4,
                            mult * model_channels_,
                            dtype
                        )
                    };
                ch = model_channels * mult;
                if
                (
                    std::find
                    (
                        attention_resolutions.begin(),
                        attention_resolutions.end(),
                        ds
                    ) != attention_resolutions.end()
                )
                {
                    dim_head = ch / num_heads;
                    layers.emplace_back
                    (
                        makeSpatialTransformer
                        (
                            ch,
                            num_heads,
                            dim_head,
                            transformer_depth,
                            context_dim,
                            dtype
                        )
                    );
                }
                if (level != 0 && i == num_res_blocks)
                {
                    int out_ch = ch;
                    layers.emplace_back
                    (
                        makeGuessWhoIAm<Upsample>(ch, out_ch, 1, dtype)
                    );
                    ds /= 2;
                }
                output_blocks_->emplace_back(std::make_shared<TimestepEmbedSequential>(layers));
                _feature_size += ch;
                
                
            }
        }
        
        out_ =
            std::make_shared<Sequential>
            (
                std::vector<std::shared_ptr<Module>>
                {
                    std::make_shared<IDK::GroupNorm>
                    (
                        32, ch, 1e-05, true, Dimensionformat::NCHW, dtype
                    ),
                    std::make_shared<IDK::SiLU>(),
                    makeTimestepBlock<IDK::Conv2d>
                    (
                        IDK::ConvNdOptions<2>(model_channels_, out_channels, {3, 3})
                            .padding({1, 1}),
                        Dimensionformat::NCHW,
                        dtype
                    )
                }
            );
                    
        register_module("time_embed", time_embed_);
        register_module("input_blocks", input_blocks_);
        register_module("middle_block", middle_block_);
        register_module("output_blocks", output_blocks_);
        register_module("out", out_);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        VARP timesteps = inputs[1];
        VARP context =
            (inputs.size() >= 3 && inputs[2].get())
            ? inputs[2]
            : nullptr;
        
        VARPS hs = {};
        VARP t_emb = timestep_embedding(timesteps, this->model_channels_, 10000, false);
        VARP emb = this->time_embed_->forward(t_emb);
        
        halide_type_t x_dtype = x->getInfo()->type;
        
        VARP h =
            x_dtype == dtype_
            ? x
            : _Cast(x, dtype_);
        
        for (auto module : *(this->input_blocks_))
        {
            h = module->onForward({h, emb, context})[0];
            hs.emplace_back(h);
        }
        
        h = this->middle_block_->onForward({h, emb, context})[0];
        
        for (auto module : *(this->output_blocks_))
        {
            h = _Concat({h, hs.back()}, 1);
            hs.pop_back();
            h = module->onForward({h, emb, context})[0];
        }
        
        h =
            x_dtype == dtype_
            ? h
            : _Cast(h, x_dtype);
            
        return {this->out_->forward(h)};
    }
};

}//namespace OpenaiModel
}//namespace MyLDM

#endif