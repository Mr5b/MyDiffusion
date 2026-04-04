#ifndef MYLDM_DIFFUSIONMODULES_MODEL
#define MYLDM_DIFFUSIONMODULES_MODEL

#include <cmath>
#include <MyModules.h>
#include "../attention.h"

namespace MyLDM
{
using namespace DonNotKnowHowToNameIt;

/*
def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
        
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(silu(temb))[:,:,None,None]

        h = self.norm2(h)
        h = silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h*/

std::shared_ptr<GroupNorm> Normalize(halide_type_t dtype, int in_channels, int num_groups = 32)
{
    return std::make_shared<GroupNorm>(num_groups, in_channels, 1e-6, true, Dimensionformat::NCHW, dtype);
}


struct Downsample : public MyModule
{
public:
    std::shared_ptr<MyModule> conv_{nullptr};
    bool with_conv_;
    
    Downsample
    (
        int in_channels,
        bool with_conv,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        with_conv_(with_conv)
    {
        if (with_conv_)
        {
            conv_ =
                std::make_shared<Conv2d>
                (
                    ConvNdOptions<2>(in_channels, in_channels, {3, 3})
                        .stride({2, 2})
                        .padding({0, 0}),
                    Dimensionformat::NCHW,
                    dtype
                );
            
            register_module("conv", conv_);
        }
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
            
        if (with_conv_)
        {
            VARP paddings =
                _Const
                (
                    std::vector<int>({0, 0, 0, 0, 0, 1, 0, 1}).data(),
                    {4, 2},
                    NCHW,
                    halide_type_of<int>()
                );
                
            x = _Pad(x, paddings, CONSTANT);
            x = conv_->forward(x);
        }
        else
        {
            x = _AvePool(x, {2, 2}, {2, 2});
        }
        
        return {x};
    }
};


struct Upsample : public MyModule
{
public:
    std::shared_ptr<MyModule> conv_{nullptr};
    bool with_conv_;
    
    Upsample
    (
        int in_channels,
        bool with_conv,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        with_conv_(with_conv)
    {
        if (with_conv_)
        {
            conv_ =
                std::make_shared<Conv2d>
                (
                    ConvNdOptions<2>(in_channels, in_channels, {3, 3})
                        .stride({1, 1})
                        .padding({1, 1}),
                    Dimensionformat::NCHW,
                    dtype
                );
            
            register_module("conv", conv_);
        }
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        //x = _Resize(x, 2, 2);
        x = _Interp({x}, 2, 2, 0, 0, MNN::Express::InterpolationMethod::NEAREST, false);
        if (with_conv_)
        {
            x = conv_->forward(x);
        }
        return {x};
    }
};

        
struct ResnetBlock : public MyModule
{
public:
    int in_channels_, out_channels_;
    bool use_conv_shortcut_;

    std::shared_ptr<MyModule> norm1_{nullptr};
    std::shared_ptr<MyModule> conv1_{nullptr};
    std::shared_ptr<MyModule> norm2_{nullptr};
    std::shared_ptr<MyModule> temb_proj_{nullptr};
    std::shared_ptr<MyModule> conv2_{nullptr};
    std::shared_ptr<MyModule> conv_shortcut_{nullptr};
    std::shared_ptr<MyModule> nin_shortcut_{nullptr};

    ResnetBlock
    (
        int in_channels,
        int out_channels = -1,
        bool conv_shortcut = false,
        int temb_channels = 512,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        in_channels_(in_channels),
        out_channels_((out_channels < 0) ? in_channels : out_channels),
        use_conv_shortcut_(conv_shortcut),
        norm1_(Normalize(dtype, in_channels)),
        conv1_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(in_channels_, out_channels_, {3, 3})
                    .stride({1, 1})
                    .padding({1, 1}),
                Dimensionformat::NCHW,
                dtype
            )
        ),
        norm2_(Normalize(dtype, out_channels_)),
        conv2_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(out_channels_, out_channels_, {3, 3})
                    .stride({1, 1})
                    .padding({1, 1}),
                Dimensionformat::NCHW,
                dtype
            )
        )
    {
        register_module("norm1", norm1_);
        register_module("conv1", conv1_);
        register_module("norm2", norm2_);
        register_module("conv2", conv2_);
        
        
        if (temb_channels > 0)
        {
            temb_proj_ =
                std::make_shared<Linear>(temb_channels, out_channels_);
            register_module("temb_proj", temb_proj_);
        }
        
        if (in_channels_ != out_channels_)
        {
            if (use_conv_shortcut_)
            {
                conv_shortcut_ =
                    std::make_shared<Conv2d>
                    (
                        ConvNdOptions<2>(in_channels_, out_channels_, {3, 3})
                            .stride({1, 1})
                            .padding({1, 1}),
                        Dimensionformat::NCHW,
                        dtype
                    );
                
                register_module("conv_shortcut", conv_shortcut_);
            }
            else
            {
                nin_shortcut_ =
                    std::make_shared<Conv2d>
                    (
                        ConvNdOptions<2>(in_channels_, out_channels_, {1, 1})
                            .stride({1, 1})
                            .padding({0, 0}),
                        Dimensionformat::NCHW,
                        dtype
                    );
                    
                register_module("nin_shortcut", nin_shortcut_);
            }
        }
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        VARP h = x;
        h = norm1_->forward(h);
        h = _Silu(h);
        h = conv1_->forward(h);
        
        VARP temb =
            (inputs.size() >= 2 && inputs[1].get())
            ? inputs[1]
            : nullptr;
        
        if (temb.get())
        {
            h = h + _Unsqueeze(temb_proj_->forward(_Silu(temb)), {2, 3});
        }

        h = norm2_->forward(h);
        h = _Silu(h);
        h = conv2_->forward(h);

        if (in_channels_ != out_channels_)
        {
            if (use_conv_shortcut_)
            {
                x = conv_shortcut_->forward(x);
            }
            else
            {
                x = nin_shortcut_->forward(x);
            }
        }
        
        return {x+h};
    }
};


struct AttnBlock : public MyModule
{
public:
    std::shared_ptr<MyModule> norm_{nullptr};
    std::shared_ptr<MyModule> q_{nullptr};
    std::shared_ptr<MyModule> k_{nullptr};
    std::shared_ptr<MyModule> v_{nullptr};
    std::shared_ptr<MyModule> proj_out_{nullptr};
    
    AttnBlock(int in_channels, halide_type_t dtype = halide_type_of<float>()) :
        norm_(Normalize(dtype, in_channels)),
        q_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(in_channels, in_channels, {1, 1})
                    .stride({1, 1})
                    .padding({0, 0}),
                Dimensionformat::NCHW,
                dtype
            )
        ),
        k_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(in_channels, in_channels, {1, 1})
                    .stride({1, 1})
                    .padding({0, 0}),
                Dimensionformat::NCHW,
                dtype
            )
        ),
        v_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(in_channels, in_channels, {1, 1})
                    .stride({1, 1})
                    .padding({0, 0}),
                Dimensionformat::NCHW,
                dtype
            )
        ),
        proj_out_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(in_channels, in_channels, {1, 1})
                    .stride({1, 1})
                    .padding({0, 0}),
                Dimensionformat::NCHW,
                dtype
            )
        )
    {
        register_module("norm", norm_);
        register_module("q", q_);
        register_module("k", k_);
        register_module("v", v_);
        register_module("proj_out", proj_out_);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        
        MY_ASSERT(x->getInfo()->dim.size() == 4, "x.ndim() != 4");
        MY_ASSERT(x->getInfo()->order == NCHW, "");
        
        VARP h_ = x;
        h_ = norm_->forward(h_);
        VARP q = q_->forward(h_);
        VARP k = k_->forward(h_);
        VARP v = v_->forward(h_);
        
        INTS shape = q->getInfo()->dim;
        const int b = shape[0];
        const int c = shape[1];
        const int h = shape[2];
        const int w = shape[3];
        
        q = _Reshape(q, {b, c, h*w});
        //q = _Permute(q, {0, 2, 1});// b, hw, c
        
        k = _Reshape(k, {b, c, h*w});// b, c, hw
        
        VARP w_ = _BatchMatMul(q, k, true, false);// b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * createScalar(std::pow(c, -0.5), w_->getInfo()->order, w_->getInfo()->type);
        w_ = _Softmax(w_, 2);
        
        v = _Reshape(v, {b, c, h*w});
        h_ = _BatchMatMul(v, w_, false, true);// b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        
        h_ = _Reshape(h_, {b, c, h, w});
        
        h_ = proj_out_->forward(h_);
        
        return {x + h_};
    }
};


struct LinAttnBlock : public LinearAttention
{
    LinAttnBlock(int in_channels, halide_type_t dtype = halide_type_of<float>()) :
        LinearAttention(in_channels, 1, in_channels, dtype)
    {}
};


std::shared_ptr<MyModule> make_attn
(
    int in_channels,
    halide_type_t dtype = halide_type_of<float>(),
    std::string attn_type = "vanilla"
)
{
    //assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    
    if (attn_type == "vanilla")
    {
        return std::make_shared<AttnBlock>(in_channels, dtype);
    }
    else if (attn_type == "none")
    {
        return std::make_shared<Identity>();
    }
    else
    {
        return std::make_shared<LinAttnBlock>(in_channels);
    }
}


/*class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = silu(h)
        h = self.conv_out(h)
        return h*/





struct Encoder : public MyModule
{
public:
    
    
    
    //downsampling
    std::shared_ptr<MyModule> conv_in_{nullptr};
    std::shared_ptr<ModuleList> down_{nullptr};
    std::shared_ptr<MyModule> mid_{nullptr};
    std::shared_ptr<MyModule> norm_out_{nullptr};
    std::shared_ptr<MyModule> conv_out_{nullptr};
    
    int num_resolutions_;
    int num_res_blocks_;
    //int temb_ch_;
    
    Encoder
    (
        int ch,
        int in_channels,
        int num_res_blocks,
        int resolution,
        int z_channels,
        std::vector<int> attn_resolutions,
        bool double_z = true,
        halide_type_t dtype = halide_type_of<float>(),
        std::vector<int> ch_mult = {1, 2, 4, 8},
        std::string attn_type = "vanilla",
        bool resamp_with_conv = true
    ) :
        num_res_blocks_(num_res_blocks),
        num_resolutions_(ch_mult.size()),
        conv_in_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(in_channels, ch, {3, 3})
                    .stride({1, 1})
                    .padding({1, 1})
            )
        ),
        down_(std::make_shared<ModuleList>()),
        mid_(std::make_shared<MyModule>())
    {
        std::vector<int> in_ch_mult{1};
        in_ch_mult.insert(in_ch_mult.end(), ch_mult.begin(), ch_mult.end());
        
        int temb_ch = 0;
        int curr_res = resolution;
        int block_in;
        
        for (int i_level = 0; i_level < num_resolutions_; i_level++)
        {
            std::shared_ptr<ModuleList> block = std::make_shared<ModuleList>();
            std::shared_ptr<ModuleList> attn = std::make_shared<ModuleList>();
            block_in = ch * in_ch_mult[i_level];
            int block_out = ch * ch_mult[i_level];
            
            for (int i_block = 0; i_block < num_res_blocks_; i_block++)
            {
                block->emplace_back
                (
                    std::make_shared<ResnetBlock>
                    (
                        block_in,
                        block_out,
                        false,
                        temb_ch,
                        dtype
                    )
                );
                
                block_in = block_out;
                if
                (
                    std::find
                    (
                        attn_resolutions.begin(),
                        attn_resolutions.end(),
                        curr_res
                    ) != attn_resolutions.end()
                )
                {
                    attn->emplace_back
                    (
                        make_attn(block_in, dtype, attn_type)
                    );
                }
            }
            
            auto down = std::make_shared<MyModule>();
            down->register_module("block", block);//0
            down->register_module("attn", attn);//1
            
            if (i_level != num_resolutions_ - 1)
            {
                down->register_module
                (
                    "downsample",
                    std::make_shared<Downsample>
                    (
                        block_in, resamp_with_conv, dtype
                    )
                );//2
                
                curr_res /= 2;
            }
            down_->emplace_back(down);
        }
    
        register_module("conv_in", conv_in_);
        register_module("down", down_);
        register_module("mid", mid_);
        
        mid_->register_module
        (
            "block_1",
            std::make_shared<ResnetBlock>
            (
                block_in,
                block_in,
                false,
                temb_ch,
                dtype
            )
        );//0
        
        mid_->register_module
        (
            "attn_1",
            make_attn(block_in, dtype, attn_type)
        );//1
        
        mid_->register_module
        (
            "block_2",
            std::make_shared<ResnetBlock>
            (
                block_in,
                block_in,
                false,
                temb_ch,
                dtype
            )
        );//2
        
        norm_out_ = Normalize(dtype, block_in);
        conv_out_ =
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>
                (
                    block_in,
                    double_z ? 2 * z_channels : z_channels,
                    {3, 3}
                )
                    .stride({1, 1})
                    .padding({1, 1})
            );
        
        register_module("norm_out", norm_out_);
        register_module("conv_out", conv_out_);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        //timestep embedding
        VARP x = inputs[0];
        VARP temb = nullptr;
        
        //downsampling
        VARPS hs = {conv_in_->forward(x)};
        for (int i_level = 0; i_level < num_resolutions_; i_level++)
        {
            auto modules = down_->getChildren(i_level)->getChildren();
            for (int i_block = 0; i_block < num_res_blocks_; i_block++)
            {
                VARP h =
                    reinterpret_cast<MyModule*>(modules[0].get())->
                        getChildren(i_block)->onForward({hs.back(), temb})[0];
                if (modules.size() >= 2 && modules[1]->getChildren().size() > 0)
                {
                    h = reinterpret_cast<MyModule*>(modules[1].get())->getChildren(i_block)->forward(h);
                }
                hs.emplace_back(h);
            }
            if (i_level != num_resolutions_ - 1)
            {
                hs.emplace_back(reinterpret_cast<MyModule*>(modules[2].get())->forward(hs.back()));
            }
        }
        
        //middle
        VARP h = hs.back();
        h = mid_->getChildren(0)->onForward({h, temb})[0];//block_1
        h = mid_->getChildren(1)->forward(h);//attn_1
        h = mid_->getChildren(2)->onForward({h, temb})[0];//block_2
        
        //end
        h = norm_out_->forward(h);
        h = _Silu(h);
        h = conv_out_->forward(h);
        
        return {h};
    }
};

/*class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = silu(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h*/
        
        
void pppp(VARP x)
{
    auto rptr = x->readMap<float>();
    int n = x->getTensor()->elementSize();
    if (n <= 20)
    {
        for (int i = 0; i < n; i++)
        {
            std::cout << rptr[i] << " ";
        }
    }
    else
    {
        for (int i = 0; i < 3; i++) std::cout << rptr[i] << " ";
        std::cout << "... ";
        for (int i = n - 3; i < n; i++) std::cout << rptr[i] << " ";
    }
    std::cout << std::endl;
}
        
struct Decoder : public MyModule
{
public:
    
    
    
    std::shared_ptr<MyModule> conv_in_{nullptr};
    std::shared_ptr<ModuleList> up_{nullptr};
    std::shared_ptr<MyModule> mid_{nullptr};
    std::shared_ptr<MyModule> norm_out_{nullptr};
    std::shared_ptr<MyModule> conv_out_{nullptr};
    
    int num_resolutions_;
    int num_res_blocks_;
    bool give_pre_end_;
    bool tanh_out_;
    
    Decoder
    (
        int ch,
        int out_ch,
        int num_res_blocks,
        int resolution,
        int z_channels,
        bool give_pre_end,
        bool tanh_out,
        std::vector<int> attn_resolutions,
        halide_type_t dtype = halide_type_of<float>(),
        std::vector<int> ch_mult = {1, 2, 4, 8},
        std::string attn_type = "vanilla",
        bool resamp_with_conv = true
    ) :
        num_res_blocks_(num_res_blocks),
        num_resolutions_(ch_mult.size()),
        mid_(std::make_shared<MyModule>()),
        give_pre_end_(give_pre_end),
        tanh_out_(tanh_out)
    {
        int temb_ch = 0;
        int curr_res = resolution / std::pow(2, (num_resolutions_ - 1));
        int block_in = ch * ch_mult[num_resolutions_ - 1];
        
        conv_in_ =
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(z_channels, block_in, {3, 3})
                    .stride({1, 1})
                    .padding({1, 1})
            );
            
        register_module("conv_in", conv_in_);
        
        
        register_module("mid", mid_);
        
        mid_->register_module
        (
            "block_1",
            std::make_shared<ResnetBlock>
            (
                block_in,
                block_in,
                false,
                temb_ch,
                dtype
            )
        );//0
        
        mid_->register_module
        (
            "attn_1",
            make_attn(block_in, dtype, attn_type)
        );//1
        
        mid_->register_module
        (
            "block_2",
            std::make_shared<ResnetBlock>
            (
                block_in,
                block_in,
                false,
                temb_ch,
                dtype
            )
        );//2
        
        std::vector<std::shared_ptr<Module>> ups;
        for (int i_level = num_resolutions_ - 1; i_level >= 0; i_level--)
        {
            std::shared_ptr<ModuleList> block = std::make_shared<ModuleList>();
            std::shared_ptr<ModuleList> attn = std::make_shared<ModuleList>();
            int block_out = ch * ch_mult[i_level];
            
            
            
            for (int i_block = 0; i_block < num_res_blocks_ + 1; i_block++)
            {
                block->emplace_back
                (
                    std::make_shared<ResnetBlock>
                    (
                        block_in,
                        block_out,
                        false,
                        temb_ch,
                        dtype
                    )
                );
                
                block_in = block_out;
                if
                (
                    std::find
                    (
                        attn_resolutions.begin(),
                        attn_resolutions.end(),
                        curr_res
                    ) != attn_resolutions.end()
                )
                {
                    attn->emplace_back
                    (
                        make_attn(block_in, dtype, attn_type)
                    );
                }
            }
            
            auto up = std::make_shared<MyModule>();
            up->register_module("block", block);//0
            up->register_module("attn", attn);//1
            
            if (i_level != 0)
            {
                up->register_module
                (
                    "upsample",
                    std::make_shared<Upsample>
                    (
                        block_in, resamp_with_conv, dtype
                    )
                );//2
                
                curr_res *= 2;
            }
            ups.insert(ups.begin(), up);
            //ups.emplace_back(up);
        }
        up_ = std::make_shared<ModuleList>(ups);
        register_module("up", up_);
        
        
        
        norm_out_ = Normalize(dtype, block_in);
        conv_out_ =
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>
                (
                    block_in,
                    out_ch,
                    {3, 3}
                )
                    .stride({1, 1})
                    .padding({1, 1})
            );
        
        register_module("norm_out", norm_out_);
        register_module("conv_out", conv_out_);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        //timestep embedding
        VARP z = inputs[0];
        VARP temb = nullptr;
        
        // z to block_in
        VARP h = conv_in_->forward(z);
        
        //middle
        h = mid_->getChildren(0)->onForward({h, temb})[0];//block_1
        h = mid_->getChildren(1)->forward(h);//attn_1
        h = mid_->getChildren(2)->onForward({h, temb})[0];//block_2

        //upsampling
        for (int i_level = num_resolutions_ - 1; i_level >= 0; i_level--)
        {
            auto modules = up_->getChildren(i_level)->getChildren();
            for (int i_block = 0, n = num_res_blocks_ + 1; i_block < n; i_block++)
            {
                h =
                    reinterpret_cast<MyModule*>(modules[0].get())->//block
                        getChildren(i_block)->onForward({h, temb})[0];
                if (modules.size() >= 2 && modules[1]->getChildren().size() > 0)
                {
                    h = reinterpret_cast<MyModule*>(modules[1].get())->getChildren(i_block)->forward(h);
                }
            }
            if (i_level != 0)
            {
                h = reinterpret_cast<MyModule*>(modules[2].get())->forward(h);
            }
        }   
        
        //end
        if (give_pre_end_) return {h};
        
        h = norm_out_->forward(h);
        h = _Silu(h);
        h = conv_out_->forward(h);
        
        if (tanh_out_) h = _Tanh(h);
        
        return {h};
    }
};

}//namespace MyLDM

#endif