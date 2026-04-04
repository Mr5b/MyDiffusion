#ifndef MYLDM_MODULES_ATTENTION_H
#define MYLDM_MODULES_ATTENTION_H

#include <MyModules.h>
#include <halide_type_range.h>

#include <cmath>

namespace MyLDM
{

using namespace DonNotKnowHowToNameIt;
using namespace MNN::Express;

/*
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)*/

std::shared_ptr<GroupNorm> Normalize(int in_channels, halide_type_t dtype = halide_type_of<float>())
{
    return std::make_shared<GroupNorm>(32, in_channels, 1e-6, true, Dimensionformat::NCHW, dtype);
}

/*

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)*/
        
        
struct GEGLU : public MyModule
{
public:
    std::shared_ptr<Linear> proj_{nullptr};
    
    GEGLU(int dim_in, int dim_out, halide_type_t dtype = halide_type_of<float>()) :
        proj_(std::make_shared<Linear>(dim_in, dim_out * 2, true, dtype))
    {
        register_module("proj", proj_);
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        x = proj_->forward(x);
        auto a = DonNotKnowHowToNameIt::chunk(x, 2, -1);
        return { a[0] * _Gelu(a[1]) };
    }
};
        
struct FeedForward : public MyModule
{
public:
    std::shared_ptr<Sequential> net_{nullptr};
    
    FeedForward
    (
        int dim, int dim_out = -1,
        int mult = 4, bool glu = false,
        halide_type_t dtype = halide_type_of<float>()
    )
    {
        int inner_dim = int(dim * mult);
        dim_out = dim_out < 0 ? dim : dim_out;
        std::shared_ptr<MyModule> project_in;
        if (!glu)
        {
            project_in =
            std::make_shared<Sequential>
            (
                std::vector<std::shared_ptr<Module>>
                {
                    std::make_shared<Linear>(dim, inner_dim, true, dtype),
                    std::make_shared<GELU>()
                }
            );
        }
        else
        {
            project_in = std::make_shared<GEGLU>(dim, inner_dim, dtype);
        }

        net_ =
            std::make_shared<Sequential>
            (
                std::vector<std::pair<std::string, std::shared_ptr<Module>>>
                {
                    std::make_pair<std::string, std::shared_ptr<Module>>
                    (
                        "0",
                        project_in
                    ),
                    std::make_pair<std::string, std::shared_ptr<Module>>
                    (
                        "2",
                        std::make_shared<Linear>(inner_dim, dim_out, true, dtype)
                    )
                }
            );
            
        register_module("net", net_);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        return {net_->forward(inputs[0])};
    }
};


        
/*class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)*/


struct LinearAttention : public MyModule
{
    int heads_;
    int dim_head_;

    std::shared_ptr<Conv2d> to_qkv{nullptr};
    std::shared_ptr<Conv2d> to_out{nullptr};
    
    LinearAttention
    (
        int dim, int heads = 4, int dim_head = 32,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        heads_(heads),
        dim_head_(dim_head),
        to_qkv
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(dim, dim_head*heads*3, {1, 1})
                    .bias(false),
                Dimensionformat::NCHW,
                dtype
            )
        ),
        to_out
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(dim_head*heads, dim, {1, 1}),
                Dimensionformat::NCHW,
                dtype
            )
        )
    {
        register_module("to_qkv", to_qkv);
        register_module("to_out", to_out);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        MY_ASSERT(x->getInfo()->dim.size() == 4, "x.ndim() != 4");
        MY_ASSERT(x->getInfo()->order == NCHW, "");
        
        INTS shape = x->getInfo()->dim;
        const int b = shape[0];
        const int c = shape[1];
        const int h = shape[2];
        const int w = shape[3];
        
        VARP qkv = to_qkv->forward(x);
        
        //std::cout << qkv << std::endl;
        
        //hidden_dim_ = heads_ * (qkv.size(1) / (3 * heads_));
        //std::cout << b*3*heads_*dim_head_*h*w << std::endl;
        /*qkv = _Reshape(qkv, {b, 3, heads_, dim_head_, h, w})
        qkv = _Permute(qkv, {1, 0, 2, 3, 4, 5});
        
        qkv = _Reshape(qkv, {3, b, heads_, dim_head_, h*w});*/
        
        //[b, 3*dim_head * heads, h, w]
        int hw = h*w;
        qkv = _Reshape(qkv, {b, 3*heads_, dim_head_, hw});
        //qkv = _Permute(qkv, {1, 0, 2, 3, 4});
        
        //qkv = _Reshape(qkv, {3, b, heads_, dim_head_, h*w});
        
        VARP q =
            _Slice
            (
                qkv,
                _Const(std::array<int, 4>({0, 0, 0, 0}).data(), {4}),
                _Const(std::array<int, 4>({b, heads_, dim_head_, hw}).data(), {4})
            );
        VARP k =
            _Slice
            (
                qkv,
                _Const(std::array<int, 4>({0, heads_, 0, 0}).data(), {4}),
                _Const(std::array<int, 4>({b, heads_, dim_head_, hw}).data(), {4})
            );;
        VARP v =
            _Slice
            (
                qkv,
                _Const(std::array<int, 4>({0, 2*heads_, 0, 0}).data(), {4}),
                _Const(std::array<int, 4>({b, heads_, dim_head_, hw}).data(), {4})
            );
        
        k = _Softmax(k);
        
        VARP context = _BatchMatMul(k, v, false, true);//[b, h, d, d]  
        
        VARP out = _BatchMatMul(context, q, true, false);//[b, h, d, n]
        
        out = _Reshape(out, {b, heads_*dim_head_, h, w});
        
        out = to_out->forward(out);
        return {out};
    }
};


/*class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_*/
        
struct SpatialSelfAttention : public MyModule
{
    std::shared_ptr<GroupNorm> norm{nullptr};
    std::shared_ptr<Conv2d> q{nullptr};
    std::shared_ptr<Conv2d> k{nullptr};
    std::shared_ptr<Conv2d> v{nullptr};
    std::shared_ptr<Conv2d> proj_out{nullptr};
    
    SpatialSelfAttention(int in_channels, halide_type_t dtype = halide_type_of<float>()) :
        norm(Normalize(in_channels, dtype)),
        q
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
        k
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
        v
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
        proj_out
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
        register_module("norm", norm);
        register_module("q", q);
        register_module("k", k);
        register_module("v", v);
        register_module("proj_out", proj_out);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        VARP h_ = x;
        MY_ASSERT(h_->getInfo()->dim.size() == 4, "x.ndim() != 4");
        MY_ASSERT(h_->getInfo()->order == NCHW, "");
        
        h_ = norm->forward(h_);
        
        VARP Q = q->forward(h_);
        VARP K = k->forward(h_);
        VARP V = v->forward(h_);
        
        INTS shape = Q->getInfo()->dim;
        const int b = shape[0];
        const int c = shape[1];
        const int h = shape[2];
        const int w = shape[3];
        
        int hw = h*w;
        Q = _Reshape(Q, {b, c, hw});
        K = _Reshape(K, {b, c, hw});
        VARP w_ = _BatchMatMul(Q, K, true, false);//[b, hw, hw]
        
        w_ = w_ * createScalar(std::pow(c, -0.5), w_->getInfo()->order, w_->getInfo()->type);
        
        w_ = _Softmax(w_);
        
        V = _Reshape(V, {b, c, hw});
        
        h_ = _BatchMatMul(V, w_, false, true);//[b, c, hw]
        h_ = _Reshape(h_, {b, c, h, w});
        h_ = proj_out->forward(h_);
        
        return {x + h_};
    }
};

/*class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)*/
        

struct CrossAttention : public MyModule
{
public:
    std::shared_ptr<MyModule> to_q{nullptr};
    std::shared_ptr<MyModule> to_k{nullptr};
    std::shared_ptr<MyModule> to_v{nullptr};
    std::shared_ptr<MyModule> to_out{nullptr};
    
    CrossAttention
    (
        int query_dim,
        int context_dim = -1,
        int heads = 8,
        int dim_head = 64,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        scale_(std::pow(dim_head, -0.5)),
        heads_(heads)
    {
        int inner_dim = dim_head * heads;
        if (context_dim <= 0) context_dim = query_dim;
        to_q = std::make_shared<Linear>(query_dim, inner_dim, false, dtype);
        to_k = std::make_shared<Linear>(context_dim, inner_dim, false, dtype);
        to_v = std::make_shared<Linear>(context_dim, inner_dim, false, dtype);
        to_out =
            std::make_shared<Sequential>
            (
                std::vector<std::shared_ptr<Module>>
                (
                    {std::make_shared<Linear>(inner_dim, query_dim, true, dtype)}
                )
            );
            
        register_module("to_q", to_q);
        register_module("to_k", to_k);
        register_module("to_v", to_v);
        register_module("to_out", to_out);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        MY_ASSERT(x->getInfo()->dim.size() == 3, "x.ndim() != 3");
        MY_ASSERT(x->getInfo()->order == NCHW, "");
        
        VARP q = to_q->forward(x);
        
        const INTS& shape = q->getInfo()->dim;
        int b = shape[0];
        int q_n = shape[1];
        
        VARP context =
            (inputs.size() >= 2 && inputs[1].get())
            ? inputs[1]
            : x;
        
        const INTS& context_shape = context->getInfo()->dim;
        int context_n = context_shape[1];
        
        VARP k = to_k->forward(context);
        VARP v = to_v->forward(context);
        
        int d = q->getInfo()->dim[2] / heads_;
        
        q = _Reshape(q, {b, q_n, heads_, d});
        q = _Permute(q, {0, 2, 1, 3});
        q = _Reshape(q, {b*heads_, q_n, d});
        
        k = _Reshape(k, {b, context_n, heads_, d});
        k = _Permute(k, {0, 2, 1, 3});
        k = _Reshape(k, {b*heads_, context_n, d});
        
        v = _Reshape(v, {b, context_n, heads_, d});
        v = _Permute(v, {0, 2, 1, 3});
        v = _Reshape(v, {b*heads_, context_n, d});
        
        VARP sim =
            _BatchMatMul(q, k, false, true) * createScalar(scale_, NCHW, q->getInfo()->type);
            
        if (inputs.size() == 3 && inputs[2].get())
        {
            VARP mask = inputs[2];
            
            {
                auto mask_info = mask->getInfo();
                int b = mask_info->dim[0];
                mask = _Reshape(mask, {b, int(mask_info->size / b)});
            }
            
            double max_neg_value = -1 * halide_type_max(sim->getInfo()->type);
            const INTS& mask_shape = mask->getInfo()->dim;
            mask =
                _BroadcastTo
                (
                    mask,
                    _Const
                    (
                        std::vector<int>
                        (
                            {heads_, mask_shape[0], mask_shape[1]}
                        ).data(),
                        {3},
                        NCHW,
                        halide_type_of<int>()
                    )
                );
                
            mask = _Reshape(mask, {mask_shape[0] * heads_, 1, mask_shape[1]});
                
            sim =
                _Select
                (
                    mask,                
                    sim,
                    createScalar(max_neg_value, NCHW, sim->getInfo()->type)
                );
        }
                            
        VARP attn = _Softmax(sim, -1);
        VARP out = _BatchMatMul(attn, v, false, false);
        
        out = _Reshape(out, {b, heads_, q_n, d});
        out = _Permute(out, {0, 2, 1, 3});
        out = _Reshape(out, {b, q_n, heads_*d});
        
        return {to_out->forward(out)};
    }
    
    float scale_;
    int heads_;
};


/*
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x*/
        

struct BasicTransformerBlock : public MyModule
{
public:
    std::shared_ptr<MyModule> attn1_{nullptr};
    std::shared_ptr<MyModule> ff_{nullptr};
    std::shared_ptr<MyModule> attn2_{nullptr};
    
    std::shared_ptr<MyModule> norm1_{nullptr};
    std::shared_ptr<MyModule> norm2_{nullptr};
    std::shared_ptr<MyModule> norm3_{nullptr};
    
    BasicTransformerBlock
    (
        int dim,
        int n_heads,
        int d_head,
        int context_dim = -1,
        bool gated_ff = true,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        attn1_(std::make_shared<CrossAttention>(dim, dim, n_heads, d_head, dtype)),
        ff_(std::make_shared<FeedForward>(dim, dim, 4, gated_ff, dtype)),
        attn2_(std::make_shared<CrossAttention>(dim, context_dim, n_heads, d_head, dtype)),
        norm1_(std::make_shared<LayerNorm>(INTS{dim}, dtype)),
        norm2_(std::make_shared<LayerNorm>(INTS{dim}, dtype)),
        norm3_(std::make_shared<LayerNorm>(INTS{dim}, dtype))
    {
        register_module("attn1", attn1_);
        register_module("ff", ff_);
        register_module("attn2", attn2_);
        register_module("norm1", norm1_);
        register_module("norm2", norm2_);
        register_module("norm3", norm3_);
    }
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        
        VARP context =
            (inputs.size() >= 2 && inputs[1].get())
            ? inputs[1]
            : nullptr;
            
        x = attn1_->forward(norm1_->forward(x)) + x;
        x = attn2_->onForward({norm2_->forward(x), context})[0] + x;
        x = ff_->forward(norm3_->forward(x)) + x;
        return {x};
    }
};


/*class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in*/
        

struct SpatialTransformer : public MyModule
{
public:
    std::shared_ptr<MyModule> norm_{nullptr};
    std::shared_ptr<MyModule> proj_in_{nullptr};
    std::shared_ptr<ModuleList> transformer_blocks_{nullptr};
    std::shared_ptr<MyModule> proj_out_{nullptr};
    
    SpatialTransformer
    (
        int in_channels,
        int n_heads,
        int d_head,
        int depth = 1,
        int context_dim = -1,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        norm_(Normalize(in_channels, dtype)),
        proj_in_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(in_channels, n_heads*d_head, {1, 1})
                    .stride({1, 1})
                    .padding({0, 0}),
                Dimensionformat::NCHW,
                dtype
            )
        ),
        transformer_blocks_(std::make_shared<ModuleList>()),
        proj_out_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(n_heads*d_head, in_channels, {1, 1})
                    .stride({1, 1})
                    .padding({0, 0}),
                Dimensionformat::NCHW,
                dtype
            )
        )
    {
        transformer_blocks_->reserve(depth);
        
        for (int i = 0; i < depth; i++)
        {
            transformer_blocks_->emplace_back
            (
                std::make_shared<BasicTransformerBlock>
                (
                    n_heads*d_head,
                    n_heads,
                    d_head,
                    context_dim,
                    true,
                    dtype
                )
            );
        }
    
        register_module("norm", norm_);
        register_module("proj_in", proj_in_);
        register_module("transformer_blocks", transformer_blocks_);
        register_module("proj_out", proj_out_);
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP x = inputs[0];
        MY_ASSERT(x->getInfo()->dim.size() == 4, "x.ndim() != 4");
        MY_ASSERT(x->getInfo()->order == NCHW, "");
        
        VARP context =
            (inputs.size() >= 2 && inputs[1].get())
            ? inputs[1]
            : nullptr;
        
        
        INTS shape = x->getInfo()->dim;
        const int b = shape[0];
        const int c = shape[1];
        const int h = shape[2];
        const int w = shape[3];
        
        VARP x_in = x;
        x = norm_->forward(x);
        x = proj_in_->forward(x);
        
        x = _Reshape(_Permute(x, {0, 2, 3, 1}), {b, h*w, c});
        
        for (auto module : *reinterpret_cast<ModuleList*>(transformer_blocks_.get()))
        {
            x = module->onForward({x, context})[0];
        }
        
        x = _Permute(_Reshape(x, {b, h, w, c}), {0, 3, 1, 2});
        x = proj_out_->forward(x);
        return {x + x_in};
    }
};


}//namespace MyLDM

#endif