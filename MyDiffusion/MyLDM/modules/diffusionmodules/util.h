#ifndef MYLDM_DIFFUSIONMODULES_UTIL
#define MYLDM_DIFFUSIONMODULES_UTIL

#include <MyModules.h>
#include <cmath>

namespace MyLDM
{
namespace IDK = DonNotKnowHowToNameIt;
using namespace MNN::Express;

//std::shared_ptr<Module> normalization(int channels);

/*def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding*/

VARP timestep_embedding
(
    VARP timesteps,
    int dim,
    int max_period = 10000,
    bool repeat_only = false
)
{
    MY_ASSERT(timesteps->getInfo()->dim.size() == 1, "");
    
    VARP embedding;
    if (!repeat_only)
    {
        int half = dim / 2;
        VARP freqs =
            _Exp
            (
                IDK::createScalar
                (
                    std::log(max_period) * -1,
                    MNN::Express::Dimensionformat::NCHW, halide_type_of<float>()
                ) *
                _LinSpace
                (
                    IDK::createScalar(0, MNN::Express::Dimensionformat::NCHW, halide_type_of<float>()),
                    IDK::createScalar(half - 1, MNN::Express::Dimensionformat::NCHW, halide_type_of<float>()),
                    _Scalar<int>(half)
                ) / IDK::createScalar(half, MNN::Express::Dimensionformat::NCHW, halide_type_of<float>())
            );
            
        {
            halide_type_t timesteps_dtype = timesteps->getInfo()->type;
            timesteps =
                timesteps_dtype == halide_type_of<float>()
                ? timesteps
                : _Cast<float>(timesteps);
        }
        
        VARP args = _Unsqueeze(timesteps, {1}) * _Unsqueeze(freqs, {0});
        embedding = _Concat({_Cos(args), _Sin(args)}, -1);
        
        if (dim % 2)
        {
            embedding =
                _Concat
                (
                    {embedding, IDK::fillValue(0, {embedding->getInfo()->dim[0], 1})},
                    -1
                );
        }
    }
    else
    {
        int timesteps_numel = static_cast<int>(timesteps->getInfo()->size);
        embedding =
            _Transpose
            (
                _BroadcastTo
                (
                    embedding,
                    _Const
                    (
                        std::vector<int>
                        (
                            {dim, timesteps_numel}
                        ).data(),
                        {1},
                        NCHW,
                        halide_type_of<int>()
                    )
                ),
                {0, 1}
            );
    }
    return embedding;
}


}//namespace MyLDM

#endif