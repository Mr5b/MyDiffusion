#include "test.h"
using namespace DonNotKnowHowToNameIt;
using namespace MNN::Express;
using namespace MyLDM;
    
VARP LoadFromImg(std::string path, int w, int h)
{
    VARP img = MyTensor::Core::load_from_image(path.c_str(), w, h);
    img = _Permute(img, {0, 3, 1, 2});
    img.fix(VARP::InputType::INPUT);
    img.setOrder(MNN::Express::Dimensionformat::NCHW);
    //auto img_ptr = img->readMap<uint8_t>();
    const INTS& img_shape = img->getInfo()->dim;
    
    img =
        _Slice
        (
            img,
            _Const
            (
                std::vector<int>({0, 0, 0, 0}).data(), {4}, NCHW, halide_type_of<int>()
            ),
            _Const
            (
                std::vector<int>({img_shape[0], 3, img_shape[2], img_shape[3]}).data(), {4}, NCHW, halide_type_of<int>()
            )
        );
    //std::cout << img_info->dim.size() << std::endl;
    
    
    
    img = _Cast<float>(img);//_Const(nullptr, img_info->dim, NCHW, halide_type_of<float>());
    img = (img - _Scalar<float>(128)) / _Scalar<float>(127);
    /*auto input_ptr = input->writeMap<float>();
    for (int i = 0, n = img_info->size; i < n; i++)
    {
        input_ptr[i] = (static_cast<float>(img_ptr[i]) - 127.5f)/127.5f;
    }*/
    //print_shape(img);
    //auto img_tensor = img->getTensor();
    
    
    
    
    return img;
}

int main()
{
    
    
    int embed_dim = 4;
    int batch_size = 1;
    int height = 8;
    int width = 8;
    
    int sequence_length = 77;
    
    int model_channels = 320;
    int in_channels = 3;
    int unet_out_channels = 4;
    int num_res_blocks = 2;
    int num_heads = 8;
    int transformer_depth = 1;
    int context_dim = 768;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult = {1, 2, 4, 4};
    std::vector<int> ch_mult = {1, 2, 4, 4};
    int z_channels = 4;
    int ch = 128;
    int resolution = 256;
    bool tanh_out = false;
    bool double_z = true;
    
    auto vae =
    std::make_shared<AutoencoderKL>
    (
        embed_dim,
        z_channels,
        ch,
        in_channels,
        num_res_blocks,
        resolution,
        std::vector<int>{},
        tanh_out,
        true,
        ch_mult,
        "vanilla",
        halide_type_of<float>()
    );
    /*AutoencoderKL
    (
        int embed_dim,
        int z_channels,
        int ch,
        int in_channels,
        int num_res_blocks,
        int resolution,
        std::vector<int> attn_resolutions,
        bool tanh_out,
        bool resamp_with_conv = true,
        std::vector<int> ch_mult = {1, 2, 4, 4},
        std::string attn_type = "vanilla",
        halide_type_t dtype = halide_type_of<float>()
    )*/
    auto unet =
    std::make_shared<MyLDM::OpenaiModel::UNetModel>
    (
        embed_dim,
        model_channels,
        unet_out_channels,
        num_res_blocks,
        num_heads,
        transformer_depth,
        context_dim,
        attention_resolutions,
        channel_mult
    );
    
    SafetensorLoader loader("/storage/emulated/0/Download/Browser/v1-5-pruned-emaonly.safetensors");
    
    DDIMSampler model(vae, unet, 0.18215f, 1000, 0.00085f, 0.0120f);
    model.load_from_safetensors(loader, "");
    /*VARP input = _LinSpace(_Scalar(0.0f), _Scalar(1.0f), _Scalar(int(batch_size*in_channels*height*width)));
    VARP timesteps = _LinSpace(_Scalar<int>(0), _Scalar<int>(2), _Scalar(int(batch_size)));
    VARP context = _LinSpace(_Scalar<float>(0), _Scalar<float>(2), _Scalar(int(batch_size*sequence_length*context_dim)));*/
 
    
    
    /*input.fix(VARP::InputType::INPUT);
    
    input.setOrder(MNN::Express::Dimensionformat::NCHW);
    
    
    input = _Reshape(input, {batch_size, in_channels, height, width}, MNN::Express::Dimensionformat::NCHW);
    context = _Reshape(context, {batch_size, sequence_length, context_dim}, MNN::Express::Dimensionformat::NCHW);
    */
    VARP img = LoadFromImg("/storage/emulated/0/Pictures/IMG_20260415_080626.jpg", 64, 64);
    
    print_shape(img);
    
    VARP output = model.sample(img, nullptr, 0.9f, 25, 0.0f);
    
    output = model.first_stage_model_->decode(output);
    
    output = _Permute(output, {0, 2, 3, 1});
    output = (output + _Scalar<float>(1)) * _Scalar<float>(127.5);
    output = _Minimum(_Maximum(output, _Scalar<float>(0)), _Scalar<float>(255));
    output = _Cast<uint8_t>(output);
    //pppp(output);
    
    auto rptr = output->readMap<uint8_t>();
    for (int i = 0, n = output->getTensor()->elementSize(); i < n; i++)
    {
        std::cout << int(rptr[i]) << " ";
    }
    std::cout << std::endl;
    
    
    const INTS& rdim = output->getInfo()->dim;
    
    int stride_bytes = rdim[2]*rdim[3]*sizeof(uint8_t);
    for (auto d : rdim) std::cout << d << " ";
    std::cout << std::endl;
    if (stbi_write_png("/storage/emulated/0/MyDiffusion/MyDiffusion/MyLDM/models/diffusion/result.png", rdim[2], rdim[1], rdim[3], rptr, stride_bytes) == 0) throw std::runtime_error("保存失败");
    
    
    //Variable::save({output}, "/storage/emulated/0/MyDiffusion/MyDiffusion/MyLDM/models/v1-5-pruned-emaonly_first_stage_model_test.mnn");
    
    //std::cout << std::endl;
    print_shape(output);
    
    return 0;
}