#include "autoencoder.h"
int main()
{
    using namespace DonNotKnowHowToNameIt;
    using namespace MNN::Express;
    using namespace MyLDM;
    
    int in_channels = 3;
    int batch_size = 1;
    int height = 64;
    int width = 64;
    
    
    int embed_dim = 4;
    int ch = 128;
    int out_ch = 3;
    int num_res_blocks = 2;
    int resolution = 0;
    int z_channels = 4;
    bool double_z = true;
    std::vector<int> ch_mult = {1, 2, 4, 4};
    bool tanh_out = false;
    
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
    
    AutoencoderKL model
    (
        embed_dim,
        z_channels,
        ch,
        in_channels,
        num_res_blocks,
        resolution,
        {},
        tanh_out,
        true,
        ch_mult
    );
    
    SafetensorLoader loader("/storage/emulated/0/Download/Browser/v1-5-pruned-emaonly.safetensors");
    model.load_from_safetensors(loader, "first_stage_model.");
    
    VARP input = _LinSpace(_Scalar(0.0f), _Scalar(1.0f), _Scalar(int(batch_size*in_channels*height*width)));
    
    input.fix(VARP::InputType::INPUT);
    
    input.setOrder(MNN::Express::Dimensionformat::NCHW);
    
    
    input = _Reshape(input, {batch_size, in_channels, height, width}, MNN::Express::Dimensionformat::NCHW);
    input->setName("test_input");
    VARP output = model.forward(input, false).first;
    output->setName("test_output");
    
    //pppp(output);
    
    auto rptr = output->readMap<float>();
    for (int i = 0, n = output->getTensor()->elementSize(); i < n; i++)
    {
        std::cout << rptr[i] << " ";
    }
    
    Variable::save({output}, "/storage/emulated/0/MyDiffusion/MyDiffusion/MyLDM/models/v1-5-pruned-emaonly_first_stage_model_test.mnn");
    
    //std::cout << std::endl;
    print_shape(output);
    
    return 0;
}