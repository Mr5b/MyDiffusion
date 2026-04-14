#include "openaimodel.h"
int main()
{
    using namespace DonNotKnowHowToNameIt;
    using namespace MNN::Express;
    using namespace MyLDM;
    
    int in_channels = 4;
    int batch_size = 1;
    int height = 8;
    int width = 8;
    
    int sequence_length = 77;
    
    int model_channels = 320;
    int out_channels = 4;
    int num_res_blocks = 2;
    int num_heads = 8;
    int transformer_depth = 1;
    int context_dim = 768;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult = {1, 2, 4, 4};
    
    /*UNetModel
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
    )*/
    
    OpenaiModel::UNetModel model
    (
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        num_heads,
        transformer_depth,
        context_dim,
        attention_resolutions,
        channel_mult
    );
    
    SafetensorLoader loader("/storage/emulated/0/Download/Browser/v1-5-pruned-emaonly.safetensors");
    model.load_from_safetensors(loader, "model.diffusion_model.");
    
    VARP input = _LinSpace(_Scalar(0.0f), _Scalar(1.0f), _Scalar(int(batch_size*in_channels*height*width)));
    VARP timesteps = _LinSpace(_Scalar<int>(0), _Scalar<int>(2), _Scalar(int(batch_size)));
    VARP context = _LinSpace(_Scalar<float>(0), _Scalar<float>(2), _Scalar(int(batch_size*sequence_length*context_dim)));
 
    
    
    input.fix(VARP::InputType::INPUT);
    
    input.setOrder(MNN::Express::Dimensionformat::NCHW);
    
    
    input = _Reshape(input, {batch_size, in_channels, height, width}, MNN::Express::Dimensionformat::NCHW);
    context = _Reshape(context, {batch_size, sequence_length, context_dim}, MNN::Express::Dimensionformat::NCHW);
    input->setName("test_input");
    VARP output = model.onForward({input, timesteps, context})[0];
    output->setName("test_output");
    
    //pppp(output);
    
    auto rptr = output->readMap<float>();
    for (int i = 0, n = output->getTensor()->elementSize(); i < n; i++)
    {
        std::cout << rptr[i] << " ";
    }
    
    //Variable::save({output}, "/storage/emulated/0/MyDiffusion/MyDiffusion/MyLDM/models/v1-5-pruned-emaonly_first_stage_model_test.mnn");
    
    //std::cout << std::endl;
    print_shape(output);
    
    return 0;
}