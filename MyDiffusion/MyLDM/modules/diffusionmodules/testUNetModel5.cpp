#include "openaimodel.h"
#include "model.h"
#include <chrono>

extern "C" void MNNSetCustomOpenCLLibraryPaths(const char** paths, int count);

int main()
{
    using namespace DonNotKnowHowToNameIt;
    using namespace MNN::Express;
    using namespace MyLDM;
    
    const char* cl_paths[] = {"/data/data/com.termux/files/usr/opt/vendor/lib/libOpenCL.so"};
    MNNSetCustomOpenCLLibraryPaths(cl_paths, 1);
    auto executor = Executor::getGlobalExecutor();
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    //executor->setGlobalExecutorConfig(MNN_FORWARD_CPU_EXTENSION, backendConfig, 8);
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, backendConfig, 1);
    int in_channels = 4;
    int batch_size = 1;
    int height = 48;
    int width = 48;
    
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
    
    std::shared_ptr<MyModule> model =
    std::make_shared<OpenaiModel::UNetModelLow>
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
    model->load_from_safetensors(loader, "model.diffusion_model.");
    
    /*std::string model_path = "/storage/emulated/0/sdv1-5/unet.mnn";
    std::shared_ptr<Module> model;
    model.reset(Module::load({"sample", "timestep", "encoder_hidden_states"}, {"out_sample"}, model_path.c_str()));
*/
    
    VARP input = _LinSpace(_Scalar<float>(0.0f), _Scalar<float>(1.0f), _Scalar<int>(int(batch_size*in_channels*height*width)));
    VARP timesteps = _LinSpace(_Scalar<float>(0.0f), _Scalar<float>(2.0f), _Scalar<int>(int(batch_size)));
    VARP context = _LinSpace(_Scalar<float>(0.0f), _Scalar<float>(2.0f), _Scalar<int>(int(batch_size*sequence_length*context_dim)));

    //timesteps = _Cast<int>(timesteps);

    input.fix(VARP::InputType::INPUT);
    
    input.setOrder(MNN::Express::Dimensionformat::NCHW);
    
    
    input = _Reshape(input, {batch_size, in_channels, height, width}, MNN::Express::Dimensionformat::NCHW);
    context = _Reshape(context, {batch_size, sequence_length, context_dim}, MNN::Express::Dimensionformat::NCHW);
    
    input.fix(VARP::InputType::INPUT);
    timesteps.fix(VARP::InputType::INPUT);
    context.fix(VARP::InputType::INPUT);
    
    //input->setName("test_input");
    auto start = std::chrono::high_resolution_clock::now();
    VARP output = model->onForward({input, timesteps, context})[0];
    
    //output->setName("test_output");
    
    
    //auto start = std::chrono::high_resolution_clock::now();
    auto rptr = output->readMap<float>();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;
    
    for (int i = 0, n = output->getTensor()->elementSize(); i < n; i++)
    {
        std::cout << rptr[i] << " ";
    }
    std::cout << std::endl;
    
    VARP input1 = _LinSpace(_Scalar<float>(0.0f), _Scalar<float>(1.0f), _Scalar<int>(int(batch_size*in_channels*height*width)));
    VARP timesteps1 = _LinSpace(_Scalar<float>(0.0f), _Scalar<float>(2.0f), _Scalar<int>(int(batch_size)));
    VARP context1 = _LinSpace(_Scalar<float>(0.0f), _Scalar<float>(2.0f), _Scalar<int>(int(batch_size*sequence_length*context_dim)));
    
    input1 = _Reshape(input, {batch_size, in_channels, height, width}, MNN::Express::Dimensionformat::NCHW);
    context1 = _Reshape(context, {batch_size, sequence_length, context_dim}, MNN::Express::Dimensionformat::NCHW);
    
    input1.fix(VARP::InputType::INPUT);
    timesteps1.fix(VARP::InputType::INPUT);
    context1.fix(VARP::InputType::INPUT);
    //timesteps = _Cast<int>(timesteps);

    /*input.fix(VARP::InputType::INPUT);
    
    input.setOrder(MNN::Express::Dimensionformat::NCHW);*/
    
    
    std::cout << "😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤TIME0:" << time.count() << "s" << std::endl;
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, backendConfig, 1);
    //auto iptr = input->writeMap<float>();
    //iptr[0] = 1;
    //std::cout << iptr << std::endl;
    //input->unMap();
    start = std::chrono::high_resolution_clock::now();
    output = model->onForward({input1, timesteps1, context1})[0];
    /*iptr = input->writeMap<float>();
    std::cout << iptr << std::endl;*/
    
    //start = std::chrono::high_resolution_clock::now();
    rptr = output->readMap<float>();
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    
    for (int i = 0, n = output->getTensor()->elementSize(); i < n; i++)
    {
        std::cout << rptr[i] << " ";
    }
    
    std::cout << std::endl;
    //pppp(output);
    //std::cout << "rptr[0]: " << rptr[0] << std::endl;
    
    //Variable::save({output}, "/storage/emulated/0/MyDiffusion/MyDiffusion/MyLDM/models/v1-5-pruned-emaonly_first_stage_model_test.mnn");
    std::cout << "😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤TIME1:" << time.count() << "s" << std::endl;
 
    
    print_shape(output);
    
    return 0;
}