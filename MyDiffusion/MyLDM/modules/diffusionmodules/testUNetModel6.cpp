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
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, backendConfig, 4);
    //executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, backendConfig, 1);
    
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
    
    std::shared_ptr<MyModule> model =
    std::make_shared<OpenaiModel::UNetModel>
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
    

    VARP input = _Input({batch_size, in_channels, height, width}, NCHW, halide_type_of<float>());
    VARP timesteps = _Input({batch_size}, NCHW, halide_type_of<int>());
    VARP context = _Input({batch_size, sequence_length, context_dim}, NCHW, halide_type_of<float>());
    
    
    
    auto iptr = input->writeMap<float>();
    auto tptr = timesteps->writeMap<int>();
    auto cptr = context->writeMap<float>();
    std::cout << iptr << std::endl;
    std::cout << tptr << std::endl;
    std::cout << cptr << std::endl;
    //output->setName("test_output");
    
    /*VARP input = _LinSpace(_Scalar<float>(0.0f), _Scalar<float>(1.0f), _Scalar<int>(int(batch_size*in_channels*height*width)));
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
    VARP output = model->onForward({input, timesteps, context})[0];*/
    
    /*input.fix(VARP::InputType::INPUT);
    timesteps.fix(VARP::InputType::INPUT);
    context.fix(VARP::InputType::INPUT);*/
    
    VARP output = model->onForward({input, timesteps, context})[0];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto rptr = output->readMap<float>();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;
    std::cout << "😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤TIME0:" << time.count() << "s" << std::endl;

    std::cout << rptr << std::endl;
    
    int n = 10;
    
    for (int i = 0; i < n; i++)
    {
        std::cout << rptr[i] << " ";
    }
    std::cout << std::endl;
    
    iptr = input->writeMap<float>();
    tptr = timesteps->writeMap<int>();
    cptr = context->writeMap<float>();
    
    tptr[0] = 3;
    
    start = std::chrono::high_resolution_clock::now();
    rptr = output->readMap<float>();
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    std::cout << "😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤TIME1:" << time.count() << "s" << std::endl;
    
    for (int i = 0; i < n; i++)
    {
        std::cout << rptr[i] << " ";
    }
    std::cout << std::endl;
    
    iptr = input->writeMap<float>();
    tptr = timesteps->writeMap<int>();
    cptr = context->writeMap<float>();
    tptr[0] = 4;
    start = std::chrono::high_resolution_clock::now();
    rptr = output->readMap<float>();
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    std::cout << "😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤TIME1:" << time.count() << "s" << std::endl;
    
    iptr = input->writeMap<float>();
    tptr = timesteps->writeMap<int>();
    cptr = context->writeMap<float>();
    tptr[0] = 2;
    start = std::chrono::high_resolution_clock::now();
    rptr = output->readMap<float>();
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    std::cout << "😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤TIME1:" << time.count() << "s" << std::endl;
    
    iptr = input->writeMap<float>();
    tptr = timesteps->writeMap<int>();
    cptr = context->writeMap<float>();
    tptr[0] = -2;
    start = std::chrono::high_resolution_clock::now();
    rptr = output->readMap<float>();
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    std::cout << "😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤TIME1:" << time.count() << "s" << std::endl;
    
    iptr = input->writeMap<float>();
    tptr = timesteps->writeMap<int>();
    cptr = context->writeMap<float>();
    tptr[0] = 5;
    start = std::chrono::high_resolution_clock::now();
    rptr = output->readMap<float>();
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    std::cout << "😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤😤TIME1:" << time.count() << "s" << std::endl;
    
    
    
    print_shape(output);
    
    return 0;
}