#include "autoencoder.h"
#include "/storage/emulated/0/TEST/stb/ImageLoader.h"

using namespace DonNotKnowHowToNameIt;
using namespace MNN::Express;
using namespace MyLDM;

int main()
{
    /*const char* cl_paths[] = {"/data/data/com.termux/files/usr/opt/vendor/lib/libOpenCL.so"};
    MNNSetCustomOpenCLLibraryPaths(cl_paths, 1);*/
    auto executor = Executor::getGlobalExecutor();
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    //backendConfig.flags = MNN_GPU_TUNING_NONE;
    executor->setGlobalExecutorConfig(MNN_FORWARD_CPU_EXTENSION, backendConfig, 8);

    
    
    int in_channels = 3;
    int batch_size = 2;
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
    
    SafetensorLoader loader("/storage/emulated/0/Download/Browser/Counterfeit-V2.5.safetensors");
    model.load_from_safetensors(loader, "first_stage_model.");
    
    /*VARP input = _LinSpace(_Scalar(0.0f), _Scalar(1.0f), _Scalar(int(batch_size*in_channels*height*width)));
    
    input.fix(VARP::InputType::INPUT);
    
    input.setOrder(MNN::Express::Dimensionformat::NCHW);
    
    
    input = _Reshape(input, {batch_size, in_channels, height, width}, MNN::Express::Dimensionformat::NCHW);
    */
    
    
    VARP img = MyTensor::Core::load_from_image("/storage/emulated/0/TEST/test_output/result15.png");
    img = _Permute(img, {0, 3, 1, 2});
    img.fix(VARP::InputType::INPUT);
    img.setOrder(MNN::Express::Dimensionformat::NCHW);
    auto img_ptr = img->readMap<uint8_t>();
    auto img_info = img->getInfo();
    
    std::cout << img_info->dim.size() << std::endl;
    
    print_shape(img);
    
    VARP input = _Const(nullptr, img_info->dim, NCHW, halide_type_of<float>());
    input.fix(VARP::InputType::INPUT);
    
    auto input_ptr = input->writeMap<float>();
    for (int i = 0, n = img_info->size; i < n; i++)
    {
        input_ptr[i] = (static_cast<float>(img_ptr[i]) - 127.5f)/127.5f;
    }
    
    input->unMap();
    
    auto input_tensor = input->getTensor();
    
    input =
        _Slice
        (
            input,
            _Const
            (
                std::vector<int>({0, 0, 0, 0}).data(), {4}, NCHW, halide_type_of<int>()
            ),
            _Const
            (
                std::vector<int>({input_tensor->batch(), 3, input_tensor->height(), input_tensor->width()}).data(), {4}, NCHW, halide_type_of<int>()
            )
        );
    
    print_shape(input);
    VARP output = model.forward(input, false).first;
    
    output = _Minimum(_Maximum(output, _Scalar<float>(-1)), _Scalar<float>(1));
    output = _Permute(output, {0, 2, 3, 1});
    
    //std::vector<uint8_t> result(output->getInfo()->size);
    uint8_t* result = (uint8_t*)malloc(output->getInfo()->size*sizeof(uint8_t));
    
    auto rptr = output->readMap<float>();
    for (int i = 0, n = output->getTensor()->elementSize(); i < 10; i++)
    {
        std::cout << rptr[i] << " ";
    }
    std::cout << std::endl;
    
    for (int i = 0, n = output->getInfo()->size; i < n; i++)
    {
        result[i] = uint8_t(((rptr[i]) + 1.0f) * (255.0f / 2.0f));
    }
    
    for (int i = 0, n = output->getTensor()->elementSize(); i < 10; i++)
    {
        std::cout << int(result[i]) << " ";
    }
    std::cout << std::endl;
    
    const INTS& rdim = output->getInfo()->dim;
    
    int stride_bytes = rdim[2]*rdim[3]*sizeof(uint8_t);
    for (auto d : rdim) std::cout << d << " ";
    std::cout << std::endl;
    if (stbi_write_png("/storage/emulated/0/MyDiffusion/MyDiffusion/MyLDM/models/result.png", rdim[2], rdim[1], rdim[3], result, stride_bytes) == 0) throw std::runtime_error("保存失败");
    
    //if (stbi_write_png("/storage/emulated/0/TEST/stb/test_save.png", shape[2], shape[1], shape[3], ptr0, stride_bytes) == 0) throw std::runtime_error("保存失败");
    
    //pppp(output);
    
    
    
    
    
    free(result);
    print_shape(output);
    
    return 0;
}