#include "test.h"
#include "tokenizer.hpp"
#include "parse_weighted_prompt.h"
#include "lora_loader.h"

//extern "C" void MNNSetCustomOpenCLLibraryPaths(const char** paths, int count);
int main()
{
    using namespace DonNotKnowHowToNameIt;
    using namespace MNN::Express;
    using namespace MyLDM;
    
    /*const char* cl_paths[] = {"/data/data/com.termux/files/usr/opt/vendor/lib/libOpenCL.so"};
    MNNSetCustomOpenCLLibraryPaths(cl_paths, 1);*/
    auto executor = Executor::getGlobalExecutor();
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    //backendConfig.flags = MNN_GPU_TUNING_NONE;
    executor->setGlobalExecutorConfig(MNN_FORWARD_CPU_EXTENSION, backendConfig, 8);
    
    int embed_dim = 4;
    int batch_size = 1;
    //int height = 8;
    //int width = 8;
    
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
        std::make_shared<MyAutoencoder>
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
        std::make_shared<MyUNetModel>
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
    
    SafetensorLoader loader("/storage/emulated/0/Download/Browser/anything-v5-PrtRE_f32.safetensors");
    SafetensorLoader lora_loader("/storage/emulated/0/test.safetensors");
    
    DDIMSampler model(vae, unet);
    
    model.load_from_safetensors(loader, "");
    auto ps = model.get_parameters_recursive();//for (auto m : ps) std::cout << m.first << std::endl;
    auto mapping = lora_load(ps, lora_loader);
        //MNN::DIFFUSION::CLIPTokenizer tokenizer;
    ExtendedCLIPTokenizer tokenizer;
    bool success = tokenizer.load("/storage/emulated/0/MyDiffusion/MyDiffusion/MyLDM/models/diffusion/tokenizer");
    if (!success)
    {
        std::cerr << "Tokenizer load failed!" << std::endl;
        return -1;
    }
    
    
    int maxlen = 77;
    //std::string pos_prompt = "An astronaut riding a horse on Mars";
    //std::string pos_prompt = "((masterpiece)), best quality, highres, extremely detailed, (((((a beautiful landscape with mountains and lake))))), sunrise, golden hour lighting, (((white clouds))), reflection on water, 8k, photorealistic, cinematic lighting, ((sharp focus))";
    std::string pos_prompt = "(1girl:1.3), (solo:1.2), (delicate face:1.2), (wind-blown hair:1.1), (natural lighting:1.2), (soft focus:1.1), (depth of field:1.1)";
    std::string neg_prompt = "(((low quality))), (((extra fingers))),(((fewer fingers))), (((worst quality))), normal quality, bad anatomy, bad proportions, extra limbs, missing arms, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, watermark, text, signature, out of frame, ugly face, distorted face";
    //std::string neg_prompt = "(((worst quality))), normal quality, bad anatomy, bad proportions, deformed, ugly, blurry, watermark, text, signature, out of frame";
    auto [ids, pos_weights_vec] = tokenizer.encode_weighted(pos_prompt, maxlen);
    auto [neg_ids, neg_weights_vec] = tokenizer.encode_weighted(neg_prompt, maxlen);
    
    std::copy(neg_ids.begin(), neg_ids.begin() + maxlen, ids.begin());
    std::copy(neg_weights_vec.begin(), neg_weights_vec.begin() + maxlen, pos_weights_vec.begin());
    
    VARP weights = _Const(pos_weights_vec.data(), {2, maxlen, 1}, NCHW, halide_type_of<float>());
    
    
    std::string text_encoder_model_path = "/storage/emulated/0/sd_mnn/text_encoder/text_encoder.mnn";
    
    std::shared_ptr<Module> text_encoder;
    text_encoder.reset
    (
        Module::load
        (
            {"input_ids"},
            {"last_hidden_state", "pooler_output"},
            text_encoder_model_path.c_str()
        )
    );
    
    VARP mPromptVar = _Input({2*maxlen}, NCHW, halide_type_of<int>());

    //std::vector<int> ids(2*mMaxTextLen, 0);
    
    memcpy((void *)mPromptVar->writeMap<int8_t>(), ids.data(), 2*maxlen*sizeof(int));
    
    auto outputs = text_encoder->onForward({mPromptVar});
    auto context = _Convert(outputs[0], NCHW) * weights;
    context.fix(VARP::CONSTANT);
    Module::destroy(text_encoder.get());
    
    print_shape(context);
    
    VARP x = _RandomNormal({1, 4, 64, 64}, halide_type_of<float>(), 0.0f, 1.0f, 404, 06);
    
    VARP output = model.sample(x, context, 7.5f, 5, 0.0f);
    
    output.fix(VARP::CONSTANT);
    
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    executor->setGlobalExecutorConfig(MNN_FORWARD_CPU_EXTENSION, backendConfig, 8);
    
    std::cout << "Decoding……" << std::endl;
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