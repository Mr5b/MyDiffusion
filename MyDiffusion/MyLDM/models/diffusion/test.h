#ifndef MYLDM_SAMPLER_H
#define MYLDM_SAMPLER_H

#include "../autoencoder.h"
#include "/storage/emulated/0/MyDiffusion/MyDiffusion/MyLDM/modules/diffusionmodules/openaimodel.h"
#include <cmath>
#include "/storage/emulated/0/TEST/stb/ImageLoader.h"

namespace MyLDM
{
using namespace MNN::Express;


struct UNetWrapper : public DonNotKnowHowToNameIt::MyModule
{
};

struct AutoencoderWrapper : public DonNotKnowHowToNameIt::MyModule
{

    virtual VARP encode(VARP x){ return nullptr; }
    virtual VARP decode(VARP x) = 0;
    
};

struct DDIMSampler : public DonNotKnowHowToNameIt::MyModule
{
    std::shared_ptr<AutoencoderWrapper> first_stage_model_;
    std::shared_ptr<UNetWrapper> model_;

    float scale_factor_ = 0.18215f;
    int num_timesteps_ = 1000;
    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alphas_cumprod_;//Π(1->t)α

    DDIMSampler
    (
        std::shared_ptr<AutoencoderWrapper> ae,
        std::shared_ptr<UNetWrapper> unet,
        float scale_factor = 0.18215f,
        int num_timesteps = 1000,
        float linear_start = 0.00085f,
        float linear_end = 0.0120f
    ) :
        first_stage_model_(ae), model_(unet),
        scale_factor_(scale_factor), num_timesteps_(num_timesteps)
    {
        register_module("first_stage_model", first_stage_model_);
        register_module("model.diffusion_model", model_);
        compute_alphas_cumprod(linear_start, linear_end);
    }

    
    MNN::Express::VARP sample
    (
        const MNN::Express::INTS& shape,
        MNN::Express::VARP context = nullptr,
        int steps = 50,
        float eta = 0.0f
    )
    {
        using namespace MNN::Express;

        if (!context.get())
        {
            INTS ctx_shape = {shape[0], 77, 768};
            context = _Const(0.0f, ctx_shape, Dimensionformat::NCHW);
        }

        VARP x = _RandomNormal(shape, halide_type_of<float>(), 0.0f, 1.0f);

        std::vector<int> timesteps;
        int step_interval = num_timesteps_ / steps;
        for (int i = steps; i >= 1; i--)
        {
            timesteps.push_back(i * step_interval - 1);
        }
        timesteps.push_back(0);

        for (size_t idx = 0; idx < timesteps.size() - 1; idx++)
        {
            int t = timesteps[idx];
            int prev_t = timesteps[idx+1];

            VARP t_batch = _Const(static_cast<float>(t), {shape[0]}, Dimensionformat::NCHW);
            
            t_batch = _Cast<int32_t>(t_batch);
            
            VARP pred_noise = model_->onForward({x, t_batch, context})[0];
            
            x = ddim_step(x, pred_noise, t, prev_t, eta);
        }

        
        VARP z = x * _Scalar<float>(1.0f / scale_factor_);
        //VARP image = first_stage_model_->decode(z);

        return z;
    }


    /*MNN::Express::VARP sample
    (
        MNN::Express::VARP init_image,
        MNN::Express::VARP context = nullptr,
        float strength = 0.8f,
        int steps = 50,
        float eta = 0.0f
    )
    {
    
        using namespace MNN::Express;
        using namespace DonNotKnowHowToNameIt;
        
    //    AutoencoderKL::encode -> DiagonalGaussianDistribution
        std::cout << "init_image: " << std::endl;
        print_shape(init_image);
        DiagonalGaussianDistribution posterior =
            first_stage_model_->encode(init_image);
        VARP z0 = posterior.mode();                     //[1,4,H',W']
        z0 = z0 * _Scalar<float>(scale_factor_);
        
        if (!context.get())
        {
            INTS ctx_shape = {z0->getInfo()->dim[0], 77, 768};
            context = _Const(0.0f, ctx_shape, Dimensionformat::NCHW);
        }
        
        
        int t_start = static_cast<int>((1.0f - strength) * num_timesteps_);
        t_start = std::max(0, std::min(t_start, num_timesteps_ - 1));

        VARP noise = _RandomNormal(z0->getInfo()->dim, halide_type_of<float>(), 0.0f, 1.0f);
        float alpha_cumprod_t = alphas_cumprod_[t_start];
        float sqrt_alpha_cumprod = std::sqrt(alpha_cumprod_t);
        float sqrt_one_minus_alpha_var = std::sqrt(1.0f - alpha_cumprod_t);

        VARP sqrt_alpha_tensor = _Scalar<float>(sqrt_alpha_cumprod);
        VARP sqrt_one_minus_tensor = _Scalar<float>(sqrt_one_minus_alpha_var);

        VARP z_noisy = sqrt_alpha_tensor * z0 + sqrt_one_minus_tensor * noise;

    
        std::vector<int> timesteps;
        int step_interval = t_start / steps;
        if (step_interval < 1) step_interval = 1;

        for (int i = steps; i >= 1; i--)
        {
            int t = i * step_interval - 1;
            if (t > t_start) continue;
            timesteps.push_back(t);
        }
    
        if (timesteps.back() != 0) timesteps.push_back(0);

        VARP x = z_noisy;
        for (size_t idx = 0; idx < timesteps.size() - 1; idx++)
        {
            int t = timesteps[idx];
            int prev_t = timesteps[idx + 1];

            VARP t_batch = _Const(static_cast<float>(t), {z0->getInfo()->dim[0]}, Dimensionformat::NCHW);
            t_batch = _Cast<int32_t>(t_batch);

            VARP pred_noise = model_->onForward({x, t_batch, context})[0];
            x = ddim_step(x, pred_noise, t, prev_t, eta);
        }

        VARP z = x * _Scalar<float>(1.0f / scale_factor_);
        VARP output_image = first_stage_model_->decode(z);

        return output_image;
    }*/

    MNN::Express::VARP sample
    (
        //const MNN::Express::INTS& latent_shape,
        VARP x,
        MNN::Express::VARP emb_combined,
        /*MNN::Express::VARP positive_emb,
        MNN::Express::VARP negative_emb,*/
        float guidance_scale,
        int steps,
        float eta
    )
    {
        using namespace MNN::Express;
        
        auto x_info = x->getInfo();
        const MNN::Express::INTS& latent_shape = x_info->dim;
        int batch = latent_shape[0];
    
        /*MY_ASSERT
        (
            positive_emb->getInfo()->dim.size() == 3 &&
            positive_emb->getInfo()->dim[0] == batch &&,
            "positive_emb shape error");
        MY_ASSERT
        (
            negative_emb->getInfo()->dim.size() == 3 &&
            negative_emb->getInfo()->dim[0] == batch,
            "negative_emb shape error"
        );*/
        MY_ASSERT
        (
            emb_combined->getInfo()->dim.size() == 3 &&
            emb_combined->getInfo()->dim[0] == 2*batch,
            "emb shape error"
        );
        
        //VARP x = _RandomNormal(latent_shape, halide_type_of<float>(), 0.0f, 1.0f);

        
        std::vector<int> timesteps;
        int step_interval = num_timesteps_ / steps;
        for (int i = steps; i >= 1; i--)
        {
            timesteps.push_back(i * step_interval - 1);
        }
        timesteps.push_back(0);

        
        VARP positive_emb;
        bool use_cfg = (guidance_scale > 1.0f);
        if (!use_cfg)
        {
            positive_emb =
                _Slice
                (
                    emb_combined,
                    _Const
                    (
                        std::vector<int>({batch, 0, 0}).data(),
                        {3}, NCHW, halide_type_of<int>()
                    ),
                    _Const
                    (
                        std::vector<int>({batch, -1, -1}).data(),
                        {3}, NCHW, halide_type_of<int>()
                    )
                );
            
        }
        
        std::cout << (use_cfg ? std::string("CFG") : std::string("CFG关闭")) << std::endl;

        if (use_cfg)
        {
            VARP guidance_scale_var = _Scalar<float>(guidance_scale);
            for (size_t idx = 0; idx < timesteps.size() - 1; idx++)
            {
                int t = timesteps[idx];
                int prev_t = timesteps[idx + 1];

                VARP t_combined = _Const(static_cast<float>(t), {2*batch}, NCHW);
                t_combined = _Cast<int32_t>(t_combined);
                
                INTS x_shape = latent_shape;
                x_shape.insert(x_shape.begin(), 2);
                VARP x_combined =
                    _BroadcastTo
                    (
                        x,
                        _Const
                        (
                            std::vector<int>
                            (
                                x_shape
                            ).data(),
                            {5}, NCHW, halide_type_of<int>()
                        )
                    );
                x_shape.erase(x_shape.begin());
                x_shape[0] = 2 * x_shape[0];
                x_combined = _Reshape(x_combined, x_shape);// [2*batch, C, H, W]
            
                VARP noise_combined = model_->onForward({x_combined, t_combined, emb_combined})[0];
                
                VARP noise_uncond =
                    _Slice
                    (
                        noise_combined,
                        _Const
                        (
                            std::vector<int>({0, 0, 0, 0}).data(),
                            {4}, NCHW, halide_type_of<int>()
                        ),
                        _Const
                        (
                            std::vector<int>({batch, -1, -1, -1}).data(),
                            {4}, NCHW, halide_type_of<int>()
                        )
                    );
                
                VARP noise_cond =
                    _Slice
                    (
                        noise_combined,
                        _Const
                        (
                            std::vector<int>({batch, 0, 0, 0}).data(),
                            {4}, NCHW, halide_type_of<int>()
                        ),
                        _Const
                        (
                            std::vector<int>({batch, -1, -1, -1}).data(),
                            {4}, NCHW, halide_type_of<int>()
                        )
                    );
                
                VARP pred_noise = noise_uncond + guidance_scale_var * (noise_cond - noise_uncond);
                
                
                x = ddim_step(x, pred_noise, t, prev_t, eta);
            }
        }
        else
        {
            for (size_t idx = 0; idx < timesteps.size() - 1; idx++)
            {
                int t = timesteps[idx];
                int prev_t = timesteps[idx + 1];
                
                VARP t_batch = _Const(static_cast<float>(t), {batch}, NCHW);
                t_batch = _Cast<int32_t>(t_batch);
                
                VARP pred_noise = model_->onForward({x, t_batch, positive_emb})[0];
                x = ddim_step(x, pred_noise, t, prev_t, eta);
            }
        }

            
        
        VARP z = x * _Scalar<float>(1.0f / scale_factor_);
        //VARP image = first_stage_model_->decode(z);
        return z;
    }
    
    
    MNN::Express::VARP sample_low
    (
        VARP x,
        MNN::Express::VARP emb_combined,
        float guidance_scale,
        int steps,
        float eta
    )
    {
        using namespace MNN::Express;
        
        auto x_info = x->getInfo();
        const MNN::Express::INTS& latent_shape = x_info->dim;
        int batch = latent_shape[0];
    
        MY_ASSERT
        (
            emb_combined->getInfo()->dim.size() == 3 &&
            emb_combined->getInfo()->dim[0] == 2*batch,
            "emb shape error"
        );
        
        //VARP x = _RandomNormal(latent_shape, halide_type_of<float>(), 0.0f, 1.0f);

        
        std::vector<int> timesteps;
        int step_interval = num_timesteps_ / steps;
        for (int i = steps; i >= 1; i--)
        {
            timesteps.push_back(i * step_interval - 1);
        }
        timesteps.push_back(0);

        
        
        bool use_cfg = (guidance_scale > 1.0f);
        
        VARP positive_emb =
            _Slice
            (
                emb_combined,
                _Const
                (
                    std::vector<int>({batch, 0, 0}).data(),
                    {3}, NCHW, halide_type_of<int>()
                ),
                _Const
                (
                    std::vector<int>({batch, -1, -1}).data(),
                    {3}, NCHW, halide_type_of<int>()
                )
            );
        //positive_emb.fix(VARP::InputType::CONSTANT);
        
        VARP negative_emb =
            _Slice
            (
                emb_combined,
                _Const
                (
                    std::vector<int>({0, 0, 0}).data(),
                    {3}, NCHW, halide_type_of<int>()
                ),
                _Const
                (
                    std::vector<int>({batch, -1, -1}).data(),
                    {3}, NCHW, halide_type_of<int>()
                )
            );
        //negative_emb.fix(VARP::InputType::CONSTANT);
        
        std::cout << (use_cfg ? std::string("CFG") : std::string("CFG关闭")) << std::endl;

        if (use_cfg)
        {
            VARP guidance_scale_var = _Scalar<float>(guidance_scale);
            for (size_t idx = 0; idx < timesteps.size() - 1; idx++)
            {
                int t = timesteps[idx];
                int prev_t = timesteps[idx + 1];

                VARP t_batch = _Const(static_cast<float>(t), {batch}, NCHW);
                t_batch = _Cast<int32_t>(t_batch);
                
                //x.fix(VARP::InputType::CONSTANT);
                
                VARP noise_uncond = model_->onForward({x, t_batch, negative_emb})[0];
                //noise_uncond.fix(VARP::InputType::CONSTANT);
                
                VARP noise_cond = model_->onForward({x, t_batch, positive_emb})[0];
                //noise_cond.fix(VARP::InputType::CONSTANT);
                
                VARP pred_noise = noise_uncond + guidance_scale_var * (noise_cond - noise_uncond);
                
                
                x = ddim_step(x, pred_noise, t, prev_t, eta);
                //x.fix(VARP::InputType::CONSTANT);
                //std::cout << "step" << std::endl;
            }
        }
        else
        {
            for (size_t idx = 0; idx < timesteps.size() - 1; idx++)
            {
                int t = timesteps[idx];
                int prev_t = timesteps[idx + 1];
                
                VARP t_batch = _Const(static_cast<float>(t), {batch}, NCHW);
                t_batch = _Cast<int32_t>(t_batch);
                
                VARP pred_noise = model_->onForward({x, t_batch, positive_emb})[0];
                //pred_noise.fix(VARP::InputType::CONSTANT);
                
                x = ddim_step(x, pred_noise, t, prev_t, eta);
                //x.fix(VARP::InputType::CONSTANT);
            }
        }

            
        
        VARP z = x * _Scalar<float>(1.0f / scale_factor_);
        //VARP image = first_stage_model_->decode(z);
        return z;
    }
    
    
    MNN::Express::VARP sample
    (
        VARP x,
        MNN::Express::VARP emb_combined,
        float guidance_scale,
        float strength,
        int steps,
        float eta,
        VARP mask = nullptr
    )
    {
        using namespace MNN::Express;
        
        auto x_info = x->getInfo();
        const MNN::Express::INTS& latent_shape = x_info->dim;
        int batch = latent_shape[0];
    
        MY_ASSERT
        (
            emb_combined->getInfo()->dim.size() == 3 &&
            emb_combined->getInfo()->dim[0] == 2*batch,
            "emb shape error"
        );
        
        
        if (mask.get())
        {
            mask = _Interp({mask}, 0, 0, latent_shape[3], latent_shape[2], MNN::Express::InterpolationMethod::NEAREST, false);
            mask = _Maximum(_Minimum(mask, _Scalar<int>(1)), _Scalar<int>(0));
            /*auto ptr0 = mask->readMap<int>();
            for (int i = 0, n = mask->getTensor()->elementSize(); i < n; i++)
            {
                std::cout << ptr0[i] << " ";
            }*/
        }
        //VARP x = _RandomNormal(latent_shape, halide_type_of<float>(), 0.0f, 1.0f);
        x = x * _Scalar<float>(scale_factor_);
        
        VARP z_orig = x;
        
        int t_start = static_cast<int>((1.0f - strength) * num_timesteps_);
        t_start = std::max(0, std::min(t_start, num_timesteps_ - 1));
        
        VARP noise = _RandomNormal(latent_shape, halide_type_of<float>(), 0.0f, 1.0f);
        float alpha_cumprod_t = alphas_cumprod_[t_start];
        float sqrt_alpha_cumprod = std::sqrt(alpha_cumprod_t);
        float sqrt_one_minus_alpha = std::sqrt(1.0f - alpha_cumprod_t);

        VARP sqrt_alpha_cumprod_var = _Scalar<float>(sqrt_alpha_cumprod);
        VARP sqrt_one_minus_alpha_var = _Scalar<float>(sqrt_one_minus_alpha);

        x = sqrt_alpha_cumprod_var * x + sqrt_one_minus_alpha_var * noise;
        
        
        std::vector<int> timesteps;
        int step_interval = num_timesteps_ / steps;
        for (int i = steps; i >= 1; i--)
        {
            timesteps.push_back(i * step_interval - 1);
        }
        timesteps.push_back(0);

        
        VARP positive_emb;
        bool use_cfg = (guidance_scale > 1.0f);
        if (!use_cfg)
        {
            positive_emb =
                _Slice
                (
                    emb_combined,
                    _Const
                    (
                        std::vector<int>({batch, 0, 0}).data(),
                        {3}, NCHW, halide_type_of<int>()
                    ),
                    _Const
                    (
                        std::vector<int>({batch, -1, -1}).data(),
                        {3}, NCHW, halide_type_of<int>()
                    )
                );
            
        }
        
        std::cout << (use_cfg ? std::string("CFG") : std::string("CFG关闭")) << std::endl;

        if (use_cfg)
        {
            VARP guidance_scale_var = _Scalar<float>(guidance_scale);
            for (size_t idx = 0; idx < timesteps.size() - 1; idx++)
            {
                int t = timesteps[idx];
                int prev_t = timesteps[idx + 1];

                VARP t_combined = _Const(static_cast<float>(t), {2*batch}, NCHW);
                t_combined = _Cast<int32_t>(t_combined);
                
                INTS x_shape = latent_shape;
                x_shape.insert(x_shape.begin(), 2);
                VARP x_combined =
                    _BroadcastTo
                    (
                        x,
                        _Const
                        (
                            std::vector<int>
                            (
                                x_shape
                            ).data(),
                            {5}, NCHW, halide_type_of<int>()
                        )
                    );
                x_shape.erase(x_shape.begin());
                x_shape[0] = 2 * x_shape[0];
                x_combined = _Reshape(x_combined, x_shape);// [2*batch, C, H, W]
            
                VARP noise_combined = model_->onForward({x_combined, t_combined, emb_combined})[0];
                
                VARP noise_uncond =
                    _Slice
                    (
                        noise_combined,
                        _Const
                        (
                            std::vector<int>({0, 0, 0, 0}).data(),
                            {4}, NCHW, halide_type_of<int>()
                        ),
                        _Const
                        (
                            std::vector<int>({batch, -1, -1, -1}).data(),
                            {4}, NCHW, halide_type_of<int>()
                        )
                    );
                
                VARP noise_cond =
                    _Slice
                    (
                        noise_combined,
                        _Const
                        (
                            std::vector<int>({batch, 0, 0, 0}).data(),
                            {4}, NCHW, halide_type_of<int>()
                        ),
                        _Const
                        (
                            std::vector<int>({batch, -1, -1, -1}).data(),
                            {4}, NCHW, halide_type_of<int>()
                        )
                    );
                
                VARP pred_noise = noise_uncond + guidance_scale_var * (noise_cond - noise_uncond);
                
                x = ddim_step(x, pred_noise, t, prev_t, eta);
                
                if (mask.get())
                {
                    float alpha_cumprod_prev = alphas_cumprod_[prev_t];
                    float sqrt_alpha_prev = sqrt(alpha_cumprod_prev);
                    float sqrt_one_minus_alpha_prev = sqrt(1.0f - alpha_cumprod_prev);
                    VARP z_orig_noisy_prev = _Scalar<float>(sqrt_alpha_prev) * z_orig + _Scalar<float>(sqrt_one_minus_alpha_prev) * noise;
                    
                    x = _Select(mask, z_orig_noisy_prev, x);
                }
            }
        }
        else
        {
            for (size_t idx = 0; idx < timesteps.size() - 1; idx++)
            {
                int t = timesteps[idx];
                int prev_t = timesteps[idx + 1];
                
                VARP t_batch = _Const(static_cast<float>(t), {batch}, NCHW);
                t_batch = _Cast<int32_t>(t_batch);
                
                VARP pred_noise = model_->onForward({x, t_batch, positive_emb})[0];
                x = ddim_step(x, pred_noise, t, prev_t, eta);
                
                if (mask.get())
                {
                    
                    float alpha_cumprod_prev = alphas_cumprod_[prev_t];
                    float sqrt_alpha_prev = sqrt(alpha_cumprod_prev);
                    float sqrt_one_minus_alpha_prev = sqrt(1.0f - alpha_cumprod_prev);
                    VARP z_orig_noisy_prev = _Scalar<float>(sqrt_alpha_prev) * z_orig + _Scalar<float>(sqrt_one_minus_alpha_prev) * noise;
                    
                    x = _Select(mask, z_orig_noisy_prev, x);
                }
            }
        }

            
        
        VARP z = x * _Scalar<float>(1.0f / scale_factor_);
        //VARP image = first_stage_model_->decode(z);
        return z;
    }
    

    void compute_alphas_cumprod(float linear_start, float linear_end)
    {
        betas_.resize(num_timesteps_);
        alphas_.resize(num_timesteps_);
        alphas_cumprod_.resize(num_timesteps_);
        float d = linear_end - linear_start;
        
        for (int i = 0; i < num_timesteps_; i++)
        {
            float t = float(i) / (num_timesteps_ - 1);
            float beta = linear_start + d * t;
            betas_[i] = beta;
            alphas_[i] = 1.0f - beta;
            
            alphas_cumprod_[i] =
                (i == 0)
                ? alphas_[0] :
                alphas_cumprod_[i-1] * alphas_[i];
        }
    }
    
    
    MNN::Express::VARP ddim_step
    (
        MNN::Express::VARP x,
        MNN::Express::VARP pred_noise,
        int t, int prev_t,
        float eta
    )
    {
        using namespace MNN::Express;
        
        float alpha_cumprod_t = alphas_cumprod_[t];
        float alpha_cumprod_prev = (prev_t >= 0) ? alphas_cumprod_[prev_t] : 1.0f;
        float sqrt_alpha_cumprod = std::sqrt(alpha_cumprod_t);
        float sqrt_one_minus_alpha = std::sqrt(1.0f - alpha_cumprod_t);

        VARP alpha_cumprod_t_var = _Scalar<float>(alpha_cumprod_t);
        VARP alpha_cumprod_prev_var = _Scalar<float>(alpha_cumprod_prev);
        VARP sqrt_alpha_cumprod_var = _Scalar<float>(sqrt_alpha_cumprod);
        VARP sqrt_one_minus_alpha_var = _Scalar<float>(sqrt_one_minus_alpha);
        
        
        VARP x0_pred = (x - sqrt_one_minus_alpha_var * pred_noise) / sqrt_alpha_cumprod_var;

        VARP sigma_t_var =
            _Scalar<float>
            (
                eta
                 * std::sqrt
                   (
                       (1.0f - alpha_cumprod_prev) / (1.0f - alpha_cumprod_t)
                        * (1.0f - (alpha_cumprod_t / alpha_cumprod_prev))
                   )
            );

        VARP noise =
            eta > 0
            ? _RandomNormal(x->getInfo()->dim, halide_type_of<float>(), 0.0f, 1.0f)
            : _ZerosLike(x);
            
        VARP dir_xt = _Sqrt(_Scalar<float>(1.0f) - alpha_cumprod_prev_var - sigma_t_var * sigma_t_var) * pred_noise;
        VARP x_prev = _Sqrt(alpha_cumprod_prev_var) * x0_pred + dir_xt + sigma_t_var * noise;
        return x_prev;
    }
};


struct MyAutoencoder : public AutoencoderWrapper
{
    std::shared_ptr<AutoencoderKL> ae_;
    
    MyAutoencoder
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
    ) :
        ae_
        (
            std::make_shared<AutoencoderKL>
            (
                embed_dim,
                z_channels,
                ch,
                in_channels,
                num_res_blocks,
                resolution,
                std::move(attn_resolutions),
                tanh_out,
                resamp_with_conv,
                std::move(ch_mult),
                attn_type,
                dtype
            )
        )
    {}
    
    VARP encode(VARP x) override
    {
        return ae_->encode(x).mode();
    }
    
    VARP decode(VARP x) override
    {
        return ae_->decode(x);
    }
    
    virtual void load_from_safetensors
    (
        const SafetensorLoader& loader,
        const std::string& prefix = "",
        SafetensorLoader::ShapeMode shape_mode = SafetensorLoader::ShapeMode::LOOSE,
        DtypePolicy dtype_policy = DtypePolicy::AS_FILE,
        bool allow_missing_tensors = false
    ) override
    {
        ae_->load_from_safetensors
        (
            loader,
            prefix,
            shape_mode,
            dtype_policy,
            allow_missing_tensors
        );
    }
    
    virtual void get_parameters_recursive
    (
        std::unordered_map<std::string, MNN::Express::VARP>& map,
        const std::string& prefix = ""
    ) override
    {
        return ae_->get_parameters_recursive(map, prefix);
    }
};

struct MyUNetModel : public UNetWrapper
{
    std::shared_ptr<OpenaiModel::UNetModel> unet_;
    
    MyUNetModel
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
    ) :
        unet_
        (
            std::make_shared<OpenaiModel::UNetModel>
            (
                in_channels,
                model_channels,
                out_channels,
                num_res_blocks,
                num_heads,
                transformer_depth,
                context_dim,
                std::move(attention_resolutions),
                std::move(channel_mult),
                dtype
            )
        )
    {}
    
    
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        return unet_->onForward(inputs);
    }
    
    virtual void load_from_safetensors
    (
        const SafetensorLoader& loader,
        const std::string& prefix = "",
        SafetensorLoader::ShapeMode shape_mode = SafetensorLoader::ShapeMode::LOOSE,
        DtypePolicy dtype_policy = DtypePolicy::AS_FILE,
        bool allow_missing_tensors = false
    ) override
    {
        unet_->load_from_safetensors
        (
            loader,
            prefix,
            shape_mode,
            dtype_policy,
            allow_missing_tensors
        );
    }
    
    virtual void get_parameters_recursive
    (
        std::unordered_map<std::string, MNN::Express::VARP>& map,
        const std::string& prefix = ""
    ) override
    {
        return unet_->get_parameters_recursive(map, prefix);
    }
};


} // namespace MyLDM
#endif