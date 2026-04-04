#ifndef MYLDM_MODELS_AUTOENCODER
#define MYLDM_MODELS_AUTOENCODER

#include <MyModules.h>
#include <modules/attention.h>
#include <modules/distributions/distributions.h>
#include <modules/diffusionmodules/model.h>

namespace MyLDM
{

using namespace DonNotKnowHowToNameIt;
using namespace MNN::Express;

/*class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x*/
        
struct AutoencoderKL : public MyModule
{
    int embed_dim_;
    std::shared_ptr<MyModule> encoder_{nullptr};
    std::shared_ptr<MyModule> decoder_{nullptr};
    std::shared_ptr<MyModule> quant_conv_{nullptr};
    std::shared_ptr<MyModule> post_quant_conv_{nullptr};
    
    AutoencoderKL
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
        embed_dim_(embed_dim),
        quant_conv_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(2 * z_channels, 2 * embed_dim_, {1, 1})
                    .stride({1, 1})
                    .padding({0, 0}),
                Dimensionformat::NCHW,
                dtype
            )
        ),
        post_quant_conv_
        (
            std::make_shared<Conv2d>
            (
                ConvNdOptions<2>(embed_dim_, z_channels, {1, 1})
                    .stride({1, 1})
                    .padding({0, 0}),
                Dimensionformat::NCHW,
                dtype
            )
        ),
        encoder_
        (
            std::make_shared<Encoder>
            (
                ch,
                in_channels,
                num_res_blocks,
                resolution,
                z_channels,
                attn_resolutions,
                true,
                dtype,
                ch_mult,
                attn_type,
                resamp_with_conv
            )
        ),
        decoder_
        (
            std::make_shared<Decoder>
            (
                ch,
                in_channels,
                num_res_blocks,
                resolution,
                z_channels,
                false,
                tanh_out,
                attn_resolutions,
                dtype,
                ch_mult,
                attn_type,
                resamp_with_conv
            )
        )
    {
        register_module("quant_conv", quant_conv_);
        register_module("post_quant_conv", post_quant_conv_);
        register_module("encoder", encoder_);
        register_module("decoder", decoder_);
    }
    
    DiagonalGaussianDistribution encode(VARP x)
    {
        VARP h = this->encoder_->forward(x);
        VARP moments = quant_conv_->forward(h);
        DiagonalGaussianDistribution posterior(moments);
        return posterior;
    }
    
    VARP decode(VARP z)
    {
        z = post_quant_conv_->forward(z);
        VARP dec = decoder_->forward(z);
        return dec;
    }
    
    /*virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override
    {
        VARP input = inputs[0];
        
        DiagonalGaussianDistribution posterior = this->encode->forward(input);
        VARP z = posterior.mode();
        dec = this->decode->forward(z);
        return dec, posterior;
    }*/
    
    std::pair<VARP, DiagonalGaussianDistribution> forward(VARP input, bool sample_posterior)
    {
        DiagonalGaussianDistribution posterior =
            this->encode(input);
        VARP z;
        
        if (sample_posterior)
        {
            z = posterior.sample();
        }
        else
        {
            z = posterior.mode();
        }
        
        VARP dec = this->decode(z);
        return std::make_pair(dec, posterior);
    }
};


}//namespace MyLDM



#endif