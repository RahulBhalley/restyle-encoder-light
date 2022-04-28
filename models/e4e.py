"""
This file defines the core research contribution
"""
import math
import torch
from torch import nn

from models.stylegan2.model import Generator
from MobileStyleGAN.core.distiller import Distiller
from MobileStyleGAN.core.utils import load_cfg, load_weights
from configs.paths_config import model_paths
from models.encoders import restyle_e4e_encoders
from utils.model_utils import RESNET_MAPPING


class e4e(nn.Module):

    def __init__(self, opts):
        super(e4e, self).__init__()
        self.set_opts(opts)

        # Define `n_styles`.
        if self.opts.decoder_type == 'StyleGAN2':
            self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2 # 18
        elif self.opts.decoder_type == 'MobileStyleGAN':
            self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 + 3 # 23
        
        print(f'n_styles: {self.n_styles}')
        
        # Define encoder and decoder architectures.
        self.encoder = self.set_encoder()
        if self.opts.decoder_type == 'StyleGAN2':
            self.decoder = Generator(self.opts.output_size, 512, 8, channel_multiplier=2)
            # Load StyleGAN's weights, if needed.
            self.load_weights()
        elif self.opts.decoder_type == 'MobileStyleGAN':
            cfg_path = 'MobileStyleGAN/configs/mobile_stylegan_ffhq.json'
            cfg = load_cfg(cfg_path)
            self.decoder = Distiller(cfg)
            
            # Delete synthesis_net to save GPU RAM.
            del self.decoder.synthesis_net
            
            # Get latent Average.
            self.latent_avg = self.decoder.compute_mean_style(style_dim=512, wsize=self.n_styles)
            print(f"latent_avg shape: {self.latent_avg.shape}")
        
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        

    def set_encoder(self):
        if self.opts.encoder_type == 'ProgressiveBackboneEncoder':
            encoder = restyle_e4e_encoders.ProgressiveBackboneEncoder(50, 'ir_se', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ResNetProgressiveBackboneEncoder':
            encoder = restyle_e4e_encoders.ResNetProgressiveBackboneEncoder(self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ProgressiveBackboneEncoderLight':
            encoder = restyle_e4e_encoders.ProgressiveBackboneEncoderLight(50, 'ir_se', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ProgressiveBackboneEncoderLightPlus':
            encoder = restyle_e4e_encoders.ProgressiveBackboneEncoderLightPlus(50, 'ir', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ResNetProgressiveBackboneEncoderLight':
            encoder = restyle_e4e_encoders.ResNetProgressiveBackboneEncoderLight(self.n_styles, self.opts)
        else:
            raise Exception(f'{self.opts.encoder_type} is not a valid encoders')
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print(f'Loading ReStyle e4e from checkpoint: {self.opts.checkpoint_path}')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self.__get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            encoder_ckpt = self.__get_encoder_checkpoint()
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print(f'Loading decoder weights from pretrained path: {self.opts.stylegan_weights}')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
            self.__load_latent_avg(ckpt, repeat=self.n_styles)

    def forward(self, x, latent=None, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, average_code=False, input_is_full=False):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # residual step
            if x.shape[1] == 6 and latent is not None:
                # learn error with respect to previous iteration
                codes = codes + latent
            else:
                # first iteration is with respect to the avg latent code
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        if average_code:
            input_is_latent = True
        else:
            input_is_latent = (not input_code) or (input_is_full)
        
        print("Do we even run e4e?")
        print(f"self.opts.decoder_type: {self.opts.decoder_type}")

        if self.opts.decoder_type == 'StyleGAN2':
            print("We run StyleGAN2 function.")
            images, result_latent = self.decoder([codes],
                                                input_is_latent=input_is_latent,
                                                randomize_noise=randomize_noise,
                                                return_latents=return_latents)
        elif self.opts.decoder_type == 'MobileStyleGAN':
            print("We run MobileStyleGAN function.")
            print(f"return_latents: {return_latents}")
            codes.squeeze_(0)
            # print(f"codes: {codes.shape}")
            c:
                images, result_latent = self.decoder(codes,
                                                    return_latents=return_latents)
            else:
                images = self.decoder(codes,
                                        return_latents=return_latents)
            # print(type(images))
            # print(len(images))
            # for item in images:
            #     print(type(item), item.shape)
            
            # for idx, item in enumerate(images):
            #     print(f"item: {idx}", item)
            # images = out[0]
            # result_latent = out[1]

            # print(images[0])
            # print(images[1])

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def __get_encoder_checkpoint(self):
        if "ffhq" in self.opts.dataset_type:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
            if self.opts.input_nc != 3:
                shape = encoder_ckpt['input_layer.0.weight'].shape
                altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
                encoder_ckpt['input_layer.0.weight'] = altered_input_layer
            return encoder_ckpt
        else:
            print('Loading encoders weights from resnet34!')
            encoder_ckpt = torch.load(model_paths['resnet34'])
            # Transfer the RGB input of the resnet34 network to the first 3 input channels of pSp's encoder
            if self.opts.input_nc != 3:
                shape = encoder_ckpt['conv1.weight'].shape
                altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['conv1.weight']
                encoder_ckpt['conv1.weight'] = altered_input_layer
            mapped_encoder_ckpt = dict(encoder_ckpt)
            for p, v in encoder_ckpt.items():
                for original_name, psp_name in RESNET_MAPPING.items():
                    if original_name in p:
                        mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
                        mapped_encoder_ckpt.pop(p)
            return encoder_ckpt

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
