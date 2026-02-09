from dataclasses import dataclass
import copy
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from easydict import EasyDict as edict

import diffusionGS
from diffusionGS.models.transformers.utils_transformer import DiTBlock,_init_weights
from diffusionGS.models.gsrenderer.renderer import Renderer
from diffusionGS.utils.base import BaseModule
from diffusionGS.models.denoiser.denoiser_utils import TimestepEmbedder, GaussiansUpsampler, ImageTokenDecoder

# Encoder
@diffusionGS.register("diffusion-gs-model")
class DGSDenoiser(BaseModule):
    r"""
    An image to gaussian diffusion models.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        use_downsample: bool = False
        num_latents: int = 256
        width: int = 1024
        in_channels: int = 3
        patch_size: int = 16
        n_gaussians: int = 2
        dim_heads: int = 64
        num_layers: int = 24
        ray_pe_type: str = "relative_plk"
        hard_pixelalign: bool = True
        clip_xyz: bool = True
        ##gaussian relative
        gaussians_sh_degree: int = 0
        use_gssplat: bool = False
        ##diffusion relative
        prior_distribution: str = "gaussian"
        ####
        use_flash: bool = False
        use_checkpoint: bool = True
        grad_checkpoint_every: int = 1

    cfg: Config
    def configure(self) -> None:
        super().configure()
        # time embedder
        self.t_embedder = TimestepEmbedder(self.cfg.width)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # image tokenlizer
        # 这是 patchify and linear layer
        # b v c h w 自动视为 b v c (hh ph) (ww pw)
        # b v c h w -> b*v n_patches width
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.cfg.patch_size,
                pw=self.cfg.patch_size,
            ),
            nn.Linear(
                self.cfg.patch_size * self.cfg.patch_size * self.cfg.in_channels,
                self.cfg.width,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)
        # gaussian pos embedding
        self.gaussians_pos_embedding = nn.Parameter(
            torch.randn(
                self.cfg.n_gaussians,     # 这里为什么额外加了2个 gaussians
                self.cfg.width,             # d = 1024
            )
        )
        nn.init.trunc_normal_(self.gaussians_pos_embedding, std=0.02)


        self.transformer_input_layernorm = nn.LayerNorm(
            self.cfg.width, bias=False
        )
        self.transformer = nn.ModuleList(
            [
                DiTBlock(
                    self.cfg.width, self.cfg.width // self.cfg.dim_heads
                )
                for _ in range(self.cfg.num_layers)
            ]
        )

        self.transformer.apply(_init_weights)
        self.upsampler = GaussiansUpsampler(self.cfg)
        self.upsampler.apply(_init_weights)     
        self.image_token_decoder = ImageTokenDecoder(self.cfg)
        self.image_token_decoder.apply(_init_weights)
        # encoder
        self.gs_renderer = Renderer(self.cfg)

        # renderer
        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained shape model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'model' in pretrained_ckpt.keys():
                pret_weights = pretrained_ckpt['model']
                _pretrained_ckpt = {}
                # breakpoint()
                for k, v in pret_weights.items():
                    if k.startswith('denoiser.') and not k.startswith('denoiser.loss_computer'):
                        _pretrained_ckpt[k.replace('denoiser.', '')] = v
                pretrained_ckpt = _pretrained_ckpt

            if 'state_dict' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['state_dict'].items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
            else:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt.items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
            self.load_state_dict(pretrained_ckpt, strict=True)

    def forward(self, input_batch, timesteps):
        guassians_parameters,_ = self.image_to_gaussians(input_batch['image'], input_batch['ray_o'], input_batch['ray_d'], timesteps)
        rendered_images = self.render_gaussians(guassians_parameters, input_batch['c2w'], input_batch['fxfycxcy'], input_batch['image'].shape[3], input_batch['image'].shape[4])
        return rendered_images, self.prepare_to_save(guassians_parameters)
    

    def prepare_to_save(self,gaussians_parameters):
        gaussians = []
        for i in range(gaussians_parameters.xyz.size(0)):
            self.gs_renderer.gaussians_model.empty()
            gaussians_model = copy.deepcopy(self.gs_renderer.gaussians_model)
            gaussians.append(
                gaussians_model.set_data(
                    gaussians_parameters.xyz[i].detach().float(),
                    gaussians_parameters.features[i].detach().float(),
                    gaussians_parameters.scaling[i].detach().float(),
                    gaussians_parameters.rotation[i].detach().float(),
                    gaussians_parameters.opacity[i].detach().float(),
                )
            )
        return gaussians

    def image_to_gaussians(self,
                        images:torch.FloatTensor,
                        ray_o:torch.FloatTensor,
                        ray_d:torch.FloatTensor,
                        t:torch.LongTensor,
                        training: bool = False):
        # 在 [-1, 1]
        if self.cfg.ray_pe_type == "relative_plk":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d           
            posed_images = torch.cat([images[:, :, :3, :, :] * 2.0 - 1.0,
                                      ray_d,
                                      nearest_pts,], dim=2)
        else:
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            posed_images = torch.cat([images[:, :, :3, :, :] * 2.0 - 1.0,
                                      o_cross_d,
                                      ray_d,], dim=2)
        b, v, c, h, w = posed_images.size()
        img_tokens = self.image_tokenizer(posed_images)
        t = self.t_embedder(t)  # [b, d]

        _, n_patches, d = img_tokens.size()  # [b*v, n_patches, d]
        img_tokens = img_tokens.reshape(b, v * n_patches, d)  # [b, v*n_patches, d]

        # [b, n_gaussians, d]
        gaussians_tokens = self.gaussians_pos_embedding.expand(b, -1, -1)   # 复制 b 份

        checkpoint_every = self.cfg.grad_checkpoint_every
        # [b, n_gaussians + v*n_patches, d]
        concat_nerf_img_tokens = torch.cat((gaussians_tokens, img_tokens), dim=1)

        # 归一化
        concat_nerf_img_tokens = self.transformer_input_layernorm(
            concat_nerf_img_tokens
        )
        # 前向时只保存输入和参数，反向时自动重算，用算力换显存
        for i in range(0, len(self.transformer), checkpoint_every):
            concat_nerf_img_tokens = torch.utils.checkpoint.checkpoint(
                self.run_layers(i, i + checkpoint_every),
                concat_nerf_img_tokens,
                t,
                use_reentrant=False,
            )
        gaussians_tokens, img_tokens = concat_nerf_img_tokens.split(
            [self.cfg.n_gaussians, v * n_patches], dim=1
        )
        gaussians = self.upsampler(gaussians_tokens, t)

        # [b, v*n_patches, p*p*gs]
        img_aligned_gaussians = self.image_token_decoder(img_tokens, t)
        img_aligned_gaussians = img_aligned_gaussians.reshape(
            b,
            -1,
            (3 + (self.cfg.gaussians_sh_degree + 1) ** 2 * 3 + 3 + 4 + 1),
        )  # [b, v*pixels, gs]
        n_img_aligned_gaussians = img_aligned_gaussians.size(1)
        all_gaussians = torch.cat((gaussians, img_aligned_gaussians), dim=1)
        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(all_gaussians)
        img_aligned_xyz = xyz[:, -n_img_aligned_gaussians:, :]  # 把 image 对应的 Gaussians 模型 xyz 取出来
        img_aligned_xyz = rearrange(
            img_aligned_xyz,
            "b (v n_patch_h n_patch_w patchsize_h patchsize_w) c -> b v c (n_patch_h patchsize_h) (n_patch_w patchsize_w)",
            v=v,
            n_patch_h=h // self.cfg.patch_size,
            n_patch_w=w // self.cfg.patch_size,
            patchsize_h=self.cfg.patch_size,
            patchsize_w=self.cfg.patch_size,
        )

        # 对 image_aligned_xyz 进行矫正
        if self.cfg.hard_pixelalign:
            depth_preact_bias = 0.
            img_aligned_xyz = torch.sigmoid(
                img_aligned_xyz.mean(dim=2, keepdim=True) + depth_preact_bias
            )
            # stx()
            # breakpoint()
            if self.cfg.ray_pe_type == 'relative_plk':
                # 深度范围扩展，经验系数
                img_aligned_xyz = (2.0 * img_aligned_xyz - 1.0) * 1.8 + o_dot_d
                # print(f"Using augmented plucker coordinates to compute xyz")
            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d
            # breakpoint()
            # 将坐标限制在 -1 到 1 之间
            if self.cfg.clip_xyz and training:
                img_aligned_xyz = img_aligned_xyz.clamp(-1.0, 1.0)

            img_aligned_xyz_reshape = rearrange(
                img_aligned_xyz,
                "b v c (n_patch_h patchsize_h) (n_patch_w patchsize_w) -> b (v n_patch_h n_patch_w patchsize_h patchsize_w) c",
                patchsize_h=self.cfg.patch_size,
                patchsize_w=self.cfg.patch_size,
            )
            xyz = torch.cat(
                (xyz[:, :-n_img_aligned_gaussians, :], img_aligned_xyz_reshape), dim=1
            )
            result_softpa = edict(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
            )

        # img_aligned_xyz的个数减去了 self.cfg.n_gaussians
        return result_softpa, img_aligned_xyz



    def render_gaussians(self, gaussian_params, c2w, fxfycxcy, height, width):
        # breakpoint()
        render_input = self.gs_renderer(
                    gaussian_params.xyz,
                    gaussian_params.features,
                    gaussian_params.scaling,
                    gaussian_params.rotation,
                    gaussian_params.opacity,
                    height,
                    width,
                    C2W=c2w,               # 
                    fxfycxcy=fxfycxcy,       # 
                )
        return render_input
    
    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype
    
    def run_layers(self, start, end):
        def custom_forward(concat_nerf_img_tokens, t):
            for i in range(start, min(end, len(self.transformer))):
                concat_nerf_img_tokens = self.transformer[i](concat_nerf_img_tokens, t)
            return concat_nerf_img_tokens

        return custom_forward