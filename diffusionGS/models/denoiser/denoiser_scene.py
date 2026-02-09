import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
from einops.layers.torch import Rearrange

import diffusionGS
from diffusionGS.models.gsrenderer.renderer import Renderer
from diffusionGS.models.transformers.utils_transformer import DiTBlock, _init_weights
from diffusionGS.utils.base import BaseModule
from diffusionGS.models.denoiser.denoiser_utils import TimestepEmbedder, GaussiansUpsampler, ImageTokenDecoder

# Encoder
@diffusionGS.register("diffusion-gs-model-scene")
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
        range_setting_type: str = "linear_depth"
        range_setting_near: int = 0 
        range_setting_far: int = 500 

    cfg: Config
    def configure(self) -> None:
        super().configure()
        ##
        ##time embedder
        self.t_embedder = TimestepEmbedder(self.cfg.width)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        ##image tokenlizer
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.cfg.patch_size,
                pw=self.cfg.patch_size,
            ),
            nn.Linear(
                self.cfg.in_channels
                * (self.cfg.patch_size**2),
                self.cfg.width,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)

        self.gaussians_pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.cfg.n_gaussians,     # 3d gaussian 的个数？n_gaussians = 2?
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
        self.gs_renderer = Renderer(self.cfg) #SceneRenderer
        # range func
        self.range_func = lambda t : torch.sigmoid(t) * (self.cfg.range_setting_far - self.cfg.range_setting_near) + self.cfg.range_setting_near  #get_point_range_func(self.cfg)
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
                        #
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
        for b in range(gaussians_parameters.xyz.size(0)):
            self.gs_renderer.gaussians_model.empty()
            gaussians_model = copy.deepcopy(self.gs_renderer.gaussians_model)
            gaussians.append(
                gaussians_model.set_data(
                    gaussians_parameters.xyz[b].detach().float(),
                    gaussians_parameters.features[b].detach().float(),
                    gaussians_parameters.scaling[b].detach().float(),
                    gaussians_parameters.rotation[b].detach().float(),
                    gaussians_parameters.opacity[b].detach().float(),
                )
            )
        return gaussians

    def image_to_gaussians(self,
                        images:torch.FloatTensor,
                        ray_o:torch.FloatTensor,
                        ray_d:torch.FloatTensor,
                        t:torch.LongTensor,
                        training: bool = False):
        # breakpoint()
        #do plucker pe
        if self.cfg.ray_pe_type == "relative_plk":
            # ray_o 和 ray_d 实际存储在一个 per-pixel ray direction map 上边, 为什么要将两者点乘后相加呢
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            # 这里假设 ray 是被归一化后的了
            # nearest_pts 表示离坐标系原点最近的点，平方求导
            nearest_pts = ray_o + o_dot_d * ray_d           
            posed_images = torch.cat(
                [
                    images[:, :, :3, :, :] * 2.0 - 1.0,
                    ray_d,
                    nearest_pts,
                ],
                dim=2,
            )
        else:
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            posed_images = torch.cat(
                [
                    images[:, :, :3, :, :] * 2.0 - 1.0,
                    o_cross_d,
                    ray_d,
                ],
                dim=2,
            )
        b, v, c, h, w = posed_images.size()
        img_tokens = self.image_tokenizer(posed_images)
        #img_tokens = torch.cat([self.cond_image_tokenizer(posed_images[:,:1]),self.image_tokenizer(posed_images[:,1:])],dim=0)
        # breakpoint()
        t = self.t_embedder(t)  # [b, d]

        _, n_patches, d = img_tokens.size()  # [b*v, n_patches, d]
        img_tokens = img_tokens.reshape(b, v * n_patches, d)  # [b, v*n_patches, d]

        # [b, n_gaussians, d]
        gaussians_tokens = self.gaussians_pos_embedding.expand(b, -1, -1)   # 复制 b 份

        # 表示每隔多少层去检查一次梯度，清空释放掉一些梯度
        checkpoint_every = self.cfg.grad_checkpoint_every
        concat_nerf_img_tokens = torch.cat((gaussians_tokens, img_tokens), dim=1)
        concat_nerf_img_tokens = self.transformer_input_layernorm(
            concat_nerf_img_tokens
        )
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

        # torch.cuda.synchronize()
        # toc = time.time()
        # print(f"input.image.size(): {input.image.size()}, time: {toc - tic:.3f} sec")

        all_gaussians = torch.cat((gaussians, img_aligned_gaussians), dim=1)
        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(all_gaussians)
        # gs post process
        # 选择最后 n_img_aligned_gaussians 个元素
        # n_img_aligned_gaussians = v * n_patches * patchsize * patchsize
        img_aligned_xyz = xyz[:, -n_img_aligned_gaussians:, :]  # 把 image 对应的 Gaussians 模型 xyz 取出来
        img_aligned_xyz = rearrange(
            img_aligned_xyz,
            "b (v h w ph pw) c -> b v c (h ph) (w pw)",
            v=v,
            h=h // self.cfg.patch_size,
            w=w // self.cfg.patch_size,
            ph=self.cfg.patch_size,
            pw=self.cfg.patch_size,
        )

        # 对 image_aligned_xyz 进行矫正
        if self.cfg.hard_pixelalign:
            img_aligned_xyz = img_aligned_xyz.mean(dim=2, keepdim=True)
            img_aligned_xyz = self.range_func(img_aligned_xyz)

            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d
            img_aligned_xyz_reshape = rearrange(
                img_aligned_xyz,
                "b v c (h ph) (w pw) -> b (v h w ph pw) c",
                ph=self.cfg.patch_size,
                pw=self.cfg.patch_size,
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


        return result_softpa, img_aligned_xyz



    def render_gaussians(self, gaussian_params, c2w, fxfycxcy, height, width):
        render_input = self.gs_renderer(
                    gaussian_params.xyz,
                    gaussian_params.features,
                    gaussian_params.scaling,
                    gaussian_params.rotation,
                    gaussian_params.opacity,
                    height,
                    width,
                    C2W=c2w,
                    fxfycxcy=fxfycxcy,
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