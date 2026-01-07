import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from diffusionGS.pipline_obj import DiffusionGSPipeline
import torch
from torchvision.utils import save_image
pipeline = DiffusionGSPipeline.from_pretrained("./pretrained", device="cuda:0", torch_dtype=torch.float16)
gs_output = pipeline("assets/test_cases/monster.png",seed=62, foreground_ratio = 0.825, extract_mesh=True)#for the case that with pose
##export gaussians
gs_output.gaussians.save_ply("./pretrained/monster/test.ply")
##export image
save_image(gs_output.render_images, "./pretrained/monster/test.png")
##export mesh
gs_output.mesh.export("./pretrained/monster/test.obj")